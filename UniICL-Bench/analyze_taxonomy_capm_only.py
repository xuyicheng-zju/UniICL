#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


BENCH_ROOT = Path(__file__).resolve().parent
OPEN_SOURCE_ROOT = BENCH_ROOT.parent
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))
if str(OPEN_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(OPEN_SOURCE_ROOT))
if str(OPEN_SOURCE_ROOT / "UniICL") not in sys.path:
    sys.path.insert(0, str(OPEN_SOURCE_ROOT / "UniICL"))

from analyze_taxonomy_features import (
    TAXONOMY_COLORS,
    TAXONOMY_ORDER,
    compute_cosine_matrix,
    compute_pearson_matrix,
    compute_same_diff_cosine_stats,
    compute_taxonomy_prototypes,
    compute_tsne_embedding,
    disable_external_judges_and_scorers,
    ensure_dir,
    l2_normalize,
    plot_heatmap,
    resolve_tasks,
    run_task,
    save_json,
    scatter_plot,
    truncate_capm_gates,
    write_metadata_jsonl,
    write_summary_csv,
)
from data.data_utils import patchify, pil_img2rgb
from eval_bagel import CAPM_ABLATION_CHOICES, SafeInterleaveInferencer, load_bagel_model
from public_path_config import DEFAULT_UNIICL_BASE_MODEL, DEFAULT_UNIICL_FINETUNED_MODEL


CAPM_FEATURE_ORDER = [
    "z_pool",
    "g_pool",
    "delta_pool",
    "p_pool",
]

DEFAULT_CENTER_FILTER_KEEP_PER_TAXONOMY = 250
DEFAULT_MARGIN_FILTER_KEEP_PER_TAXONOMY = 250
DEFAULT_MARGIN_FILTER_CANDIDATE_MULTIPLIER = 4


class CapmOnlyFeatureInferencer(SafeInterleaveInferencer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_task: Optional[str] = None
        self.current_taxonomy: Optional[str] = None
        self.episode_records: List[Dict] = []
        self._episode_counter = 0
        self._reset_episode_state()

    def set_run_context(self, task: str, taxonomy: str) -> None:
        self.current_task = task
        self.current_taxonomy = taxonomy

    def drain_episode_records(self) -> List[Dict]:
        records = self.episode_records
        self.episode_records = []
        return records

    def _reset_episode_state(self) -> None:
        self._episode_features: Dict[str, np.ndarray] = {}
        self._episode_error: Optional[str] = None
        self._episode_capm_available = False
        self._episode_demo_count = 0
        self._episode_tau: Dict[str, object] = {}

    @staticmethod
    def _new_turn() -> Dict[str, object]:
        return {
            "item_indices": [],
            "user_items": [],
            "assistant_items": [],
            "has_assistant": False,
            "assistant_open": False,
            "mode": "user",
        }

    @staticmethod
    def _has_turn_content(turn: Optional[Dict[str, object]]) -> bool:
        if turn is None:
            return False
        return bool(
            turn["user_items"] or turn["assistant_items"] or turn["has_assistant"]
        )

    @staticmethod
    def _append_item(turn: Dict[str, object], mode: str, item) -> None:
        key = "assistant_items" if mode == "assistant" else "user_items"
        if isinstance(item, str):
            text = item.strip()
            if not text:
                return
            turn[key].append(text)
        else:
            turn[key].append(item)
        if mode == "assistant":
            turn["assistant_open"] = False

    @staticmethod
    def _split_inline_assistant(payload: str) -> Tuple[str, Optional[str]]:
        for pattern in (r"\n\s*Assistant:\s*", r"\sAssistant:\s*"):
            match = re.search(pattern, payload, flags=re.IGNORECASE)
            if match:
                return payload[: match.start()], payload[match.end() :]
        return payload, None

    def _parse_turns(self, input_lists) -> List[Dict[str, object]]:
        turns: List[Dict[str, object]] = []
        cur_turn: Optional[Dict[str, object]] = None

        def finalize_turn() -> None:
            nonlocal cur_turn
            if self._has_turn_content(cur_turn):
                turns.append(cur_turn)
            cur_turn = None

        for idx, item in enumerate(input_lists):
            if isinstance(item, Image.Image):
                if cur_turn is None:
                    cur_turn = self._new_turn()
                cur_turn["item_indices"].append(idx)
                mode = str(cur_turn["mode"])
                self._append_item(cur_turn, mode, item)
                continue

            if not isinstance(item, str):
                continue

            text = item
            stripped = text.strip()
            if stripped == "":
                if cur_turn is not None:
                    cur_turn["item_indices"].append(idx)
                continue

            if re.match(r"^\s*User:\s*", text, flags=re.IGNORECASE):
                if self._has_turn_content(cur_turn):
                    finalize_turn()
                cur_turn = self._new_turn()
                cur_turn["item_indices"].append(idx)
                payload = re.sub(r"^\s*User:\s*", "", text, flags=re.IGNORECASE)
                user_part, assistant_part = self._split_inline_assistant(payload)
                self._append_item(cur_turn, "user", user_part)
                if assistant_part is not None:
                    cur_turn["has_assistant"] = True
                    cur_turn["mode"] = "assistant"
                    if assistant_part.strip():
                        self._append_item(cur_turn, "assistant", assistant_part)
                    else:
                        cur_turn["assistant_open"] = True
                continue

            if re.match(r"^\s*Assistant:\s*", text, flags=re.IGNORECASE):
                if cur_turn is None:
                    cur_turn = self._new_turn()
                cur_turn["item_indices"].append(idx)
                cur_turn["has_assistant"] = True
                cur_turn["mode"] = "assistant"
                payload = re.sub(r"^\s*Assistant:\s*", "", text, flags=re.IGNORECASE)
                if payload.strip():
                    self._append_item(cur_turn, "assistant", payload)
                else:
                    cur_turn["assistant_open"] = True
                continue

            if cur_turn is None:
                cur_turn = self._new_turn()
            cur_turn["item_indices"].append(idx)
            mode = str(cur_turn["mode"])
            self._append_item(cur_turn, mode, text)

        if self._has_turn_content(cur_turn):
            finalize_turn()

        return turns

    def _extract_demo_turns(self, input_lists) -> List[Dict[str, object]]:
        turns = self._parse_turns(input_lists)
        if len(turns) < 2:
            return []

        query_turn_idx = None
        for idx in range(len(turns) - 1, -1, -1):
            turn = turns[idx]
            if turn["has_assistant"] and turn["assistant_open"]:
                query_turn_idx = idx
                break
        if query_turn_idx is None:
            query_turn_idx = len(turns) - 1

        demo_turns: List[Dict[str, object]] = []
        for turn in turns[:query_turn_idx]:
            if turn["assistant_items"]:
                demo_turns.append(turn)
        return demo_turns

    @staticmethod
    def _build_generation_placeholder(input_lists) -> Optional[Image.Image]:
        last_image = None
        for item in input_lists:
            if isinstance(item, Image.Image):
                last_image = item
        if last_image is None:
            return None
        return last_image.copy()

    @torch.no_grad()
    def _encode_segment_items(
        self,
        items: List[object],
        segment_id: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        embeds_parts: List[torch.Tensor] = []
        seg_parts: List[torch.Tensor] = []

        for item in items:
            if isinstance(item, Image.Image):
                image_pil = self.vae_transform.resize_transform(pil_img2rgb(item))
                image_tensor = self.vit_transform(image_pil)
                vit_position_ids = self.model.get_flattened_position_ids(
                    image_tensor.size(1),
                    image_tensor.size(2),
                    self.model.vit_patch_size,
                    max_num_patches_per_side=self.model.vit_max_num_patch_per_side,
                )
                vit_tokens = patchify(image_tensor, self.model.vit_patch_size)
                vit_tokens = vit_tokens.to(device=self.device)
                vit_position_ids = vit_position_ids.to(device=self.device)
                cu_seqlens = torch.tensor(
                    [0, vit_tokens.shape[0]],
                    dtype=torch.int32,
                    device=self.device,
                )

                with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                    vit_embed = self.model.vit_model(
                        packed_pixel_values=vit_tokens,
                        packed_flattened_position_ids=vit_position_ids,
                        cu_seqlens=cu_seqlens,
                        max_seqlen=vit_tokens.shape[0],
                    )
                    vit_embed = self.model.connector(vit_embed)
                    pos_emb = self.model.vit_pos_embed(vit_position_ids)
                    vit_embed = (vit_embed + pos_emb).to(torch.bfloat16)

                embeds_parts.append(vit_embed)
                seg_parts.append(
                    torch.full(
                        (vit_embed.shape[0],),
                        segment_id,
                        device=self.device,
                        dtype=torch.long,
                    )
                )
                continue

            if isinstance(item, str):
                text = item.strip()
                if not text:
                    continue
                token_ids = self.tokenizer.encode(text)
                if not token_ids:
                    continue
                token_ids_t = torch.tensor(token_ids, dtype=torch.long, device=self.device)
                with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                    token_embed = self.model.language_model.model.embed_tokens(token_ids_t)
                embeds_parts.append(token_embed)
                seg_parts.append(
                    torch.full(
                        (len(token_ids),),
                        segment_id,
                        device=self.device,
                        dtype=torch.long,
                    )
                )

        return embeds_parts, seg_parts

    @torch.no_grad()
    def _build_demo_inputs(self, turn: Dict[str, object]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        user_embeds, user_seg = self._encode_segment_items(list(turn["user_items"]), segment_id=0)
        assistant_embeds, assistant_seg = self._encode_segment_items(
            list(turn["assistant_items"]),
            segment_id=1,
        )
        if not user_embeds or not assistant_embeds:
            return None

        demo_embed = torch.cat(user_embeds + assistant_embeds, dim=0).unsqueeze(0)
        demo_seg = torch.cat(user_seg + assistant_seg, dim=0).unsqueeze(0)
        return demo_embed, demo_seg

    @torch.no_grad()
    def _compute_operator_outputs(
        self,
        c_in: torch.Tensor,
        c_out: torch.Tensor,
        g: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        operator = self.model.capm.encoder.operator
        if getattr(self.model.capm.config, "ablation_mode", "none") == "no_low_rank_transformation":
            zeros_p = torch.zeros(
                g.shape[0],
                self.model.capm.config.operator_rank,
                device=g.device,
                dtype=g.dtype,
            )
            zeros_delta = torch.zeros_like(g)
            return zeros_p, zeros_delta, g

        diff = c_out - c_in
        prod = c_in * c_out
        feat_raw = torch.cat([c_in, c_out, diff, prod], dim=-1)
        feat = operator.feat_norm(feat_raw)
        scales = operator.head_net(feat) * operator.op_gain
        rank = scales.shape[-1] // 3
        u_scale, v_scale, alpha = scales.split(rank, dim=-1)

        U = operator.U_base.unsqueeze(0) * u_scale.unsqueeze(1)
        V = operator.V_base.unsqueeze(0) * v_scale.unsqueeze(1)
        p = torch.einsum("bdr,bd->br", V, g)
        p = p * alpha
        delta = torch.einsum("bdr,br->bd", U, p)
        z = g + delta
        return p, delta, z

    @torch.no_grad()
    def _compute_tau_outputs(self, z_pool: torch.Tensor) -> Dict[str, object]:
        if self.model.capm is None:
            return {}

        aligner = getattr(self.model.capm, "aligner", None)
        if aligner is None:
            return {}

        fixed_tau = getattr(self.model.capm.config, "fixed_tau", None)
        tau_logit = aligner.tau_logit.detach().to(torch.float32)
        tau_base = aligner.tau_min + (aligner.tau_max - aligner.tau_min) * torch.sigmoid(
            tau_logit
        )
        tau_base_value = float(tau_base.reshape(-1)[0].cpu().item())

        if fixed_tau is not None:
            return {
                "tau_value": float(fixed_tau),
                "tau_base": tau_base_value,
                "delta_tau": None,
                "tau_ratio": None,
                "tau_is_fixed": True,
                "tau_fixed": float(fixed_tau),
            }

        target_device = aligner.tau_logit.device
        head_param = next(aligner.tau_head.parameters(), None)
        target_dtype = head_param.dtype if head_param is not None else z_pool.dtype
        z_for_tau = z_pool.detach().to(device=target_device, dtype=target_dtype)

        delta_tau = aligner.tau_head(z_for_tau)
        delta_tau = 0.25 * torch.tanh(delta_tau)
        tau_ratio = torch.sigmoid(
            aligner.tau_logit.to(device=delta_tau.device, dtype=delta_tau.dtype) + delta_tau
        )
        tau = aligner.tau_min + (aligner.tau_max - aligner.tau_min) * tau_ratio

        return {
            "tau_value": float(tau.reshape(-1)[0].detach().to(torch.float32).cpu().item()),
            "tau_base": tau_base_value,
            "delta_tau": float(
                delta_tau.reshape(-1)[0].detach().to(torch.float32).cpu().item()
            ),
            "tau_ratio": float(
                tau_ratio.reshape(-1)[0].detach().to(torch.float32).cpu().item()
            ),
            "tau_is_fixed": False,
            "tau_fixed": None,
        }

    @torch.no_grad()
    def _encode_capm_demo_turns(self, demo_turns: List[Dict[str, object]]) -> None:
        if not demo_turns or self.model.capm is None:
            return

        all_z: List[torch.Tensor] = []
        all_g: List[torch.Tensor] = []
        all_delta: List[torch.Tensor] = []
        all_p: List[torch.Tensor] = []

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            for turn in demo_turns:
                built = self._build_demo_inputs(turn)
                if built is None:
                    continue
                demo_embed, demo_seg = built
                c_in, c_out, facets = self.model.capm.encoder.prober(demo_embed, demo_seg)
                g = self.model.capm.encoder.pooler(facets)
                if self.model.capm.encoder.detach_operator_from_facets:
                    g_for_op = g.detach()
                else:
                    g_for_op = g

                if getattr(self.model.capm.config, "ablation_mode", "none") == "no_decoupled_encoding":
                    c_in = g_for_op
                    c_out = g_for_op

                p, delta, z = self._compute_operator_outputs(c_in, c_out, g_for_op)
                all_g.append(g_for_op)
                all_p.append(p)
                all_delta.append(delta)
                all_z.append(z)

        if not all_z:
            return

        all_z = self.model.capm.demo_interaction(all_z)
        z_pool = torch.stack(all_z, dim=1).mean(dim=1)
        g_pool = torch.stack(all_g, dim=1).mean(dim=1)
        delta_pool = torch.stack(all_delta, dim=1).mean(dim=1)
        p_pool = torch.stack(all_p, dim=1).mean(dim=1)

        self._episode_capm_available = True
        self._episode_demo_count = len(all_z)
        self._episode_features["z_pool"] = z_pool[0].detach().to(torch.float32).cpu().numpy()
        self._episode_features["g_pool"] = g_pool[0].detach().to(torch.float32).cpu().numpy()
        self._episode_features["delta_pool"] = delta_pool[0].detach().to(torch.float32).cpu().numpy()
        self._episode_features["p_pool"] = p_pool[0].detach().to(torch.float32).cpu().numpy()
        self._episode_tau = self._compute_tau_outputs(z_pool)

    def interleave_inference(self, *args, **kwargs):
        self._reset_episode_state()
        self._episode_counter += 1
        input_lists = kwargs.get("input_lists")
        if input_lists is None and args:
            input_lists = args[0]
        understanding_output = bool(kwargs.get("understanding_output", False))

        try:
            if input_lists is not None:
                demo_turns = self._extract_demo_turns(input_lists)
                self._encode_capm_demo_turns(demo_turns)
            if understanding_output:
                return [""]
            return [self._build_generation_placeholder(input_lists)]
        except Exception as exc:
            self._episode_error = str(exc)
            raise
        finally:
            self.episode_records.append(
                {
                    "episode_index": self._episode_counter,
                    "task": self.current_task,
                    "taxonomy": self.current_taxonomy,
                    "capm_available": self._episode_capm_available,
                    "demo_count": self._episode_demo_count,
                    "episode_error": self._episode_error,
                    "tau": dict(self._episode_tau),
                    "features": dict(self._episode_features),
                }
            )


def feature_dims_from_model(model) -> Dict[str, int]:
    d_capm = model.capm.config.d_capm if model.capm is not None else 0
    rank = model.capm.config.operator_rank if model.capm is not None else 0
    return {
        "z_pool": d_capm,
        "g_pool": d_capm,
        "delta_pool": d_capm,
        "p_pool": rank,
    }


def build_feature_arrays(
    records: Sequence[Dict],
    feature_dims: Dict[str, int],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[Dict]]:
    arrays: Dict[str, np.ndarray] = {}
    masks: Dict[str, np.ndarray] = {}
    for feature_name, dim in feature_dims.items():
        if dim <= 0:
            continue
        arrays[feature_name] = np.full((len(records), dim), np.nan, dtype=np.float32)
        masks[feature_name] = np.zeros(len(records), dtype=bool)

    metadata: List[Dict] = []
    for row_idx, record in enumerate(records):
        rec_features = record.get("features", {})
        for feature_name, vec in rec_features.items():
            if feature_name not in arrays or vec is None:
                continue
            vec = np.asarray(vec, dtype=np.float32)
            if vec.shape[0] != arrays[feature_name].shape[1]:
                raise ValueError(
                    f"Feature dim mismatch for {feature_name}: expected "
                    f"{arrays[feature_name].shape[1]}, got {vec.shape[0]}"
                )
            arrays[feature_name][row_idx] = vec
            masks[feature_name][row_idx] = True

        metadata.append(
            {
                "episode_index": int(record["episode_index"]),
                "task": record["task"],
                "taxonomy": record["taxonomy"],
                "sample_id": record["sample_id"],
                "capm_available": bool(record["capm_available"]),
                "demo_count": int(record["demo_count"]),
                "inference_failed": bool(record["inference_failed"]),
                "episode_error": record["episode_error"],
                "tau_value": record.get("tau", {}).get("tau_value"),
                "tau_base": record.get("tau", {}).get("tau_base"),
                "delta_tau": record.get("tau", {}).get("delta_tau"),
                "tau_ratio": record.get("tau", {}).get("tau_ratio"),
                "tau_is_fixed": bool(record.get("tau", {}).get("tau_is_fixed", False)),
                "tau_fixed": record.get("tau", {}).get("tau_fixed"),
            }
        )
    return arrays, masks, metadata


def summarize_tau_records(records: Sequence[Dict]) -> Dict[str, object]:
    tau_values = [
        float(record["tau"]["tau_value"])
        for record in records
        if record.get("tau", {}).get("tau_value") is not None
    ]
    tau_base_values = [
        float(record["tau"]["tau_base"])
        for record in records
        if record.get("tau", {}).get("tau_base") is not None
    ]
    delta_tau_values = [
        float(record["tau"]["delta_tau"])
        for record in records
        if record.get("tau", {}).get("delta_tau") is not None
    ]
    tau_ratio_values = [
        float(record["tau"]["tau_ratio"])
        for record in records
        if record.get("tau", {}).get("tau_ratio") is not None
    ]
    tau_fixed_count = sum(
        bool(record.get("tau", {}).get("tau_is_fixed", False))
        for record in records
        if record.get("tau", {}).get("tau_value") is not None
    )

    def stats(values: List[float], prefix: str) -> Dict[str, object]:
        if not values:
            return {
                f"{prefix}_count": 0,
                f"{prefix}_mean": None,
                f"{prefix}_std": None,
                f"{prefix}_min": None,
                f"{prefix}_max": None,
            }
        arr = np.asarray(values, dtype=np.float32)
        return {
            f"{prefix}_count": int(arr.size),
            f"{prefix}_mean": float(arr.mean()),
            f"{prefix}_std": float(arr.std()),
            f"{prefix}_min": float(arr.min()),
            f"{prefix}_max": float(arr.max()),
        }

    summary = {
        "tau_fixed_count": int(tau_fixed_count),
    }
    summary.update(stats(tau_values, "tau"))
    summary.update(stats(tau_base_values, "tau_base"))
    summary.update(stats(delta_tau_values, "delta_tau"))
    summary.update(stats(tau_ratio_values, "tau_ratio"))
    return summary


def ordered_present_taxonomies(labels: Sequence[str]) -> List[str]:
    present = set(labels)
    ordered = [label for label in TAXONOMY_ORDER if label in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def compute_taxonomy_selection_metrics(
    x: np.ndarray,
    labels: Sequence[str],
) -> Dict[str, object]:
    if x.shape[0] != len(labels):
        raise ValueError("Feature rows and label count must match")

    x_norm = l2_normalize(x)
    ordered_labels, prototypes = compute_taxonomy_prototypes(x_norm, labels)
    proto_map = {label: prototypes[idx] for idx, label in enumerate(ordered_labels)}

    radii = np.zeros(x.shape[0], dtype=np.float32)
    margins = np.zeros(x.shape[0], dtype=np.float32)
    nearest_other: List[Optional[str]] = []

    for idx, label in enumerate(labels):
        sample = x_norm[idx]
        own_dist = float(np.linalg.norm(sample - proto_map[label]))
        min_other_dist = float("inf")
        min_other_label: Optional[str] = None
        for other_label in ordered_labels:
            if other_label == label:
                continue
            dist = float(np.linalg.norm(sample - proto_map[other_label]))
            if dist < min_other_dist:
                min_other_dist = dist
                min_other_label = other_label

        radii[idx] = own_dist
        if min_other_label is None:
            margins[idx] = 0.0
            nearest_other.append(None)
        else:
            margins[idx] = min_other_dist - own_dist
            nearest_other.append(min_other_label)

    return {
        "radii": radii,
        "margins": margins,
        "nearest_other_taxonomy": nearest_other,
        "ordered_taxonomies": ordered_labels,
    }


def select_center_filtered_indices(
    labels: Sequence[str],
    radii: np.ndarray,
    margins: np.ndarray,
    keep_per_taxonomy: int,
) -> np.ndarray:
    if keep_per_taxonomy <= 0:
        return np.zeros(0, dtype=np.int32)

    labels_arr = np.asarray(labels)
    selected: List[np.ndarray] = []
    for label in ordered_present_taxonomies(labels):
        label_indices = np.flatnonzero(labels_arr == label)
        if label_indices.size == 0:
            continue
        order = np.lexsort((-margins[label_indices], radii[label_indices]))
        selected.append(label_indices[order[:keep_per_taxonomy]])

    if not selected:
        return np.zeros(0, dtype=np.int32)
    return np.concatenate(selected).astype(np.int32, copy=False)


def select_margin_filtered_indices(
    labels: Sequence[str],
    radii: np.ndarray,
    margins: np.ndarray,
    keep_per_taxonomy: int,
    candidate_multiplier: int,
) -> np.ndarray:
    if keep_per_taxonomy <= 0:
        return np.zeros(0, dtype=np.int32)

    labels_arr = np.asarray(labels)
    candidate_keep = max(keep_per_taxonomy, keep_per_taxonomy * max(candidate_multiplier, 1))
    center_pool = select_center_filtered_indices(
        labels=labels,
        radii=radii,
        margins=margins,
        keep_per_taxonomy=candidate_keep,
    )

    selected: List[np.ndarray] = []
    for label in ordered_present_taxonomies(labels):
        label_indices = center_pool[labels_arr[center_pool] == label]
        if label_indices.size == 0:
            continue
        order = np.lexsort((radii[label_indices], -margins[label_indices]))
        selected.append(label_indices[order[:keep_per_taxonomy]])

    if not selected:
        return np.zeros(0, dtype=np.int32)
    return np.concatenate(selected).astype(np.int32, copy=False)


def build_visual_selection_summary_row(
    feature_name: str,
    view: str,
    selected_indices: np.ndarray,
    x: np.ndarray,
    labels: Sequence[str],
    keep_per_taxonomy: Optional[int],
) -> Dict[str, object]:
    selected_labels = [labels[idx] for idx in selected_indices]
    counts_by_taxonomy = {
        taxonomy: int(sum(label == taxonomy for label in selected_labels))
        for taxonomy in ordered_present_taxonomies(labels)
    }
    selected_x = x[selected_indices]
    same_mean, diff_mean, gap = compute_same_diff_cosine_stats(selected_x, selected_labels)
    return {
        "feature": feature_name,
        "view": view,
        "selected_samples": int(selected_indices.size),
        "taxonomy_count": len(set(selected_labels)),
        "target_keep_per_taxonomy": keep_per_taxonomy,
        "min_selected_per_taxonomy": min(counts_by_taxonomy.values()) if counts_by_taxonomy else 0,
        "max_selected_per_taxonomy": max(counts_by_taxonomy.values()) if counts_by_taxonomy else 0,
        "same_taxonomy_cosine_mean": same_mean,
        "diff_taxonomy_cosine_mean": diff_mean,
        "same_minus_diff_gap": gap,
    }


def analyze_features(
    arrays: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    metadata: Sequence[Dict],
    output_dir: Path,
) -> None:
    plots_dir = ensure_dir(output_dir / "plots")
    summary_rows: List[Dict[str, object]] = []
    taxonomy_count_rows: List[Dict[str, object]] = []
    visual_selection_rows: List[Dict[str, object]] = []

    task_labels_all = [row["task"] for row in metadata]
    task_color_map = {}
    from matplotlib import pyplot as plt

    cmap = plt.get_cmap("tab20")
    for idx, task in enumerate(sorted(set(task_labels_all))):
        task_color_map[task] = cmap(idx % 20)

    for taxonomy in TAXONOMY_ORDER:
        row = {"taxonomy": taxonomy, "sample_count": sum(m["taxonomy"] == taxonomy for m in metadata)}
        for feature_name in CAPM_FEATURE_ORDER:
            if feature_name in masks:
                row[f"{feature_name}_valid"] = int(
                    sum(
                        bool(mask) and metadata[idx]["taxonomy"] == taxonomy
                        for idx, mask in enumerate(masks[feature_name])
                    )
                )
        taxonomy_count_rows.append(row)

    for feature_name in CAPM_FEATURE_ORDER:
        if feature_name not in arrays:
            continue
        valid_mask = masks[feature_name]
        if valid_mask.sum() < 2:
            continue

        x = arrays[feature_name][valid_mask]
        labels = [metadata[idx]["taxonomy"] for idx, flag in enumerate(valid_mask) if flag]
        tasks = [metadata[idx]["task"] for idx, flag in enumerate(valid_mask) if flag]

        tsne_points = compute_tsne_embedding(x)
        scatter_plot(
            tsne_points,
            labels,
            TAXONOMY_COLORS,
            f"{feature_name} t-SNE by Taxonomy",
            plots_dir / f"{feature_name}_tsne_taxonomy.png",
        )
        scatter_plot(
            tsne_points,
            labels,
            TAXONOMY_COLORS,
            f"{feature_name} t-SNE by Taxonomy (Raw)",
            plots_dir / f"{feature_name}_tsne_taxonomy_raw.png",
        )
        scatter_plot(
            tsne_points,
            tasks,
            task_color_map,
            f"{feature_name} t-SNE by Task",
            plots_dir / f"{feature_name}_tsne_task.png",
        )

        selection_metrics = compute_taxonomy_selection_metrics(x, labels)
        radii = selection_metrics["radii"]
        margins = selection_metrics["margins"]

        center_filtered_indices = select_center_filtered_indices(
            labels=labels,
            radii=radii,
            margins=margins,
            keep_per_taxonomy=DEFAULT_CENTER_FILTER_KEEP_PER_TAXONOMY,
        )
        if center_filtered_indices.size >= 2:
            scatter_plot(
                tsne_points[center_filtered_indices],
                [labels[idx] for idx in center_filtered_indices],
                TAXONOMY_COLORS,
                f"{feature_name} t-SNE by Taxonomy (Center Filtered)",
                plots_dir / f"{feature_name}_tsne_taxonomy_center_filtered.png",
            )
            visual_selection_rows.append(
                build_visual_selection_summary_row(
                    feature_name=feature_name,
                    view="center_filtered",
                    selected_indices=center_filtered_indices,
                    x=x,
                    labels=labels,
                    keep_per_taxonomy=DEFAULT_CENTER_FILTER_KEEP_PER_TAXONOMY,
                )
            )

        margin_filtered_indices = select_margin_filtered_indices(
            labels=labels,
            radii=radii,
            margins=margins,
            keep_per_taxonomy=DEFAULT_MARGIN_FILTER_KEEP_PER_TAXONOMY,
            candidate_multiplier=DEFAULT_MARGIN_FILTER_CANDIDATE_MULTIPLIER,
        )
        if margin_filtered_indices.size >= 2:
            scatter_plot(
                tsne_points[margin_filtered_indices],
                [labels[idx] for idx in margin_filtered_indices],
                TAXONOMY_COLORS,
                f"{feature_name} t-SNE by Taxonomy (Margin Filtered)",
                plots_dir / f"{feature_name}_tsne_taxonomy_margin_filtered.png",
            )
            visual_selection_rows.append(
                build_visual_selection_summary_row(
                    feature_name=feature_name,
                    view="margin_filtered",
                    selected_indices=margin_filtered_indices,
                    x=x,
                    labels=labels,
                    keep_per_taxonomy=DEFAULT_MARGIN_FILTER_KEEP_PER_TAXONOMY,
                )
            )

        proto_labels, prototypes = compute_taxonomy_prototypes(x, labels)
        cosine_matrix = compute_cosine_matrix(prototypes)
        pearson_matrix = compute_pearson_matrix(prototypes)
        plot_heatmap(
            cosine_matrix,
            proto_labels,
            f"{feature_name} Taxonomy Prototype Cosine",
            plots_dir / f"{feature_name}_taxonomy_cosine.png",
            vmin=-1.0,
            vmax=1.0,
        )
        plot_heatmap(
            pearson_matrix,
            proto_labels,
            f"{feature_name} Taxonomy Prototype Pearson",
            plots_dir / f"{feature_name}_taxonomy_pearson.png",
            vmin=-1.0,
            vmax=1.0,
        )

        same_mean, diff_mean, gap = compute_same_diff_cosine_stats(x, labels)
        summary_rows.append(
            {
                "feature": feature_name,
                "valid_samples": int(valid_mask.sum()),
                "dim": int(x.shape[1]),
                "taxonomy_count": len(set(labels)),
                "same_taxonomy_cosine_mean": same_mean,
                "diff_taxonomy_cosine_mean": diff_mean,
                "same_minus_diff_gap": gap,
            }
        )
        visual_selection_rows.append(
            build_visual_selection_summary_row(
                feature_name=feature_name,
                view="raw",
                selected_indices=np.arange(x.shape[0], dtype=np.int32),
                x=x,
                labels=labels,
                keep_per_taxonomy=None,
            )
        )

    write_summary_csv(
        output_dir / "feature_separation_summary.csv",
        summary_rows,
        [
            "feature",
            "valid_samples",
            "dim",
            "taxonomy_count",
            "same_taxonomy_cosine_mean",
            "diff_taxonomy_cosine_mean",
            "same_minus_diff_gap",
        ],
    )
    write_summary_csv(
        output_dir / "taxonomy_feature_availability.csv",
        taxonomy_count_rows,
        [
            "taxonomy",
            "sample_count",
            "z_pool_valid",
            "g_pool_valid",
            "delta_pool_valid",
            "p_pool_valid",
        ],
    )
    write_summary_csv(
        output_dir / "feature_visual_selection_summary.csv",
        visual_selection_rows,
        [
            "feature",
            "view",
            "selected_samples",
            "taxonomy_count",
            "target_keep_per_taxonomy",
            "min_selected_per_taxonomy",
            "max_selected_per_taxonomy",
            "same_taxonomy_cosine_mean",
            "diff_taxonomy_cosine_mean",
            "same_minus_diff_gap",
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Low-cost CAPM-only taxonomy feature extraction for UniICL-Bench."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_UNIICL_FINETUNED_MODEL,
        help="Path to the finetuned UniICL checkpoint.",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default=DEFAULT_UNIICL_BASE_MODEL,
        help="Path to the base UniICL checkpoint.",
    )
    parser.add_argument(
        "--use-mixed-weights",
        action="store_true",
        help="Load base model first, then overwrite with finetuned understanding weights.",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default=".",
        help="Root directory of UniICL-Bench.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./taxonomy_capm_analysis",
        help="Directory for predictions, features, and plots.",
    )
    parser.add_argument(
        "--task",
        nargs="+",
        default=["all"],
        help="Tasks to run, or 'all'.",
    )
    parser.add_argument(
        "--k-shot",
        type=int,
        default=2,
        help="Number of demos to use where the task supports k-shot evaluation.",
    )
    parser.add_argument(
        "--enable-external-judges",
        action="store_true",
        help="Keep original judge/scorer calls enabled. Default is disabled.",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Only run feature extraction and save raw arrays.",
    )
    parser.add_argument(
        "--no-capm",
        action="store_true",
        help="Disable CAPM loading.",
    )
    parser.add_argument(
        "--capm-inject-layers",
        type=int,
        default=None,
        help="Optional CAPM gate truncation for top-layer injection ablations.",
    )
    parser.add_argument(
        "--capm-ablation-mode",
        type=str,
        default="none",
        choices=CAPM_ABLATION_CHOICES,
        help="Inference-time CAPM ablation mode.",
    )
    parser.add_argument(
        "--capm-fixed-tau",
        type=float,
        default=None,
        help="Optional fixed CAPM routing tau.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = resolve_tasks(args.task)
    benchmark_dir = Path(args.benchmark_dir).resolve()
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    prediction_dir = ensure_dir(output_dir / "predictions")

    if not args.enable_external_judges:
        disable_external_judges_and_scorers()

    model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids = load_bagel_model(
        args.model_path,
        base_model_path=args.base_model_path,
        use_mixed_weights=args.use_mixed_weights,
        enable_capm=not args.no_capm,
    )

    if model.capm is not None:
        model.capm.config.ablation_mode = args.capm_ablation_mode
        if args.capm_fixed_tau is not None:
            if args.capm_fixed_tau <= 0:
                raise ValueError("--capm-fixed-tau must be positive")
            model.capm.config.fixed_tau = args.capm_fixed_tau
        truncate_capm_gates(model, args.capm_inject_layers)
    elif args.capm_ablation_mode != "none":
        print(
            f"Warning: CAPM ablation mode '{args.capm_ablation_mode}' requested, "
            "but CAPM is disabled or unavailable."
        )

    inferencer = CapmOnlyFeatureInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )

    all_records: List[Dict] = []
    task_run_summary: List[Dict[str, object]] = []

    for task in tasks:
        print("=" * 80)
        print(f"Running task={task} k_shot={args.k_shot} [CAPM-only]")
        print("=" * 80)
        results, merged_records = run_task(
            inferencer=inferencer,
            task=task,
            k_shot=args.k_shot,
            benchmark_dir=benchmark_dir,
            output_dir=prediction_dir,
        )
        all_records.extend(merged_records)
        task_summary = {
            "task": task,
            "num_results": len(results),
            "num_feature_records": len(merged_records),
            "capm_valid": sum("z_pool" in r["features"] for r in merged_records),
        }
        task_summary.update(summarize_tau_records(merged_records))
        task_run_summary.append(task_summary)

    feature_dims = feature_dims_from_model(model)
    arrays, masks, metadata = build_feature_arrays(all_records, feature_dims)

    npz_payload = {}
    for feature_name, array in arrays.items():
        npz_payload[feature_name] = array
        npz_payload[f"{feature_name}_mask"] = masks[feature_name]
    np.savez_compressed(output_dir / "taxonomy_features.npz", **npz_payload)
    write_metadata_jsonl(output_dir / "taxonomy_features_metadata.jsonl", metadata)
    overall_summary = {
        "num_feature_records": len(all_records),
        "capm_valid": sum("z_pool" in r["features"] for r in all_records),
    }
    overall_summary.update(summarize_tau_records(all_records))
    save_json(
        output_dir / "task_run_summary.json",
        {
            "tasks": task_run_summary,
            "overall": overall_summary,
        },
    )

    if not args.skip_analysis:
        analyze_features(arrays, masks, metadata, output_dir / "analysis")

    print(f"\nSaved features to: {output_dir / 'taxonomy_features.npz'}")
    print(f"Saved metadata to: {output_dir / 'taxonomy_features_metadata.jsonl'}")
    if not args.skip_analysis:
        print(f"Saved analysis to: {output_dir / 'analysis'}")


if __name__ == "__main__":
    main()
