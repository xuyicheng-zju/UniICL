#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


BENCH_ROOT = Path(__file__).resolve().parent
OPEN_SOURCE_ROOT = BENCH_ROOT.parent
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))
if str(OPEN_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(OPEN_SOURCE_ROOT))
if str(OPEN_SOURCE_ROOT / "UniICL") not in sys.path:
    sys.path.insert(0, str(OPEN_SOURCE_ROOT / "UniICL"))

import utils.evaluators as evaluators
from data.data_utils import patchify, pil_img2rgb
from eval_bagel import (
    CAPM_ABLATION_CHOICES,
    SafeInterleaveInferencer,
    load_bagel_model,
)
from public_path_config import (
    CANONICAL_TASK_ORDER,
    DEFAULT_UNIICL_BASE_MODEL,
    DEFAULT_UNIICL_FINETUNED_MODEL,
    get_task_data_path,
    get_task_image_dir,
    normalize_task_name,
)
from utils.evaluators import (
    eval_aesthetic_assessment,
    eval_attr_rec_gen,
    eval_authenticity_detection,
    eval_caption_styled,
    eval_chain_of_editing,
    eval_fcb_classification,
    eval_fci_t2i,
    eval_grounding,
    eval_i2i_editing,
    eval_image_perfection,
    eval_planning,
    eval_t2i,
    eval_visualcloze_g,
    eval_visualcloze_u,
    eval_vqa_gen,
)


TASK_TO_TAXONOMY = {
    "visual_grounding": "Perception",
    "attribute_recognition": "Perception",
    "image_manipulation": "Perception",
    "style_aware_caption": "Imitation",
    "scene_reasoning": "Imitation",
    "instructional_generation": "Imitation",
    "fast_concept_mapping": "Conception",
    "fast_concept_generation": "Conception",
    "world_aware_planning": "Deduction",
    "chain_of_editing": "Deduction",
    "analogical_inference": "Analogy",
    "analogical_editing": "Analogy",
    "aesthetic_assessment": "Discernment",
    "forgery_detection": "Discernment",
    "visual_refinement": "Discernment",
}

TAXONOMY_ORDER = [
    "Perception",
    "Imitation",
    "Conception",
    "Deduction",
    "Analogy",
    "Discernment",
]

TAXONOMY_COLORS = {
    "Perception": "#0072B2",
    "Imitation": "#E69F00",
    "Conception": "#009E73",
    "Deduction": "#D55E00",
    "Analogy": "#CC79A7",
    "Discernment": "#4D4D4D",
}

TAXONOMY_DRAW_ORDER = [
    "Discernment",
    "Imitation",
    "Perception",
    "Analogy",
    "Conception",
    "Deduction",
]

TASK_CONFIG = {
    "visual_grounding": eval_grounding,
    "attribute_recognition": eval_attr_rec_gen,
    "scene_reasoning": eval_vqa_gen,
    "style_aware_caption": eval_caption_styled,
    "instructional_generation": eval_t2i,
    "image_manipulation": eval_i2i_editing,
    "aesthetic_assessment": eval_aesthetic_assessment,
    "forgery_detection": eval_authenticity_detection,
    "visual_refinement": eval_image_perfection,
    "fast_concept_mapping": eval_fcb_classification,
    "fast_concept_generation": eval_fci_t2i,
    "world_aware_planning": eval_planning,
    "chain_of_editing": eval_chain_of_editing,
    "analogical_editing": eval_visualcloze_g,
    "analogical_inference": eval_visualcloze_u,
}

FEATURE_ORDER = [
    "z_pool",
    "g_pool",
    "delta_pool",
    "p_pool",
    "hs_full_pool",
]


def disable_external_judges_and_scorers() -> None:
    def _disabled_judge(*args, **kwargs):
        return ""

    def _disabled_extraction(*args, **kwargs):
        return None

    def _zero_score(*args, **kwargs):
        return -1.0

    def _zero_bert(*args, **kwargs):
        zero = torch.tensor([0.0])
        return zero, zero, zero

    evaluators.call_vllm_judge = _disabled_judge
    evaluators.mllm_assisted_extraction = _disabled_extraction
    evaluators.compute_qalign_score = _zero_score
    evaluators.compute_hpsv3_score = _zero_score
    evaluators.compute_clip_score = _zero_score
    evaluators.load_clip_model = lambda *args, **kwargs: None
    evaluators.bert_score = _zero_bert


def resolve_tasks(task_args: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    if len(task_args) == 1 and str(task_args[0]).lower() == "all":
        return list(CANONICAL_TASK_ORDER)
    for task in task_args:
        normalized.append(normalize_task_name(task))
    return normalized


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _extract_primary_id(result: Dict, fallback: str) -> str:
    for key in ("id", "image_name", "generated_path", "source_path", "prompt"):
        value = result.get(key)
        if isinstance(value, str) and value:
            return value
    return fallback


class FeatureTracingInferencer(SafeInterleaveInferencer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_task: Optional[str] = None
        self.current_taxonomy: Optional[str] = None
        self.episode_records: List[Dict] = []
        self._episode_counter = 0
        self._forward_trace_buffer: List[torch.Tensor] = []
        self._capture_forward_traces = False
        self._install_forward_trace_hook()
        self._reset_episode_state()

    def _install_forward_trace_hook(self) -> None:
        original_forward_inference = self.model.language_model.forward_inference

        def wrapped_forward_inference(*args, **kwargs):
            output = original_forward_inference(*args, **kwargs)
            if self._capture_forward_traces:
                self._forward_trace_buffer.append(
                    output.packed_query_sequence.detach().to(torch.float32).cpu()
                )
            return output

        self.model.language_model.forward_inference = wrapped_forward_inference

    def set_run_context(self, task: str, taxonomy: str) -> None:
        self.current_task = task
        self.current_taxonomy = taxonomy

    def drain_episode_records(self) -> List[Dict]:
        records = self.episode_records
        self.episode_records = []
        return records

    def _reset_episode_state(self) -> None:
        self._episode_full_chunks: List[torch.Tensor] = []
        self._episode_features: Dict[str, np.ndarray] = {}
        self._episode_variable_features: Dict[str, np.ndarray] = {}
        self._episode_error: Optional[str] = None
        self._episode_capm_available = False
        self._episode_demo_count = 0
        self._episode_hidden_token_count = 0
        self._episode_query_item_indices: set[int] = set()
        self._episode_item_cursor = 0

    def _append_hidden_chunk(self, hidden: torch.Tensor, trim_special: bool = True) -> None:
        if hidden is None or hidden.numel() == 0:
            return
        if hidden.ndim == 3:
            hidden = hidden.squeeze(0)
        if hidden.ndim != 2:
            return
        if trim_special and hidden.shape[0] > 2:
            hidden = hidden[1:-1]
        if hidden.numel() == 0:
            return
        hidden = hidden.to(torch.float32).cpu()
        self._episode_full_chunks.append(hidden)
        self._episode_hidden_token_count += int(hidden.shape[0])

    def _infer_query_item_indices(
        self,
        input_lists,
        understanding_output: bool,
        num_demos: int,
    ) -> set[int]:
        demo_tuples = []
        demo_item_count = 0
        query_item_indices: set[int] = set()

        if understanding_output:
            parsed_demos, parsed_query_item_indices = self._parse_understanding_icl_for_capm(input_lists)
            demo_tuples = parsed_demos
            query_item_indices = parsed_query_item_indices
            if demo_tuples:
                num_demos = len(demo_tuples)

        if num_demos == 0 and not demo_tuples:
            i = 0
            detected_a = []
            while i + 3 < len(input_lists):
                if (
                    isinstance(input_lists[i], str)
                    and isinstance(input_lists[i + 1], Image.Image)
                    and isinstance(input_lists[i + 2], str)
                    and isinstance(input_lists[i + 3], str)
                    and "Assistant" in input_lists[i + 3]
                ):
                    user_text = input_lists[i].strip() + " " + input_lists[i + 2].strip()
                    assistant_text = input_lists[i + 3].strip()
                    if assistant_text.startswith("\nAssistant:"):
                        assistant_text = assistant_text[len("\nAssistant:") :].strip()
                    elif assistant_text.startswith("Assistant:"):
                        assistant_text = assistant_text[len("Assistant:") :].strip()
                    detected_a.append((input_lists[i + 1], user_text, assistant_text))
                    i += 4
                else:
                    break

            if len(detected_a) >= 2:
                demo_tuples = detected_a[:-1]
                demo_item_count = len(demo_tuples) * 4
                query_item_indices = set(range(demo_item_count, len(input_lists)))

            if not demo_tuples:
                i = 0
                detected_b = []
                while i + 1 < len(input_lists):
                    if isinstance(input_lists[i], Image.Image) and isinstance(input_lists[i + 1], str):
                        detected_b.append(i)
                        i += 2
                    else:
                        break
                if len(detected_b) >= 2:
                    demo_item_count = (len(detected_b) - 1) * 2
                    query_item_indices = set(range(demo_item_count, len(input_lists)))

        if query_item_indices:
            return query_item_indices
        if demo_item_count > 0:
            return set(range(demo_item_count, len(input_lists)))
        return set(range(len(input_lists)))

    @torch.no_grad()
    def _build_demo_inputs(
        self,
        image: Image.Image,
        user_text: str,
        assistant_text: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embeds_parts: List[torch.Tensor] = []
        seg_parts: List[torch.Tensor] = []

        image_pil = self.vae_transform.resize_transform(pil_img2rgb(image))
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
        cu_seqlens = torch.tensor([0, vit_tokens.shape[0]], dtype=torch.int32, device=self.device)

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
            seg_parts.append(torch.zeros(vit_embed.shape[0], device=self.device, dtype=torch.long))

            if user_text:
                user_ids = self.tokenizer.encode(user_text)
                user_ids_t = torch.tensor(user_ids, dtype=torch.long, device=self.device)
                user_embed = self.model.language_model.model.embed_tokens(user_ids_t)
                embeds_parts.append(user_embed)
                seg_parts.append(torch.zeros(len(user_ids), device=self.device, dtype=torch.long))

            if assistant_text:
                assistant_ids = self.tokenizer.encode(assistant_text)
                assistant_ids_t = torch.tensor(assistant_ids, dtype=torch.long, device=self.device)
                assistant_embed = self.model.language_model.model.embed_tokens(assistant_ids_t)
                embeds_parts.append(assistant_embed)
                seg_parts.append(torch.ones(len(assistant_ids), device=self.device, dtype=torch.long))

        demo_embed = torch.cat(embeds_parts, dim=0).unsqueeze(0)
        demo_seg = torch.cat(seg_parts, dim=0).unsqueeze(0)
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
    def _encode_capm_demos(self, demo_tuples) -> None:
        if not demo_tuples or self.model.capm is None:
            return

        all_z: List[torch.Tensor] = []
        all_g: List[torch.Tensor] = []
        all_delta: List[torch.Tensor] = []
        all_p: List[torch.Tensor] = []
        all_c_in: List[torch.Tensor] = []
        all_c_out: List[torch.Tensor] = []
        all_facets: List[torch.Tensor] = []

        demo_embeds_list: List[torch.Tensor] = []
        segment_ids_list: List[torch.Tensor] = []
        K = self.model.capm.K

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            for image, user_text, assistant_text in demo_tuples:
                demo_embed, demo_seg = self._build_demo_inputs(image, user_text, assistant_text)
                demo_embeds_list.append(demo_embed)
                segment_ids_list.append(demo_seg)

                c_in, c_out, facets = self.model.capm.encoder.prober(demo_embed, demo_seg)
                g = self.model.capm.encoder.pooler(facets)
                if self.model.capm.encoder.detach_operator_from_facets:
                    g_for_op = g.detach()
                else:
                    g_for_op = g

                if getattr(self.model.capm.config, "ablation_mode", "none") == "no_decoupled_encoding":
                    c_in = g_for_op
                    c_out = g_for_op
                    facets = g_for_op.unsqueeze(1).expand(-1, facets.shape[1], -1)

                p, delta, z = self._compute_operator_outputs(c_in, c_out, g_for_op)

                all_c_in.append(c_in)
                all_c_out.append(c_out)
                all_facets.append(facets)
                all_g.append(g_for_op)
                all_p.append(p)
                all_delta.append(delta)
                all_z.append(z)

            all_z = self.model.capm.demo_interaction(all_z)
            z_pool = torch.stack(all_z, dim=1).mean(dim=1)
            g_pool = torch.stack(all_g, dim=1).mean(dim=1)
            delta_pool = torch.stack(all_delta, dim=1).mean(dim=1)
            p_pool = torch.stack(all_p, dim=1).mean(dim=1)

            all_tokens = []
            all_type_ids = []
            for z, c_in, c_out, facets in zip(all_z, all_c_in, all_c_out, all_facets):
                tokens_i = torch.cat(
                    [
                        z.unsqueeze(1),
                        c_in.unsqueeze(1),
                        c_out.unsqueeze(1),
                        facets,
                    ],
                    dim=1,
                )
                type_ids_i = torch.tensor(
                    [0, 1, 2] + [3] * K,
                    device=tokens_i.device,
                    dtype=torch.long,
                )
                all_tokens.append(tokens_i)
                all_type_ids.append(type_ids_i)

            B_tokens = torch.cat(all_tokens, dim=1)
            type_ids = torch.cat(all_type_ids, dim=0)
            B_cal = self.model.capm.calibrator(B_tokens, type_ids)

        self.model.capm._cached_bank_cal = B_cal
        self.model.capm._cached_bank_mask = None
        self.model.capm._cached_z_pool = z_pool

        self._episode_capm_available = True
        self._episode_demo_count = len(demo_tuples)
        self._episode_features["z_pool"] = z_pool[0].detach().to(torch.float32).cpu().numpy()
        self._episode_features["g_pool"] = g_pool[0].detach().to(torch.float32).cpu().numpy()
        self._episode_features["delta_pool"] = delta_pool[0].detach().to(torch.float32).cpu().numpy()
        self._episode_features["p_pool"] = p_pool[0].detach().to(torch.float32).cpu().numpy()

        print(f"  [CAPM] Encoded {len(demo_tuples)} demos into pattern bank")

    @torch.no_grad()
    def update_context_text(self, text, gen_context, capm_active=False):
        self._episode_item_cursor += 1
        self._forward_trace_buffer = []
        self._capture_forward_traces = True
        try:
            gen_context = super().update_context_text(text, gen_context, capm_active=capm_active)
        finally:
            self._capture_forward_traces = False
        if self._forward_trace_buffer:
            hidden = self._forward_trace_buffer[-1]
            self._append_hidden_chunk(hidden, trim_special=True)
        self._forward_trace_buffer = []
        return gen_context

    @torch.no_grad()
    def update_context_image(self, image, gen_context, vae=True, vit=True, capm_active=False):
        self._episode_item_cursor += 1
        self._forward_trace_buffer = []
        self._capture_forward_traces = True
        try:
            gen_context = super().update_context_image(
                image,
                gen_context,
                vae=vae,
                vit=vit,
                capm_active=capm_active,
            )
        finally:
            self._capture_forward_traces = False
        if self._forward_trace_buffer:
            for hidden in self._forward_trace_buffer:
                self._append_hidden_chunk(hidden, trim_special=True)
        self._forward_trace_buffer = []
        return gen_context

    def interleave_inference(self, *args, **kwargs):
        self._reset_episode_state()
        self._episode_counter += 1
        input_lists = kwargs.get("input_lists")
        if input_lists is None and args:
            input_lists = args[0]
        understanding_output = kwargs.get("understanding_output", False)
        num_demos = int(kwargs.get("num_demos", 0))
        if input_lists is None:
            self._episode_query_item_indices = set()
        else:
            self._episode_query_item_indices = self._infer_query_item_indices(
                input_lists=input_lists,
                understanding_output=understanding_output,
                num_demos=num_demos,
            )
        try:
            return super().interleave_inference(*args, **kwargs)
        except Exception as exc:
            self._episode_error = str(exc)
            raise
        finally:
            hs_full_pool = None
            hs_full_tokens = None
            if self._episode_full_chunks:
                hs_full_tokens = torch.cat(self._episode_full_chunks, dim=0).to(torch.float32).cpu().numpy()
                hs_full_pool = hs_full_tokens.mean(axis=0)
            if hs_full_pool is not None:
                self._episode_features["hs_full_pool"] = hs_full_pool
            if hs_full_tokens is not None:
                self._episode_variable_features["hs_full_tokens"] = hs_full_tokens

            self.episode_records.append(
                {
                    "episode_index": self._episode_counter,
                    "task": self.current_task,
                    "taxonomy": self.current_taxonomy,
                    "capm_available": self._episode_capm_available,
                    "demo_count": self._episode_demo_count,
                    "hidden_token_count": self._episode_hidden_token_count,
                    "episode_error": self._episode_error,
                    "features": dict(self._episode_features),
                    "variable_features": dict(self._episode_variable_features),
                }
            )


def truncate_capm_gates(model, num_inject_layers: Optional[int]) -> None:
    if model.capm is None or num_inject_layers is None:
        return
    trained_gates = len(model.capm.gates)
    k = num_inject_layers
    if k <= 0:
        model.capm = None
        print("⏭️ --capm-inject-layers=0, CAPM disabled")
        return
    if k > trained_gates:
        print(
            f"⚠️ --capm-inject-layers={k} > trained gates ({trained_gates}), using all {trained_gates}"
        )
        return
    if k < trained_gates:
        import torch.nn as nn

        model.capm.gates = nn.ModuleList(list(model.capm.gates)[-k:])
        print(f"✂️ CAPM gates truncated: {trained_gates} -> {k}")


def run_task(
    inferencer: FeatureTracingInferencer,
    task: str,
    k_shot: int,
    benchmark_dir: Path,
    output_dir: Path,
) -> Tuple[List[Dict], List[Dict]]:
    output_path = output_dir / f"{task}_results.json"
    image_dir = get_task_image_dir(task, benchmark_dir)
    data_path = get_task_data_path(task, benchmark_dir)
    taxonomy = TASK_TO_TAXONOMY[task]
    eval_func = TASK_CONFIG[task]

    inferencer.set_run_context(task, taxonomy)
    inferencer.drain_episode_records()

    if task == "visual_refinement":
        results = eval_func(
            inferencer,
            data_path,
            image_dir,
            str(output_path),
            k_shot,
            qalign_model=None,
            skip_scoring=True,
        )
    elif task == "analogical_editing":
        results = eval_func(
            inferencer,
            data_path,
            image_dir,
            str(output_path),
            k_shot,
        )
    elif task == "instructional_generation":
        results = eval_func(
            inferencer,
            data_path,
            image_dir,
            str(output_path),
            k_shot,
            hps_model=None,
        )
    else:
        results = eval_func(
            inferencer,
            data_path,
            image_dir,
            str(output_path),
            k_shot,
        )

    episode_records = inferencer.drain_episode_records()
    if len(episode_records) != len(results):
        print(
            f"[WARN] {task}: feature records ({len(episode_records)}) != results ({len(results)}). "
            f"Using min length."
        )
    merged: List[Dict] = []
    for idx, (episode_record, result) in enumerate(zip(episode_records, results)):
        merged_record = dict(episode_record)
        merged_record["result_index"] = idx
        merged_record["sample_id"] = _extract_primary_id(result, f"{task}_{idx:05d}")
        merged_record["inference_failed"] = bool(result.get("inference_failed", False))
        merged_record["result_keys"] = sorted(result.keys())
        merged.append(merged_record)
    return results, merged


def feature_dims_from_model(model) -> Dict[str, int]:
    d_capm = model.capm.config.d_capm if model.capm is not None else 0
    rank = model.capm.config.operator_rank if model.capm is not None else 0
    hidden = int(model.hidden_size)
    return {
        "z_pool": d_capm,
        "g_pool": d_capm,
        "delta_pool": d_capm,
        "p_pool": rank,
        "hs_full_pool": hidden,
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
                "hidden_token_count": int(record["hidden_token_count"]),
                "inference_failed": bool(record["inference_failed"]),
                "episode_error": record["episode_error"],
            }
        )
    return arrays, masks, metadata


def write_metadata_jsonl(path: Path, metadata: Sequence[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in metadata:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_variable_feature_archive(path: Path, records: Sequence[Dict]) -> None:
    payload: List[Dict[str, object]] = []
    for record in records:
        variable_features = record.get("variable_features", {})
        hs_full_tokens = variable_features.get("hs_full_tokens")
        payload.append(
            {
                "episode_index": int(record["episode_index"]),
                "task": record["task"],
                "taxonomy": record["taxonomy"],
                "sample_id": record["sample_id"],
                "hidden_token_count": int(record["hidden_token_count"]),
                "hs_full_tokens": None
                if hs_full_tokens is None
                else torch.from_numpy(np.asarray(hs_full_tokens, dtype=np.float32)),
            }
        )
    torch.save(payload, path)


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def compute_tsne_embedding(x: np.ndarray, seed: int = 42) -> np.ndarray:
    if x.shape[0] < 2:
        raise ValueError("Need at least 2 samples for t-SNE")
    x_scaled = StandardScaler().fit_transform(x)
    pca_dim = min(50, x_scaled.shape[0] - 1, x_scaled.shape[1])
    if pca_dim >= 2:
        x_scaled = PCA(n_components=pca_dim, random_state=seed).fit_transform(x_scaled)
    perplexity = min(30, max(5, x.shape[0] // 8))
    perplexity = min(perplexity, x.shape[0] - 1)
    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=seed,
    )
    return tsne.fit_transform(x_scaled)


def scatter_plot(
    points: np.ndarray,
    labels: Sequence[str],
    color_map: Dict[str, str],
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 6))
    present_labels = set(labels)
    taxonomy_plot = present_labels.issubset(set(TAXONOMY_ORDER)) and present_labels.issubset(set(color_map))
    label_order = [label for label in color_map if label in present_labels]
    draw_order = label_order
    if taxonomy_plot:
        draw_order = [label for label in TAXONOMY_DRAW_ORDER if label in present_labels]
    marker_size = 18
    marker_alpha = 0.8
    marker_edgecolors = "none"
    marker_linewidths = 0.0
    if taxonomy_plot:
        num_points = len(labels)
        if num_points >= 4000:
            marker_size = 9
            marker_alpha = 0.42
            marker_linewidths = 0.18
        elif num_points >= 2000:
            marker_size = 12
            marker_alpha = 0.52
            marker_linewidths = 0.22
        elif num_points >= 1000:
            marker_size = 16
            marker_alpha = 0.62
            marker_linewidths = 0.26
        else:
            marker_size = 22
            marker_alpha = 0.72
            marker_linewidths = 0.30
        marker_edgecolors = "white"

    legend_handles = {}
    for zorder, label in enumerate(draw_order, start=1):
        mask = np.array([lab == label for lab in labels], dtype=bool)
        scatter = plt.scatter(
            points[mask, 0],
            points[mask, 1],
            s=marker_size,
            alpha=marker_alpha,
            color=color_map[label],
            edgecolors=marker_edgecolors,
            linewidths=marker_linewidths,
            label=label,
            zorder=zorder,
        )
        legend_handles[label] = scatter
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    legend_order = label_order
    if taxonomy_plot:
        legend_order = [label for label in TAXONOMY_ORDER if label in legend_handles]
    plt.legend(
        [legend_handles[label] for label in legend_order],
        legend_order,
        frameon=False,
        fontsize=8,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def compute_taxonomy_prototypes(
    x: np.ndarray,
    labels: Sequence[str],
) -> Tuple[List[str], np.ndarray]:
    ordered_labels = [label for label in TAXONOMY_ORDER if label in set(labels)]
    prototypes: List[np.ndarray] = []
    for label in ordered_labels:
        mask = np.array([lab == label for lab in labels], dtype=bool)
        prototypes.append(x[mask].mean(axis=0))
    return ordered_labels, np.stack(prototypes, axis=0)


def compute_cosine_matrix(prototypes: np.ndarray) -> np.ndarray:
    if prototypes.shape[0] == 1:
        return np.ones((1, 1), dtype=np.float32)
    protos = l2_normalize(prototypes)
    return protos @ protos.T


def compute_pearson_matrix(prototypes: np.ndarray) -> np.ndarray:
    if prototypes.shape[0] == 1:
        return np.ones((1, 1), dtype=np.float32)
    matrix = np.corrcoef(prototypes)
    matrix = np.nan_to_num(matrix, nan=0.0)
    return matrix


def plot_heatmap(
    matrix: np.ndarray,
    labels: Sequence[str],
    title: str,
    output_path: Path,
    vmin: float,
    vmax: float,
) -> None:
    masked = np.ma.masked_invalid(matrix)
    plt.figure(figsize=(7, 6))
    plt.imshow(masked, cmap="coolwarm", vmin=vmin, vmax=vmax)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = "NA" if np.isnan(value) else f"{value:.2f}"
            plt.text(j, i, text, ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def compute_same_diff_cosine_stats(
    x: np.ndarray,
    labels: Sequence[str],
) -> Tuple[float, float, float]:
    x_norm = l2_normalize(x)
    sim = x_norm @ x_norm.T
    n = sim.shape[0]
    same_vals: List[float] = []
    diff_vals: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                same_vals.append(float(sim[i, j]))
            else:
                diff_vals.append(float(sim[i, j]))
    same_mean = float(np.mean(same_vals)) if same_vals else float("nan")
    diff_mean = float(np.mean(diff_vals)) if diff_vals else float("nan")
    gap = same_mean - diff_mean if not (math.isnan(same_mean) or math.isnan(diff_mean)) else float("nan")
    return same_mean, diff_mean, gap


def write_summary_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def analyze_features(
    arrays: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    metadata: Sequence[Dict],
    output_dir: Path,
) -> None:
    plots_dir = ensure_dir(output_dir / "plots")
    summary_rows: List[Dict[str, object]] = []
    taxonomy_count_rows: List[Dict[str, object]] = []

    task_labels_all = [row["task"] for row in metadata]
    task_color_map = {}
    cmap = plt.get_cmap("tab20")
    for idx, task in enumerate(sorted(set(task_labels_all))):
        task_color_map[task] = cmap(idx % 20)

    for taxonomy in TAXONOMY_ORDER:
        row = {"taxonomy": taxonomy, "sample_count": sum(m["taxonomy"] == taxonomy for m in metadata)}
        for feature_name in FEATURE_ORDER:
            if feature_name in masks:
                row[f"{feature_name}_valid"] = int(
                    sum(
                        bool(mask)
                        and metadata[idx]["taxonomy"] == taxonomy
                        for idx, mask in enumerate(masks[feature_name])
                    )
                )
        taxonomy_count_rows.append(row)

    for feature_name in FEATURE_ORDER:
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
            tasks,
            task_color_map,
            f"{feature_name} t-SNE by Task",
            plots_dir / f"{feature_name}_tsne_task.png",
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
            "hs_full_pool_valid",
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone UniICL-Bench taxonomy feature extraction and analysis script."
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
        default="./taxonomy_feature_analysis",
        help="Directory for predictions, saved features, and plots.",
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
        help="Keep original judge/scorer calls enabled. Default is disabled for low-cost analysis.",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Only run inference and save raw feature arrays.",
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
            f"⚠️ CAPM ablation mode '{args.capm_ablation_mode}' requested, "
            "but CAPM is disabled or unavailable."
        )

    inferencer = FeatureTracingInferencer(
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
        taxonomy = TASK_TO_TAXONOMY[task]
        print("=" * 80)
        print(f"Running task={task} taxonomy={taxonomy} k_shot={args.k_shot}")
        print("=" * 80)
        results, merged_records = run_task(
            inferencer=inferencer,
            task=task,
            k_shot=args.k_shot,
            benchmark_dir=benchmark_dir,
            output_dir=prediction_dir,
        )
        all_records.extend(merged_records)
        task_run_summary.append(
            {
                "task": task,
                "taxonomy": taxonomy,
                "num_results": len(results),
                "num_feature_records": len(merged_records),
                "capm_valid": sum(bool(r["capm_available"]) for r in merged_records),
                "hidden_valid": sum("hs_full_pool" in r["features"] for r in merged_records),
            }
        )

    feature_dims = feature_dims_from_model(model)
    arrays, masks, metadata = build_feature_arrays(all_records, feature_dims)

    npz_payload = {}
    for feature_name, array in arrays.items():
        npz_payload[feature_name] = array
        npz_payload[f"{feature_name}_mask"] = masks[feature_name]
    np.savez_compressed(output_dir / "taxonomy_features.npz", **npz_payload)
    write_metadata_jsonl(output_dir / "taxonomy_features_metadata.jsonl", metadata)
    save_variable_feature_archive(output_dir / "taxonomy_variable_features.pt", all_records)
    save_json(output_dir / "task_run_summary.json", {"tasks": task_run_summary})

    if not args.skip_analysis:
        analyze_features(arrays, masks, metadata, output_dir / "analysis")

    print(f"\nSaved features to: {output_dir / 'taxonomy_features.npz'}")
    print(f"Saved variable token features to: {output_dir / 'taxonomy_variable_features.pt'}")
    print(f"Saved metadata to: {output_dir / 'taxonomy_features_metadata.jsonl'}")
    if not args.skip_analysis:
        print(f"Saved analysis to: {output_dir / 'analysis'}")


if __name__ == "__main__":
    main()
