#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


BENCH_ROOT = Path(__file__).resolve().parent
OPEN_SOURCE_ROOT = BENCH_ROOT.parent
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))
if str(OPEN_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(OPEN_SOURCE_ROOT))
if str(OPEN_SOURCE_ROOT / "UniICL") not in sys.path:
    sys.path.insert(0, str(OPEN_SOURCE_ROOT / "UniICL"))

from analyze_taxonomy_capm_only import CAPM_FEATURE_ORDER, analyze_features
from analyze_taxonomy_features import ensure_dir, write_summary_csv


UNDERSTANDING_TASKS = {
    "visual_grounding",
    "attribute_recognition",
    "scene_reasoning",
    "style_aware_caption",
    "fast_concept_mapping",
    "world_aware_planning",
    "analogical_inference",
    "aesthetic_assessment",
    "forgery_detection",
}

GENERATION_TASKS = {
    "instructional_generation",
    "image_manipulation",
    "fast_concept_generation",
    "chain_of_editing",
    "analogical_editing",
    "visual_refinement",
}

SPLIT_TASKS = {
    "all": None,
    "understanding": UNDERSTANDING_TASKS,
    "generation": GENERATION_TASKS,
}

TAXONOMY_ORDER = [
    "Perception",
    "Imitation",
    "Conception",
    "Deduction",
    "Analogy",
    "Discernment",
]


def load_metadata(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), eps, None)


def compute_same_diff_gap(x: np.ndarray, labels: Sequence[str]) -> float:
    x_norm = l2_normalize(x)
    sim = x_norm @ x_norm.T
    same_vals: List[float] = []
    diff_vals: List[float] = []
    for i in range(sim.shape[0]):
        for j in range(i + 1, sim.shape[1]):
            if labels[i] == labels[j]:
                same_vals.append(float(sim[i, j]))
            else:
                diff_vals.append(float(sim[i, j]))
    if not same_vals or not diff_vals:
        return float("nan")
    return float(np.mean(same_vals) - np.mean(diff_vals))


def compute_centroid_accuracy(x: np.ndarray, labels: Sequence[str]) -> tuple[float, Dict[str, float]]:
    x_norm = l2_normalize(x)
    y = np.asarray(labels)
    proto_names: List[str] = []
    protos: List[np.ndarray] = []
    for name in TAXONOMY_ORDER:
        sel = y == name
        if not np.any(sel):
            continue
        proto = x_norm[sel].mean(axis=0)
        proto = proto / np.clip(np.linalg.norm(proto), 1e-8, None)
        proto_names.append(name)
        protos.append(proto)
    if not protos:
        return float("nan"), {}
    proto_arr = np.stack(protos, axis=0)
    pred = np.asarray([proto_names[i] for i in (x_norm @ proto_arr.T).argmax(axis=1)])
    overall = float((pred == y).mean())
    per_tax: Dict[str, float] = {}
    for name in proto_names:
        sel = y == name
        per_tax[name] = float((pred[sel] == y[sel]).mean())
    return overall, per_tax


def subset_data(
    arrays: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    metadata: Sequence[Dict],
    split_name: str,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[Dict]]:
    task_filter = SPLIT_TASKS[split_name]
    selected_indices = [
        idx for idx, row in enumerate(metadata) if task_filter is None or row["task"] in task_filter
    ]
    sub_metadata = [metadata[idx] for idx in selected_indices]
    sub_arrays = {
        feature_name: arr[selected_indices]
        for feature_name, arr in arrays.items()
    }
    sub_masks = {
        feature_name: mask[selected_indices]
        for feature_name, mask in masks.items()
    }
    return sub_arrays, sub_masks, sub_metadata


def build_core_masks(
    arrays: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    metadata: Sequence[Dict],
    per_taxonomy: int,
) -> Dict[str, np.ndarray]:
    core_masks: Dict[str, np.ndarray] = {}
    labels = np.asarray([row["taxonomy"] for row in metadata])

    for feature_name in CAPM_FEATURE_ORDER:
        valid_mask = masks[feature_name].astype(bool)
        feature_core_mask = np.zeros_like(valid_mask, dtype=bool)
        valid_indices = np.flatnonzero(valid_mask)
        if len(valid_indices) == 0:
            core_masks[feature_name] = feature_core_mask
            continue

        x_valid = arrays[feature_name][valid_mask]
        y_valid = labels[valid_mask]
        x_valid = l2_normalize(x_valid)

        for taxonomy in TAXONOMY_ORDER:
            local_idx = np.flatnonzero(y_valid == taxonomy)
            if len(local_idx) == 0:
                continue

            local_x = x_valid[local_idx]
            centroid = local_x.mean(axis=0)
            centroid = centroid / np.clip(np.linalg.norm(centroid), 1e-8, None)
            sims = local_x @ centroid
            keep_n = min(per_taxonomy, len(local_idx))
            chosen = local_idx[np.argsort(-sims)[:keep_n]]
            feature_core_mask[valid_indices[chosen]] = True

        core_masks[feature_name] = feature_core_mask

    return core_masks


def build_summary_rows(
    mode_name: str,
    split_name: str,
    arrays: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    metadata: Sequence[Dict],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for feature_name in CAPM_FEATURE_ORDER:
        if feature_name not in arrays:
            continue
        valid_mask = masks[feature_name].astype(bool)
        x = arrays[feature_name][valid_mask]
        if len(x) < 2:
            continue
        labels = [metadata[idx]["taxonomy"] for idx, flag in enumerate(valid_mask) if flag]
        gap = compute_same_diff_gap(x, labels)
        acc, per_tax = compute_centroid_accuracy(x, labels)
        row: Dict[str, object] = {
            "mode": mode_name,
            "split": split_name,
            "feature": feature_name,
            "valid_samples": int(valid_mask.sum()),
            "same_minus_diff_gap": gap,
            "centroid_acc": acc,
        }
        for name in TAXONOMY_ORDER:
            row[f"{name}_acc"] = per_tax.get(name, float("nan"))
        rows.append(row)
    return rows


def make_grid(
    image_map: Dict[tuple[str, str], Path],
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(
        len(row_labels),
        len(col_labels),
        figsize=(5.5 * len(col_labels), 4.5 * len(row_labels)),
    )
    if len(row_labels) == 1 and len(col_labels) == 1:
        axes = np.asarray([[axes]])
    elif len(row_labels) == 1:
        axes = np.asarray([axes])
    elif len(col_labels) == 1:
        axes = np.asarray([[ax] for ax in axes])

    for row_idx, row_label in enumerate(row_labels):
        for col_idx, col_label in enumerate(col_labels):
            ax = axes[row_idx, col_idx]
            img_path = image_map.get((row_label, col_label))
            ax.axis("off")
            if img_path is None or not img_path.exists():
                ax.text(0.5, 0.5, "Missing", ha="center", va="center", fontsize=14)
                continue
            ax.imshow(mpimg.imread(img_path))
            if row_idx == 0:
                ax.set_title(col_label, fontsize=14)
            if col_idx == 0:
                ax.text(
                    -0.08,
                    0.5,
                    row_label,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=13,
                )

    fig.suptitle(title, fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create split-specific comparison plots for CAPM-only taxonomy features."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Completed taxonomy CAPM analysis directory, e.g. taxonomy_capm_analysis_8gpu_2shot.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: <input-dir>/split_views",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "core", "both"],
        default="both",
        help="Whether to generate full-sample views, centroid-near core views, or both.",
    )
    parser.add_argument(
        "--core-per-taxonomy",
        type=int,
        default=40,
        help="Number of nearest-to-centroid samples to keep per taxonomy for core views.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = ensure_dir(
        Path(args.output_dir).resolve() if args.output_dir else input_dir / "split_views"
    )

    npz = np.load(input_dir / "taxonomy_features.npz")
    metadata = load_metadata(input_dir / "taxonomy_features_metadata.jsonl")

    arrays = {feature_name: np.asarray(npz[feature_name]) for feature_name in CAPM_FEATURE_ORDER}
    masks = {
        feature_name: np.asarray(npz[f"{feature_name}_mask"]).astype(bool)
        for feature_name in CAPM_FEATURE_ORDER
    }

    active_modes: List[str]
    if args.mode == "both":
        active_modes = ["full", "core"]
    else:
        active_modes = [args.mode]

    all_summary_rows: List[Dict[str, object]] = []
    split_labels = list(SPLIT_TASKS.keys())

    for mode_name in active_modes:
        mode_dir = ensure_dir(output_dir / mode_name)
        split_dirs: Dict[str, Path] = {}

        for split_name in SPLIT_TASKS:
            sub_arrays, sub_masks, sub_metadata = subset_data(arrays, masks, metadata, split_name)
            if mode_name == "core":
                sub_masks = build_core_masks(
                    sub_arrays,
                    sub_masks,
                    sub_metadata,
                    per_taxonomy=args.core_per_taxonomy,
                )
            split_dir = ensure_dir(mode_dir / split_name)
            split_dirs[split_name] = split_dir
            analyze_features(sub_arrays, sub_masks, sub_metadata, split_dir)
            all_summary_rows.extend(
                build_summary_rows(mode_name, split_name, sub_arrays, sub_masks, sub_metadata)
            )

        for view_name, title in [
            ("tsne_taxonomy", f"CAPM Taxonomy t-SNE Comparison ({mode_name})"),
            ("tsne_task", f"CAPM Task t-SNE Comparison ({mode_name})"),
            ("taxonomy_cosine", f"CAPM Taxonomy Prototype Cosine Comparison ({mode_name})"),
            ("taxonomy_pearson", f"CAPM Taxonomy Prototype Pearson Comparison ({mode_name})"),
        ]:
            image_map: Dict[tuple[str, str], Path] = {}
            for feature_name in CAPM_FEATURE_ORDER:
                for split_name in split_labels:
                    image_map[(feature_name, split_name)] = (
                        split_dirs[split_name] / "plots" / f"{feature_name}_{view_name}.png"
                    )
            make_grid(
                image_map=image_map,
                row_labels=CAPM_FEATURE_ORDER,
                col_labels=split_labels,
                output_path=mode_dir / f"comparison_{view_name}_grid.png",
                title=title,
            )

    summary_fields = [
        "mode",
        "split",
        "feature",
        "valid_samples",
        "same_minus_diff_gap",
        "centroid_acc",
        "Perception_acc",
        "Imitation_acc",
        "Conception_acc",
        "Deduction_acc",
        "Analogy_acc",
        "Discernment_acc",
    ]
    write_summary_csv(output_dir / "split_feature_summary.csv", all_summary_rows, summary_fields)

    print(f"Saved split comparison plots to: {output_dir}")


if __name__ == "__main__":
    main()
