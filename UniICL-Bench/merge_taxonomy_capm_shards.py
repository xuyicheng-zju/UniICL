#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from analyze_taxonomy_capm_only import CAPM_FEATURE_ORDER, analyze_features
from analyze_taxonomy_features import ensure_dir, save_json, write_metadata_jsonl


def load_metadata(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def summarize_tau_metadata(metadata: List[Dict]) -> Dict[str, object]:
    def values_for(key: str) -> List[float]:
        values: List[float] = []
        for row in metadata:
            value = row.get(key)
            if value is not None:
                values.append(float(value))
        return values

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
        "tau_fixed_count": int(sum(bool(row.get("tau_is_fixed", False)) for row in metadata)),
    }
    summary.update(stats(values_for("tau_value"), "tau"))
    summary.update(stats(values_for("tau_base"), "tau_base"))
    summary.update(stats(values_for("delta_tau"), "delta_tau"))
    summary.update(stats(values_for("tau_ratio"), "tau_ratio"))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge CAPM-only taxonomy feature shards and rerun analysis."
    )
    parser.add_argument(
        "--shard-dir",
        action="append",
        required=True,
        help="Shard output directory produced by analyze_taxonomy_capm_only.py. Repeat for each shard.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Final merged output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shard_dirs = [Path(p).resolve() for p in args.shard_dir]
    output_dir = ensure_dir(Path(args.output_dir).resolve())

    arrays_by_feature: Dict[str, List[np.ndarray]] = {name: [] for name in CAPM_FEATURE_ORDER}
    masks_by_feature: Dict[str, List[np.ndarray]] = {name: [] for name in CAPM_FEATURE_ORDER}
    metadata: List[Dict] = []
    task_summaries: List[Dict] = []

    for shard_dir in shard_dirs:
        npz_path = shard_dir / "taxonomy_features.npz"
        metadata_path = shard_dir / "taxonomy_features_metadata.jsonl"
        summary_path = shard_dir / "task_run_summary.json"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing shard features: {npz_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing shard metadata: {metadata_path}")
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing shard summary: {summary_path}")

        shard_npz = np.load(npz_path)
        for feature_name in CAPM_FEATURE_ORDER:
            arrays_by_feature[feature_name].append(shard_npz[feature_name])
            masks_by_feature[feature_name].append(shard_npz[f"{feature_name}_mask"])

        metadata.extend(load_metadata(metadata_path))
        with summary_path.open("r", encoding="utf-8") as f:
            task_summaries.extend(json.load(f).get("tasks", []))

    merged_arrays: Dict[str, np.ndarray] = {}
    merged_masks: Dict[str, np.ndarray] = {}
    npz_payload = {}
    for feature_name in CAPM_FEATURE_ORDER:
        merged_arrays[feature_name] = np.concatenate(arrays_by_feature[feature_name], axis=0)
        merged_masks[feature_name] = np.concatenate(masks_by_feature[feature_name], axis=0)
        npz_payload[feature_name] = merged_arrays[feature_name]
        npz_payload[f"{feature_name}_mask"] = merged_masks[feature_name]

    np.savez_compressed(output_dir / "taxonomy_features.npz", **npz_payload)
    write_metadata_jsonl(output_dir / "taxonomy_features_metadata.jsonl", metadata)
    overall_summary = {
        "num_feature_records": len(metadata),
        "capm_valid": int(sum(bool(row.get("capm_available")) for row in metadata)),
        "inference_failed": int(sum(bool(row.get("inference_failed")) for row in metadata)),
    }
    overall_summary.update(summarize_tau_metadata(metadata))
    save_json(
        output_dir / "task_run_summary.json",
        {
            "tasks": task_summaries,
            "overall": overall_summary,
        },
    )
    analyze_features(merged_arrays, merged_masks, metadata, output_dir / "analysis")

    print(f"Merged {len(shard_dirs)} shards into: {output_dir}")


if __name__ == "__main__":
    main()
