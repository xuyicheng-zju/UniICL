#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image
import numpy as np


BENCH_ROOT = Path(__file__).resolve().parent
OPEN_SOURCE_ROOT = BENCH_ROOT.parent
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))
if str(OPEN_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(OPEN_SOURCE_ROOT))
if str(OPEN_SOURCE_ROOT / "UniICL") not in sys.path:
    sys.path.insert(0, str(OPEN_SOURCE_ROOT / "UniICL"))

from analyze_taxonomy_capm_only import (
    CapmOnlyFeatureInferencer,
    analyze_features,
    build_feature_arrays,
    feature_dims_from_model,
    summarize_tau_records,
)
from analyze_taxonomy_features import (
    disable_external_judges_and_scorers,
    ensure_dir,
    save_json,
    truncate_capm_gates,
    write_metadata_jsonl,
)
from eval_bagel import CAPM_ABLATION_CHOICES, load_bagel_model
from public_path_config import (
    DATASETS_ROOT,
    DEFAULT_UNIICL_BASE_MODEL,
    DEFAULT_UNIICL_FINETUNED_MODEL,
)
from utils.icl import build_icl_input


TASK_SPECS = {
    "visual_grounding": {
        "taxonomy": "Perception",
        "rel_path": "Visual-Grounding/visual_grounding_train_icl.jsonl",
        "loader": "jsonl",
        "builder": "generic_und",
    },
    "style_aware_caption": {
        "taxonomy": "Imitation",
        "rel_path": "Style-Aware-Caption/style_aware_caption_train_icl.jsonl",
        "loader": "jsonl",
        "builder": "generic_und",
    },
    "fast_concept_mapping": {
        "taxonomy": "Conception",
        "rel_path": "Fast-Concept-Mapping/fast_concept_mapping_train_icl.jsonl",
        "loader": "jsonl",
        "builder": "generic_und",
    },
    "world_aware_planning": {
        "taxonomy": "Deduction",
        "rel_path": "World-Aware-Planning/world_aware_planning_train_icl.json",
        "loader": "json",
        "builder": "planning_train",
    },
    "analogical_inference": {
        "taxonomy": "Analogy",
        "rel_path": "Analogical-Inference/analogical_inference_train_icl.jsonl",
        "loader": "jsonl",
        "builder": "generic_und",
    },
    "aesthetic_assessment": {
        "taxonomy": "Discernment",
        "rel_path": "Aesthetic-Assessment/aesthetic_assessment_train_icl.jsonl",
        "loader": "jsonl",
        "builder": "generic_und",
    },
}


TASK_ALIASES = {
    "grounding": "visual_grounding",
    "caption": "style_aware_caption",
    "fcm": "fast_concept_mapping",
    "planning": "world_aware_planning",
    "analogy": "analogical_inference",
    "esthetic": "aesthetic_assessment",
    "aesthetic": "aesthetic_assessment",
}


DEFAULT_TASK_ORDER = [
    "visual_grounding",
    "style_aware_caption",
    "fast_concept_mapping",
    "world_aware_planning",
    "analogical_inference",
    "aesthetic_assessment",
]


def normalize_task_name(task: str) -> str:
    normalized = str(task).strip().lower().replace("-", "_")
    normalized = TASK_ALIASES.get(normalized, normalized)
    if normalized not in TASK_SPECS:
        raise ValueError(f"Unknown task: {task}")
    return normalized


def resolve_tasks(task_args: Sequence[str]) -> List[str]:
    if len(task_args) == 1 and str(task_args[0]).lower() == "all":
        return list(DEFAULT_TASK_ORDER)
    return [normalize_task_name(task) for task in task_args]


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "samples" in data and isinstance(data["samples"], list):
            return data["samples"]
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
    raise ValueError(f"Unsupported JSON training format: {path}")


def load_samples(task: str, datasets_root: Path) -> List[Dict]:
    spec = TASK_SPECS[task]
    path = datasets_root / spec["rel_path"]
    if spec["loader"] == "jsonl":
        return load_jsonl(path)
    return load_json(path)


def sample_examples(samples: Sequence[Dict], limit: int, seed: int) -> List[Tuple[int, Dict]]:
    indexed = list(enumerate(samples))
    if limit <= 0 or limit >= len(indexed):
        return indexed
    rng = random.Random(seed)
    chosen = sorted(rng.sample(range(len(indexed)), limit))
    return [indexed[idx] for idx in chosen]


def resolve_image_path(image_root: Path, rel_path: str) -> str:
    path = Path(rel_path)
    if path.is_absolute():
        return str(path)
    return str(image_root / rel_path)


def build_generic_understanding_input(sample: Dict, image_root: Path) -> List[object]:
    demos = sample.get("demos", [])
    target_image_path = resolve_image_path(image_root, sample["image_name"])
    target_question = sample.get("instruction") or sample.get("text", "")
    return build_icl_input(demos, str(image_root), target_image_path, target_question)


def build_planning_train_input(sample: Dict, image_root: Path) -> List[object]:
    convs = sample.get("conversations", [])
    raw_images = sample.get("images", [])
    if not convs or not raw_images:
        raise ValueError("Planning sample missing conversations or images")

    image_paths = [resolve_image_path(image_root, p) for p in raw_images]
    for path in image_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing planning image: {path}")

    task_instruction = ""
    if convs and convs[0].get("from") in ["human", "User"]:
        task_instruction = convs[0].get("value", "")

    history_pairs: List[Tuple[str, str]] = []
    img_idx = 0
    pending_image: str | None = None
    for conv in convs[1:]:
        role = conv.get("from")
        value = conv.get("value", "")
        if role == "observation":
            if img_idx >= len(image_paths):
                break
            pending_image = image_paths[img_idx]
            img_idx += 1
        elif role in ["gpt", "Assistant"] and pending_image is not None:
            history_pairs.append((pending_image, value))
            pending_image = None

    if not history_pairs:
        raise ValueError("Planning sample has no observation/assistant pairs")

    input_list: List[object] = []
    if task_instruction:
        input_list.append(f"User: {task_instruction}\nAssistant: Sure!")

    for image_path, assistant_text in history_pairs[:-1]:
        input_list.append("\nUser: ")
        input_list.append(Image.open(image_path).convert("RGB"))
        input_list.append(f"\nAssistant: {assistant_text}")

    query_image_path, _ = history_pairs[-1]
    input_list.append("\nUser: ")
    input_list.append(Image.open(query_image_path).convert("RGB"))
    input_list.append("\nAssistant:")
    return input_list


def build_input_list(task: str, sample: Dict, image_root: Path) -> List[object]:
    builder = TASK_SPECS[task]["builder"]
    if builder == "planning_train":
        return build_planning_train_input(sample, image_root)
    return build_generic_understanding_input(sample, image_root)


def extract_sample_id(task: str, item: Dict, index: int) -> str:
    for key in ("id", "original_id", "image_name"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
        if key == "id" and isinstance(value, int):
            return str(value)
    return f"{task}_{index:06d}"


def run_training_task(
    inferencer: CapmOnlyFeatureInferencer,
    task: str,
    datasets_root: Path,
    samples_per_task: int,
    sample_seed: int,
) -> Tuple[List[Dict], Dict[str, object]]:
    spec = TASK_SPECS[task]
    taxonomy = spec["taxonomy"]
    raw_samples = load_samples(task, datasets_root)
    selected = sample_examples(raw_samples, samples_per_task, sample_seed)

    merged_records: List[Dict] = []
    failures = 0

    for local_idx, (raw_index, sample) in enumerate(selected):
        inferencer.set_run_context(task, taxonomy)
        inferencer.drain_episode_records()

        inference_failed = False
        error_message = None
        try:
            input_list = build_input_list(task, sample, datasets_root)
            inferencer.interleave_inference(
                input_lists=input_list,
                understanding_output=True,
                think=False,
            )
        except Exception as exc:
            inference_failed = True
            error_message = str(exc)
            failures += 1

        episode_records = inferencer.drain_episode_records()
        if episode_records:
            episode_record = episode_records[-1]
        else:
            episode_record = {
                "episode_index": local_idx + 1,
                "task": task,
                "taxonomy": taxonomy,
                "capm_available": False,
                "demo_count": 0,
                "episode_error": error_message,
                "features": {},
            }

        merged_record = dict(episode_record)
        merged_record["sample_id"] = extract_sample_id(task, sample, raw_index)
        merged_record["train_index"] = int(raw_index)
        merged_record["inference_failed"] = inference_failed
        merged_record["episode_error"] = error_message or merged_record.get("episode_error")
        merged_records.append(merged_record)

    summary = {
        "task": task,
        "taxonomy": taxonomy,
        "num_source_samples": len(raw_samples),
        "num_selected_samples": len(selected),
        "capm_valid": sum("z_pool" in r["features"] for r in merged_records),
        "inference_failed": failures,
    }
    summary.update(summarize_tau_records(merged_records))
    return merged_records, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CAPM-only taxonomy feature extraction on UniICL-760K understanding training tasks."
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
        "--datasets-root",
        type=str,
        default=DATASETS_ROOT,
        help="Root directory of UniICL-760K.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./taxonomy_capm_train_understanding",
        help="Directory for extracted features and plots.",
    )
    parser.add_argument(
        "--task",
        nargs="+",
        default=["all"],
        help="Subset of tasks to run, or 'all'.",
    )
    parser.add_argument(
        "--samples-per-task",
        type=int,
        default=5000,
        help="Maximum number of training samples to draw per task.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed for deterministic training sample selection.",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Only save raw arrays and metadata.",
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
    datasets_root = Path(args.datasets_root).resolve()
    output_dir = ensure_dir(Path(args.output_dir).resolve())

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
        print(
            f"Running training task={task} taxonomy={TASK_SPECS[task]['taxonomy']} "
            f"samples_per_task={args.samples_per_task}"
        )
        print("=" * 80)
        merged_records, summary = run_training_task(
            inferencer=inferencer,
            task=task,
            datasets_root=datasets_root,
            samples_per_task=args.samples_per_task,
            sample_seed=args.sample_seed,
        )
        all_records.extend(merged_records)
        task_run_summary.append(summary)

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
        "inference_failed": sum(int(r["inference_failed"]) for r in all_records),
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
