#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from public_path_config import DEFAULT_HPSV3_CHECKPOINT, DEFAULT_QALIGN_MODEL  # noqa: E402
from utils.scoring import (  # noqa: E402
    compute_hpsv3_score,
    compute_qalign_score,
    load_hpsv3_model,
    load_qalign_model,
)


DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "eval_results_capm_component"
DEFAULT_SHOTS = [0, 1, 2, 4, 8]
DEFAULT_TASKS = ["instructional_generation", "visual_refinement"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill missing metrics for CAPM component ablation runs. "
            "This script rescoring only the two generation-side tasks: "
            "instructional_generation (HPSv3) and visual_refinement (Q-Align)."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory containing CAPM component mode folders.",
    )
    parser.add_argument(
        "--mode",
        action="append",
        default=[],
        help=(
            "Specific CAPM ablation mode(s) to rescore. Can be repeated. "
            "Default: auto-discover all modes under --results-root."
        ),
    )
    parser.add_argument(
        "--shots",
        nargs="*",
        type=int,
        default=DEFAULT_SHOTS,
        help="Shot values to process. Default: 0 1 2 4 8",
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=DEFAULT_TASKS,
        default=[],
        help=(
            "Generation-side task(s) to rescore. Can be repeated. "
            "Default: instructional_generation and visual_refinement."
        ),
    )
    parser.add_argument(
        "--hps-checkpoint",
        type=str,
        default=DEFAULT_HPSV3_CHECKPOINT,
        help="HPSv3 checkpoint path for instructional_generation scoring.",
    )
    parser.add_argument(
        "--qalign-model",
        type=str,
        default=DEFAULT_QALIGN_MODEL,
        help="Q-Align model path for visual_refinement scoring.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for Q-Align. HPSv3 follows the current visible GPU.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional path to save a JSON summary. Defaults to <results-root>/rescored_gen_tasks_summary.json",
    )
    return parser.parse_args()


def discover_mode_dirs(results_root: Path, requested_modes: List[str]) -> Dict[str, Path]:
    if requested_modes:
        mode_dirs = {}
        for mode in requested_modes:
            candidate = results_root / mode / "uniicl"
            if not candidate.is_dir():
                raise FileNotFoundError(f"Mode directory not found: {candidate}")
            mode_dirs[mode] = candidate
        return mode_dirs

    mode_dirs = {}
    if not results_root.is_dir():
        raise FileNotFoundError(f"Results root not found: {results_root}")
    for child in sorted(results_root.iterdir()):
        candidate = child / "uniicl"
        if child.is_dir() and candidate.is_dir():
            mode_dirs[child.name] = candidate
    if not mode_dirs:
        raise FileNotFoundError(f"No mode directories found under {results_root}")
    return mode_dirs


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def resolve_generated_path(raw_path: str | None, result_path: Path) -> Path | None:
    if not raw_path:
        return None

    candidate = Path(raw_path)
    candidates = [candidate]
    if not candidate.is_absolute():
        candidates.extend(
            [
                result_path.parent / candidate,
                SCRIPT_DIR / candidate,
                REPO_ROOT / candidate,
            ]
        )

    for item in candidates:
        if item.exists():
            return item

    return None


def rescore_instructional_generation(result_path: Path, hps_model) -> dict:
    obj = load_json(result_path)
    results = obj.get("results", [])
    total_score = 0.0
    valid_count = 0
    missing_count = 0

    for row in tqdm(results, desc=f"{result_path.parent.parent.parent.name} {result_path.parent.name} instructional_generation", leave=False):
        gen_path = resolve_generated_path(row.get("generated_path"), result_path)
        prompt = row.get("prompt", row.get("instruction", ""))
        score = -1.0

        if gen_path is not None:
            score = compute_hpsv3_score(str(gen_path), prompt, hps_model)
            if score >= 0:
                total_score += score
                valid_count += 1
        else:
            missing_count += 1

        row["hpsv3_score"] = score

    mean_score = total_score / valid_count if valid_count > 0 else 0.0
    obj["mean_hpsv3_score"] = mean_score
    obj["valid_count"] = valid_count
    obj["total_count"] = len(results)
    obj["missing_count"] = missing_count
    dump_json(result_path, obj)

    return {
        "task": "instructional_generation",
        "result_path": str(result_path),
        "metric": mean_score,
        "valid_count": valid_count,
        "total_count": len(results),
        "missing_count": missing_count,
    }


def rescore_visual_refinement(result_path: Path, qalign_model) -> dict:
    obj = load_json(result_path)
    results = obj.get("results", [])
    total_efficiency = 0.0
    valid_count = 0
    missing_count = 0

    for row in tqdm(results, desc=f"{result_path.parent.parent.parent.name} {result_path.parent.name} visual_refinement", leave=False):
        gen_path = resolve_generated_path(row.get("generated_path"), result_path)
        s_in = row.get("qalign_score_input")
        s_out = None
        s_out_quality = None
        s_out_aesthetics = None

        if gen_path is not None:
            score_dict = compute_qalign_score(str(gen_path), qalign_model)
            if score_dict is not None:
                s_out = score_dict["total_score"]
                s_out_quality = score_dict["quality_score"]
                s_out_aesthetics = score_dict["aesthetics_score"]
        else:
            missing_count += 1

        efficiency = None
        if s_in is not None and s_out is not None:
            denominator = 5.0 - float(s_in)
            if abs(denominator) > 1e-6:
                efficiency = (float(s_out) - float(s_in)) / denominator * 100.0
                total_efficiency += efficiency
                valid_count += 1

        row["qalign_score_output"] = s_out
        row["qalign_score_output_quality"] = s_out_quality
        row["qalign_score_output_aesthetics"] = s_out_aesthetics
        row["efficiency"] = efficiency

    mean_efficiency = total_efficiency / valid_count if valid_count > 0 else 0.0
    obj["mean_efficiency"] = mean_efficiency
    obj["valid_count"] = valid_count
    obj["total_count"] = len(results)
    obj["missing_count"] = missing_count
    obj["scoring_skipped"] = False
    dump_json(result_path, obj)

    return {
        "task": "visual_refinement",
        "result_path": str(result_path),
        "metric": mean_efficiency,
        "valid_count": valid_count,
        "total_count": len(results),
        "missing_count": missing_count,
    }


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    summary_path = args.summary_path.resolve() if args.summary_path else (results_root / "rescored_gen_tasks_summary.json")
    requested_tasks = args.task or list(DEFAULT_TASKS)

    mode_dirs = discover_mode_dirs(results_root, args.mode)

    print("=== Discovered CAPM Component Modes ===")
    for mode, path in mode_dirs.items():
        print(f"  - {mode}: {path}")

    print("\n=== Requested Tasks ===")
    for task in requested_tasks:
        print(f"  - {task}")

    hps_model = None
    qalign_model = None
    if "instructional_generation" in requested_tasks:
        print(f"\nLoading HPSv3 model from {args.hps_checkpoint} ...")
        hps_model = load_hpsv3_model(args.hps_checkpoint)
        if hps_model is None:
            raise RuntimeError("Failed to load HPSv3 model.")

    if "visual_refinement" in requested_tasks:
        print(f"\nLoading Q-Align model from {args.qalign_model} on {args.device} ...")
        qalign_model = load_qalign_model(model_path=args.qalign_model, device=args.device)
        if qalign_model is None:
            raise RuntimeError("Failed to load Q-Align model.")

    summary_rows = []

    for mode, mode_dir in mode_dirs.items():
        for shot in args.shots:
            shot_dir = mode_dir / f"{shot}shot"
            if not shot_dir.is_dir():
                print(f"[skip] Missing shot directory: {shot_dir}")
                continue

            if "instructional_generation" in requested_tasks:
                t2i_result = shot_dir / "instructional_generation_results.json"
                if t2i_result.exists():
                    row = rescore_instructional_generation(t2i_result, hps_model)
                    row.update({"mode": mode, "shot": shot})
                    summary_rows.append(row)
                    print(
                        f"[ok] {mode} {shot}shot instructional_generation "
                        f"mean_hpsv3_score={row['metric']:.6f} valid={row['valid_count']}/{row['total_count']}"
                    )
                else:
                    print(f"[skip] Missing file: {t2i_result}")

            if "visual_refinement" in requested_tasks:
                vref_result = shot_dir / "visual_refinement_results.json"
                if vref_result.exists():
                    row = rescore_visual_refinement(vref_result, qalign_model)
                    row.update({"mode": mode, "shot": shot})
                    summary_rows.append(row)
                    print(
                        f"[ok] {mode} {shot}shot visual_refinement "
                        f"mean_efficiency={row['metric']:.6f} valid={row['valid_count']}/{row['total_count']}"
                    )
                else:
                    print(f"[skip] Missing file: {vref_result}")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(summary_path, summary_rows)

    print("\n=== Done ===")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
