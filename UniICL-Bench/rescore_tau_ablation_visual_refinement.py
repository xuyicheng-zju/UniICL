#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from public_path_config import DEFAULT_QALIGN_MODEL  # noqa: E402


def _load_scoring_module():
    scoring_path = SCRIPT_DIR / "utils" / "scoring.py"
    spec = importlib.util.spec_from_file_location("benchmark_scoring", scoring_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load scoring module from {scoring_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SCORING = _load_scoring_module()
compute_qalign_score = _SCORING.compute_qalign_score
load_qalign_model = _SCORING.load_qalign_model


DEFAULT_SHOTS = [1, 2, 4, 8]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rescore UniICL tau ablation visual_refinement runs with a single "
            "Q-Align model load."
        )
    )
    parser.add_argument(
        "--bench-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="UniICL-Bench root directory.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help=(
            "Optional explicit tau ablation shot root, e.g. "
            "eval_results_tau_ablation_1shot. If omitted, discover from --shots."
        ),
    )
    parser.add_argument(
        "--shot",
        dest="shots",
        nargs="*",
        type=int,
        default=DEFAULT_SHOTS,
        help="Shot values to process. Default: 1 2 4 8",
    )
    parser.add_argument(
        "--mode",
        action="append",
        default=[],
        help=(
            "Specific tau mode(s) to rescore, e.g. tau_0p1. "
            "Can be repeated. Default: auto-discover."
        ),
    )
    parser.add_argument(
        "--qalign-model",
        type=str,
        default=DEFAULT_QALIGN_MODEL,
        help="Q-Align model path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for Q-Align scoring.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help=(
            "Optional output JSON summary path. "
            "Default: <bench-dir>/rescore_tau_ablation_visual_refinement_summary.json"
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def discover_shot_roots(bench_dir: Path, explicit_root: Path | None, shots: List[int]) -> Dict[int, Path]:
    if explicit_root is not None:
        explicit_root = explicit_root.resolve()
        if not explicit_root.is_dir():
            raise FileNotFoundError(f"Results root not found: {explicit_root}")

        name = explicit_root.name
        if not (name.startswith("eval_results_tau_ablation_") and name.endswith("shot")):
            raise ValueError(
                f"Explicit results root should look like eval_results_tau_ablation_<K>shot, got: {name}"
            )
        shot_str = name[len("eval_results_tau_ablation_") : -len("shot")]
        shot = int(shot_str)
        return {shot: explicit_root}

    shot_roots: Dict[int, Path] = {}
    for shot in shots:
        root = (bench_dir / f"eval_results_tau_ablation_{shot}shot").resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Shot root not found: {root}")
        shot_roots[shot] = root
    return shot_roots


def discover_mode_dirs(shot_root: Path, requested_modes: List[str]) -> Dict[str, Path]:
    if requested_modes:
        mode_dirs = {}
        for mode in requested_modes:
            candidate = shot_root / mode / "uniicl"
            if not candidate.is_dir():
                raise FileNotFoundError(f"Mode directory not found: {candidate}")
            mode_dirs[mode] = candidate
        return mode_dirs

    mode_dirs = {}
    for child in sorted(shot_root.iterdir()):
        candidate = child / "uniicl"
        if child.is_dir() and candidate.is_dir():
            mode_dirs[child.name] = candidate
    if not mode_dirs:
        raise FileNotFoundError(f"No tau mode directories found under {shot_root}")
    return mode_dirs


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


def rescore_one_result(result_path: Path, qalign_model, desc: str) -> dict:
    obj = load_json(result_path)
    results = obj.get("results", [])
    total_efficiency = 0.0
    valid_count = 0
    missing_count = 0

    for row in tqdm(results, desc=desc, leave=False):
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
        "result_path": str(result_path),
        "metric": mean_efficiency,
        "valid_count": valid_count,
        "total_count": len(results),
        "missing_count": missing_count,
    }


def main() -> None:
    args = parse_args()
    bench_dir = args.bench_dir.resolve()
    if not bench_dir.is_dir():
        raise FileNotFoundError(f"UniICL-Bench directory not found: {bench_dir}")

    summary_path = (
        args.summary_path.resolve()
        if args.summary_path is not None
        else bench_dir / "rescore_tau_ablation_visual_refinement_summary.json"
    )

    shot_roots = discover_shot_roots(bench_dir, args.results_root, args.shots)

    print("=== Discovered Tau Ablation Shot Roots ===")
    for shot, shot_root in shot_roots.items():
        print(f"  - {shot}shot: {shot_root}")

    print(f"\nLoading Q-Align model on {args.device}...")
    qalign_model = load_qalign_model(model_path=args.qalign_model, device=args.device)
    print("Q-Align model loaded!")

    summary_rows = []
    for shot, shot_root in sorted(shot_roots.items()):
        mode_dirs = discover_mode_dirs(shot_root, args.mode)
        print(f"\n=== {shot}shot ===")
        for mode, mode_dir in sorted(mode_dirs.items()):
            result_path = mode_dir / f"{shot}shot" / "visual_refinement_results.json"
            if not result_path.is_file():
                print(f"[skip] Missing result file: {result_path}")
                continue

            print(f"[run] shot={shot} mode={mode}")
            row = rescore_one_result(
                result_path=result_path,
                qalign_model=qalign_model,
                desc=f"{shot}shot {mode}",
            )
            row.update({"shot": shot, "mode": mode, "task": "visual_refinement"})
            summary_rows.append(row)
            print(
                f"  mean_efficiency={row['metric']:.4f} "
                f"valid={row['valid_count']}/{row['total_count']} "
                f"missing={row['missing_count']}"
            )

    summary_obj = {
        "task": "visual_refinement",
        "rows": summary_rows,
    }
    dump_json(summary_path, summary_obj)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
