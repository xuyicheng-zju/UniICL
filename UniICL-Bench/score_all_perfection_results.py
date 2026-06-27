#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

from tqdm import tqdm

from public_path_config import DEFAULT_QALIGN_MODEL, UNICL_GEN_ROOT

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))


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


DEFAULT_SEARCH_ROOTS = [
    SCRIPT_DIR / "eval_results_capm_final",
    SCRIPT_DIR / "eval_results_robust",
    SCRIPT_DIR / "eval_results",
]
DEFAULT_BENCHMARK_DIR = SCRIPT_DIR / "Visual-Refinement"
DEFAULT_CANONICAL_DATA = DEFAULT_BENCHMARK_DIR / "visual_refinement_benchmark.jsonl"
DEFAULT_DEGRADED_DIR = Path(UNICL_GEN_ROOT) / "degraded"
DEFAULT_GT_DIR = Path(UNICL_GEN_ROOT) / "T2I"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score every discovered Visual Refinement run in one pass. "
            "Q-Align is loaded once and benchmark variants are auto-matched."
        )
    )
    parser.add_argument(
        "--search-root",
        action="append",
        default=[],
        help=(
            "Search root to scan recursively for visual_refinement_generated directories. "
            "Can be repeated. Defaults to eval_results_capm_final, eval_results_robust, eval_results."
        ),
    )
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--degraded-dir", type=Path, default=DEFAULT_DEGRADED_DIR)
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--qalign-model", type=str, default=DEFAULT_QALIGN_MODEL)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=SCRIPT_DIR / "visual_refinement_scoring_summary.json",
        help="Path to save the summary JSON for all scored runs.",
    )
    return parser.parse_args()


def find_generated_dirs(search_roots: Iterable[Path]) -> List[Path]:
    found = set()
    for root in search_roots:
        if not root.exists():
            continue
        for dirname in ("visual_refinement_generated", "perfection_generated"):
            for path in root.rglob(dirname):
                if path.is_dir():
                    found.add(path.resolve())
    return sorted(found)


def resolve_variant(generated_dir: Path) -> str | None:
    parts = list(generated_dir.parts)
    if "eval_results_robust" in parts:
        idx = parts.index("eval_results_robust")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def resolve_benchmark_path(generated_dir: Path, benchmark_dir: Path) -> Path:
    variant = resolve_variant(generated_dir)
    if variant:
        candidate = benchmark_dir / f"visual_refinement_benchmark_{variant}.jsonl"
        if candidate.exists():
            return candidate
    return benchmark_dir / "visual_refinement_benchmark.jsonl"


def load_benchmark_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def describe_run(generated_dir: Path) -> Dict[str, str | None]:
    parts = generated_dir.parts
    info = {
        "variant": None,
        "model": None,
        "shot": None,
    }
    if "eval_results_robust" in parts:
        idx = parts.index("eval_results_robust")
        if idx + 3 < len(parts):
            info["variant"] = parts[idx + 1]
            info["model"] = parts[idx + 2]
            info["shot"] = parts[idx + 3]
    elif "eval_results_capm_final" in parts:
        idx = parts.index("eval_results_capm_final")
        if idx + 2 < len(parts):
            info["variant"] = "standard"
            info["model"] = parts[idx + 1]
            info["shot"] = parts[idx + 2]
    elif "eval_results" in parts:
        idx = parts.index("eval_results")
        if idx + 2 < len(parts):
            info["variant"] = "standard"
            info["model"] = parts[idx + 1]
            info["shot"] = parts[idx + 2]
    return info


def score_one_dir(
    generated_dir: Path,
    benchmark_rows: List[dict],
    qalign_model,
    output_path: Path,
) -> Dict[str, object]:
    results = []
    total_efficiency = 0.0
    valid_count = 0
    missing_count = 0

    for item in tqdm(benchmark_rows, desc=f"Scoring {generated_dir.parent.name}", leave=False):
        sample_id = item.get("sample_id", item.get("id", "unknown"))
        gen_img_path = generated_dir / f"perfected_{sample_id}.png"

        if not gen_img_path.exists():
            missing_count += 1
            results.append(
                {
                    "sample_id": sample_id,
                    "generated_path": str(gen_img_path),
                    "qalign_score_input": item.get("score_l"),
                    "qalign_score_output": None,
                    "qalign_score_output_quality": None,
                    "qalign_score_output_aesthetics": None,
                    "qalign_score_gt": item.get("score_h"),
                    "efficiency": None,
                    "error": "Generated image not found",
                }
            )
            continue

        s_in = item.get("score_l")
        s_gt = item.get("score_h")
        try:
            score_dict = compute_qalign_score(str(gen_img_path), qalign_model)
            if score_dict:
                s_out = score_dict["total_score"]
                s_out_quality = score_dict["quality_score"]
                s_out_aesthetics = score_dict["aesthetics_score"]
            else:
                s_out = None
                s_out_quality = None
                s_out_aesthetics = None
        except Exception as exc:  # noqa: BLE001
            s_out = None
            s_out_quality = None
            s_out_aesthetics = None
            error_msg = str(exc)
        else:
            error_msg = None

        efficiency = None
        if s_in is not None and s_out is not None:
            denominator = 5.0 - s_in
            if abs(denominator) > 1e-6:
                efficiency = (s_out - s_in) / denominator * 100
                total_efficiency += efficiency
                valid_count += 1

        row = {
            "sample_id": sample_id,
            "generated_path": str(gen_img_path),
            "qalign_score_input": s_in,
            "qalign_score_output": s_out,
            "qalign_score_output_quality": s_out_quality,
            "qalign_score_output_aesthetics": s_out_aesthetics,
            "qalign_score_gt": s_gt,
            "efficiency": efficiency,
        }
        if error_msg:
            row["error"] = error_msg
        results.append(row)

    mean_efficiency = total_efficiency / valid_count if valid_count > 0 else 0.0
    result_data = {
        "mean_efficiency": mean_efficiency,
        "valid_count": valid_count,
        "total_count": len(benchmark_rows),
        "missing_count": missing_count,
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    return result_data


def main() -> None:
    args = parse_args()

    search_roots = [Path(p).resolve() for p in args.search_root] if args.search_root else DEFAULT_SEARCH_ROOTS
    benchmark_dir = args.benchmark_dir.resolve()
    degraded_dir = args.degraded_dir.resolve()
    gt_dir = args.gt_dir.resolve()

    if not benchmark_dir.exists():
        raise FileNotFoundError(f"UniICL-Bench directory not found: {benchmark_dir}")
    if not degraded_dir.exists():
        raise FileNotFoundError(f"Degraded image directory not found: {degraded_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"GT image directory not found: {gt_dir}")

    generated_dirs = find_generated_dirs(search_roots)
    if not generated_dirs:
        print("No visual_refinement_generated directories found.")
        return

    print("=== Discovered Perfection Runs ===")
    for d in generated_dirs:
        print(f"  - {d.relative_to(REPO_ROOT)}")

    print(f"\nLoading Q-Align model on {args.device}...")
    qalign_model = load_qalign_model(model_path=args.qalign_model, device=args.device)
    print("Q-Align model loaded!")

    benchmark_cache: Dict[Path, List[dict]] = {}
    summary_rows = []

    for generated_dir in generated_dirs:
        output_path = generated_dir.parent / "visual_refinement_scores.json"
        if output_path.exists() and not args.overwrite:
            print(f"\nSkip existing: {output_path.relative_to(REPO_ROOT)}")
            with output_path.open("r", encoding="utf-8") as f:
                saved = json.load(f)
            result_data = saved
        else:
            data_path = resolve_benchmark_path(generated_dir, benchmark_dir)
            if data_path not in benchmark_cache:
                benchmark_cache[data_path] = load_benchmark_rows(data_path)
            print(
                f"\nScoring {generated_dir.relative_to(REPO_ROOT)} "
                f"using {data_path.relative_to(REPO_ROOT)} ..."
            )
            result_data = score_one_dir(
                generated_dir=generated_dir,
                benchmark_rows=benchmark_cache[data_path],
                qalign_model=qalign_model,
                output_path=output_path,
            )

        run_info = describe_run(generated_dir)
        summary_rows.append(
            {
                "generated_dir": str(generated_dir.relative_to(REPO_ROOT)),
                "output_path": str(output_path.relative_to(REPO_ROOT)),
                "variant": run_info["variant"],
                "model": run_info["model"],
                "shot": run_info["shot"],
                "mean_efficiency": result_data.get("mean_efficiency", 0.0),
                "valid_count": result_data.get("valid_count", 0),
                "total_count": result_data.get("total_count", 0),
                "missing_count": result_data.get("missing_count", 0),
            }
        )

    summary_path = args.summary_path.resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)

    print("\n=== Summary ===")
    print(f"Runs scored: {len(summary_rows)}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
