"""Public release module documentation."""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

from public_path_config import DEFAULT_DINO_MODEL

# ============================================================

# ============================================================

class DINOv3Extractor:
    """Public release documentation."""

    def __init__(self, model_name: str = DEFAULT_DINO_MODEL, device: Optional[str] = None):
        from transformers import AutoImageProcessor, AutoModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DINOv3] Loading model {model_name} → {self.device}")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print(f"[DINOv3] Model loaded")

    @torch.no_grad()
    def extract(self, image_path: str) -> Optional[torch.Tensor]:
        """Public release documentation."""
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"  [WARN] Failed to read image {image_path}: {e}")
            return None

        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)


        feat = getattr(outputs, "pooler_output", None)
        if feat is None:
            feat = outputs.last_hidden_state[:, 0, :]  # (1, D)
        feat = F.normalize(feat, dim=-1)
        return feat.squeeze(0).cpu()  # (D,)

    @torch.no_grad()
    def extract_batch(self, image_paths: list) -> list:
        """Public release documentation."""
        results = []
        for p in image_paths:
            results.append(self.extract(p))
        return results

    @staticmethod
    def cosine_similarity(feat_a: torch.Tensor, feat_b: torch.Tensor) -> float:
        """Public release documentation."""
        if feat_a is None or feat_b is None:
            return None
        sim = torch.dot(feat_a, feat_b).item()

        return float(np.clip(sim, -1.0, 1.0))


# ============================================================

# ============================================================

def resolve_generated_path(raw_path: str, json_file: Path, benchmark_root: Path) -> Optional[str]:
    """Public release documentation."""
    if not raw_path:
        return None
    p = Path(raw_path)
    if p.is_absolute():
        return str(p) if p.exists() else None


    rel = str(p)
    if rel.startswith("./"):
        rel = rel[2:]


    candidate = benchmark_root / rel
    if candidate.exists():
        return str(candidate)


    candidate2 = json_file.parent / p
    if candidate2.exists():
        return str(candidate2)

    return None


def resolve_reference_path(raw_path: str) -> Optional[str]:
    """Public release documentation."""
    if not raw_path:
        return None
    p = Path(raw_path)
    if p.is_absolute():
        return str(p) if p.exists() else None

    return str(p) if p.exists() else None


# ============================================================

# ============================================================

def process_result_file(
    json_path: Path,
    extractor: DINOv3Extractor,
    benchmark_root: Path,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict:
    """Public release documentation."""
    print(f"\n{'='*70}")
    print(f"Processing: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        print("  [SKIP] results is empty")
        return {"file": str(json_path), "processed": 0, "skipped": 0, "mean": None}

    processed = 0
    skipped = 0
    sim_values = []

    for item in tqdm(results, desc=f"  {json_path.parent.parent.name}/{json_path.parent.name}", leave=False):

        existing = item.get("dinov3_similarity")
        if existing is not None and not overwrite:

            sim_values.append(float(existing))
            continue


        gen_raw = item.get("generated_path", "")
        gen_path = resolve_generated_path(gen_raw, json_path, benchmark_root)
        if not gen_path:
            tqdm.write(f"  [WARN] Generated image not found: {gen_raw}")
            skipped += 1
            item["dinov3_similarity"] = None
            continue


        ref_raw = item.get("query_reference", "")
        ref_path = resolve_reference_path(ref_raw)
        if not ref_path:
            tqdm.write(f"  [WARN] Reference image not found: {ref_raw}")
            skipped += 1
            item["dinov3_similarity"] = None
            continue


        feat_gen = extractor.extract(gen_path)
        feat_ref = extractor.extract(ref_path)
        sim = extractor.cosine_similarity(feat_gen, feat_ref)

        if sim is None:
            tqdm.write(f"  [WARN] Feature extraction failed, id={item.get('id', '?')}")
            skipped += 1
            item["dinov3_similarity"] = None
        else:
            item["dinov3_similarity"] = round(sim, 6)
            sim_values.append(sim)
            processed += 1


    mean_sim = float(np.mean(sim_values)) if sim_values else None


    data["mean_dinov3_similarity"] = round(mean_sim, 6) if mean_sim is not None else None


    if not dry_run:

        tmp_path = json_path.with_suffix(".tmp.json")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        shutil.move(str(tmp_path), str(json_path))

    total_valid = len(sim_values)
    print(f"  Processing: {processed} newly computed | skipped: {skipped} | valid total: {total_valid} items")
    if mean_sim is not None:
        print(f"  DINOv3 mean similarity: {mean_sim:.4f}")
    else:
        print(f"  DINOv3 mean similarity: N/A (no valid results)")

    return {
        "file": str(json_path),
        "processed_new": processed,
        "skipped": skipped,
        "total_valid": total_valid,
        "mean_dinov3_similarity": mean_sim,
    }


# ============================================================

# ============================================================

def find_result_files(search_dirs: list) -> list:
    """Public release documentation."""
    found = []
    for d in search_dirs:
        d = Path(d)
        if not d.exists():
            print(f"[WARN] Search directory does not exist: {d}")
            continue
        for pattern in ("analogical_editing_results.json", "visualcloze-g_results.json"):
            for p in sorted(d.rglob(pattern)):
                found.append(p)
    return sorted(set(found))


# ============================================================

# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analogical Editing post-evaluation DINOv3 similarity tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )


    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--search-dirs",
        nargs="+",
        metavar="DIR",
        help="Recursively search for analogical_editing_results.json files under the given directories",
    )
    input_group.add_argument(
        "--result-files",
        nargs="+",
        metavar="FILE",
        help="Directly specify analogical_editing_results.json files to process",
    )

    parser.add_argument(
        "--benchmark-root",
        type=str,
        default=None,
        help="UniICL-Bench root used to resolve relative generated_path values. "
             "Defaults to the script directory or the parent of search-dirs.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_DINO_MODEL,
        help="DINOv3 model path or Hugging Face identifier",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device, for example cuda:0 or cpu (auto-detected by default)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute items that already contain dinov3_similarity",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute only and skip file writes",
    )

    args = parser.parse_args()


    if args.result_files:
        json_files = [Path(f) for f in args.result_files]
        missing = [f for f in json_files if not f.exists()]
        if missing:
            print(f"[ERROR] The following files do not exist: {[str(f) for f in missing]}")
            sys.exit(1)
    else:
        json_files = find_result_files(args.search_dirs)
        if not json_files:
            print(f"[ERROR] No analogical_editing_results.json files found under: {args.search_dirs}")
            sys.exit(1)

    print(f"Found {len(json_files)} result files:")
    for f in json_files:
        print(f"  {f}")


    if args.benchmark_root:
        benchmark_root = Path(args.benchmark_root).resolve()
    else:

        benchmark_root = Path(__file__).resolve().parent
    print(f"\n[INFO] benchmark_root = {benchmark_root}")


    extractor = DINOv3Extractor(model_name=args.model, device=args.device)


    all_stats = []
    for json_path in json_files:
        stats = process_result_file(
            json_path=json_path,
            extractor=extractor,
            benchmark_root=benchmark_root,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        all_stats.append(stats)


    print(f"\n{'='*70}")
    print("DINOv3 similarity summary report")
    print(f"{'='*70}")
    print(f"{'File path':<60} {'DINOv3 mean':>12} {'Valid':>8}")
    print(f"{'-'*70}")

    for s in all_stats:
        short_path = Path(s["file"])

        parts = short_path.parts
        if len(parts) >= 3:
            display = "/".join(parts[-3:])
        else:
            display = str(short_path)
        mean_val = s["mean_dinov3_similarity"]
        mean_str = f"{mean_val:.4f}" if mean_val is not None else "  N/A  "
        print(f"  {display:<58} {mean_str:>12} {s['total_valid']:>8}")

    print(f"{'='*70}")


    from collections import defaultdict
    model_shot_stats = defaultdict(dict)
    for s in all_stats:
        p = Path(s["file"])
        parts = p.parts
        if len(parts) >= 3:
            model_name = parts[-3]
            shot_key = parts[-2]
        else:
            model_name = "unknown"
            shot_key = "?shot"
        model_shot_stats[model_name][shot_key] = s["mean_dinov3_similarity"]

    if len(model_shot_stats) > 1 or any(len(v) > 1 for v in model_shot_stats.values()):
        print("\nGrouped by model x shot:")
        for model, shots in sorted(model_shot_stats.items()):
            print(f"\n  [{model}]")
            for shot, val in sorted(shots.items()):
                val_str = f"{val:.4f}" if val is not None else "N/A"
                print(f"    {shot:>8}: {val_str}")

    if args.dry_run:
        print("\n[DRY-RUN] No files were written")
    else:
        print(f"\nUpdated {len(json_files)} result files with dinov3_similarity values")


if __name__ == "__main__":
    main()
