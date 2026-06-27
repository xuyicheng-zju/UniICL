#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import json
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_TARGETS = (
    "AIGI-Holmes",
    "AVA",
    "World-Aware Planning",
    "LAION-HR",
    "T2I",
    "I2I",
    "degraded",
    "Concept",
    "Chain-of-Editing",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package UniICL image folders into HF-friendly tar shards."
    )
    parser.add_argument(
        "--release-root",
        default=".",
        help="Path to UniICL-760K release root. Archives are written under release_root/image_archives by default.",
    )
    parser.add_argument(
        "--source-root",
        default=None,
        help="Root directory containing image folders. The script tries <source>/<target>/images then <source>/<target>.",
    )
    parser.add_argument(
        "--source-dir",
        action="append",
        default=[],
        help="Explicit target-to-source mapping in the form TARGET=/absolute/path/to/files . Can be passed multiple times.",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=list(DEFAULT_TARGETS),
        help="Image targets to package.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write tar shards. Defaults to <release-root>/image_archives.",
    )
    parser.add_argument(
        "--max-size-gb",
        type=float,
        default=9.5,
        help="Maximum uncompressed payload size per tar shard.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of parallel tar-writing workers.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan shards without writing tar files.",
    )
    return parser.parse_args()


def parse_source_dir_overrides(items: list[str]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --source-dir value: {item}")
        target, path = item.split("=", 1)
        overrides[target] = Path(path).resolve()
    return overrides


def resolve_source_dir(source_root: Path | None, overrides: dict[str, Path], target: str) -> Path:
    if target in overrides:
        candidate = overrides[target]
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(f"Explicit source dir for '{target}' does not exist: {candidate}")
    if source_root is None:
        raise FileNotFoundError(
            f"No source directory provided for target '{target}'. Use --source-root or --source-dir {target}=..."
        )
    candidates = [
        source_root / target / "images",
        source_root / target,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve source directory for target '{target}' under {source_root}"
    )


def collect_files(source_dir: Path) -> list[Path]:
    files = [
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.name != ".DS_Store"
    ]
    files.sort()
    return files


def shard_files(files: list[Path], max_size_bytes: int) -> list[list[Path]]:
    shards: list[list[Path]] = []
    current: list[Path] = []
    current_size = 0

    for path in files:
        size = path.stat().st_size
        if current and current_size + size > max_size_bytes:
            shards.append(current)
            current = []
            current_size = 0
        current.append(path)
        current_size += size

    if current:
        shards.append(current)
    return shards


def write_tar(archive_path: Path, target: str, source_dir: Path, files: list[Path]) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="w") as tar:
        for path in files:
            rel = path.relative_to(source_dir).as_posix()
            arcname = f"images/{target}/{rel}"
            tar.add(path, arcname=arcname, recursive=False)


def write_tar_job(job: tuple[Path, str, Path, list[Path], int]) -> tuple[Path, int]:
    archive_path, target, source_dir, files, shard_bytes = job
    write_tar(archive_path, target, source_dir, files)
    return archive_path, shard_bytes


def main() -> None:
    args = parse_args()
    release_root = Path(args.release_root).resolve()
    source_root = Path(args.source_root).resolve() if args.source_root else None
    source_overrides = parse_source_dir_overrides(args.source_dir)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else release_root / "image_archives"
    max_size_bytes = int(args.max_size_gb * (1024 ** 3))

    summary: dict[str, object] = {
        "release_root": str(release_root),
        "source_root": str(source_root) if source_root else None,
        "output_dir": str(output_dir),
        "max_size_bytes": max_size_bytes,
        "workers": args.workers,
        "targets": {},
    }
    jobs: list[tuple[Path, str, Path, list[Path], int]] = []

    for target in args.targets:
        source_dir = resolve_source_dir(source_root, source_overrides, target)
        files = collect_files(source_dir)
        shards = shard_files(files, max_size_bytes)

        target_summary = {
            "source_dir": str(source_dir),
            "file_count": len(files),
            "total_bytes": sum(path.stat().st_size for path in files),
            "archive_count": len(shards),
            "archives": [],
        }

        print(f"[{target}] files={len(files)} archives={len(shards)} source={source_dir}")
        for idx, shard in enumerate(shards, start=1):
            archive_name = f"{target.replace(' ', '_').lower()}-{idx:05d}.tar"
            archive_path = output_dir / target / archive_name
            shard_bytes = sum(path.stat().st_size for path in shard)
            target_summary["archives"].append(
                {
                    "archive": str(archive_path),
                    "file_count": len(shard),
                    "payload_bytes": shard_bytes,
                }
            )
            print(f"  - {archive_name}: files={len(shard)} payload_bytes={shard_bytes}")
            if not args.dry_run:
                jobs.append((archive_path, target, source_dir, shard, shard_bytes))

        summary["targets"][target] = target_summary

    if not args.dry_run and jobs:
        print(f"\nWriting {len(jobs)} tar shards with {args.workers} workers ...")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_job = {executor.submit(write_tar_job, job): job for job in jobs}
            for future in as_completed(future_to_job):
                archive_path, shard_bytes = future.result()
                print(f"[done] {archive_path} bytes={shard_bytes}")

    summary_path = output_dir / "image_archives_manifest.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nsummary={summary_path}")


if __name__ == "__main__":
    main()
