#!/usr/bin/env python3
"""Flatten sharded image folders into a single dataset-level directory.

This script is intended for public-release dataset layouts like:

    UniICL-760K/images/LAION-HR/subdir_1/xxx.jpg
    UniICL-760K/images/LAION-HR/subdir_2/yyy.jpg

and will move files into:

    UniICL-760K/images/LAION-HR/xxx.jpg
    UniICL-760K/images/LAION-HR/yyy.jpg

The default targets match the current operational need: LAION-HR and T2I.
"""

from __future__ import annotations

import argparse
import asyncio
import filecmp
import json
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_TARGETS = ("LAION-HR", "T2I")


@dataclass
class FileOpResult:
    status: str
    src: str
    dst: str | None = None
    message: str | None = None


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Flatten sharded image folders for UniICL public data layout."
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=script_dir / "images",
        help="Root directory that contains dataset image folders.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=list(DEFAULT_TARGETS),
        help="Dataset image folders to flatten.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=96,
        help="Number of concurrent filesystem workers.",
    )
    parser.add_argument(
        "--mode",
        choices=("move", "copy"),
        default="move",
        help="Whether to move or copy files into the flattened layout.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without modifying files.",
    )
    parser.add_argument(
        "--keep-empty-dirs",
        action="store_true",
        help="Do not remove empty shard directories after flattening.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=5000,
        help="Progress print interval.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional JSON path for a machine-readable summary.",
    )
    return parser.parse_args()


def iter_nested_files(dataset_root: Path) -> Iterable[Path]:
    for path in dataset_root.rglob("*"):
        if path.is_file() and path.parent != dataset_root:
            yield path


def files_match(path_a: Path, path_b: Path) -> bool:
    if path_a.stat().st_size != path_b.stat().st_size:
        return False
    return filecmp.cmp(path_a, path_b, shallow=False)


def process_one_file(src: Path, dataset_root: Path, mode: str, dry_run: bool) -> FileOpResult:
    dst = dataset_root / src.name

    if dst == src:
        return FileOpResult(status="skipped_same_path", src=str(src), dst=str(dst))

    if dst.exists():
        try:
            same_content = files_match(src, dst)
        except Exception as exc:  # pragma: no cover - defensive
            return FileOpResult(
                status="failed",
                src=str(src),
                dst=str(dst),
                message=f"Failed to compare files: {exc}",
            )
        if same_content:
            if dry_run:
                return FileOpResult(
                    status="duplicate_would_remove",
                    src=str(src),
                    dst=str(dst),
                    message="Identical file already exists at target root.",
                )
            if mode == "move":
                src.unlink()
                return FileOpResult(
                    status="duplicate_removed",
                    src=str(src),
                    dst=str(dst),
                    message="Removed duplicate nested file.",
                )
            return FileOpResult(
                status="duplicate_skipped",
                src=str(src),
                dst=str(dst),
                message="Identical file already exists at target root.",
            )
        return FileOpResult(
            status="collision",
            src=str(src),
            dst=str(dst),
            message="Target filename already exists with different content.",
        )

    if dry_run:
        return FileOpResult(status=f"would_{mode}", src=str(src), dst=str(dst))

    try:
        if mode == "move":
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))
        return FileOpResult(status=mode, src=str(src), dst=str(dst))
    except Exception as exc:  # pragma: no cover - defensive
        return FileOpResult(
            status="failed",
            src=str(src),
            dst=str(dst),
            message=str(exc),
        )


async def flatten_dataset(
    dataset_root: Path,
    workers: int,
    mode: str,
    dry_run: bool,
    cleanup_empty_dirs: bool,
    log_every: int,
) -> dict:
    nested_files = list(iter_nested_files(dataset_root))
    stats: Counter[str] = Counter()
    collisions: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    processed = 0

    if not nested_files:
        return {
            "dataset": dataset_root.name,
            "root": str(dataset_root),
            "discovered": 0,
            "stats": {},
            "collisions": [],
            "failures": [],
            "cleaned_dirs": 0,
            "elapsed_sec": 0.0,
        }

    queue: asyncio.Queue[Path | None] = asyncio.Queue()
    for src in nested_files:
        queue.put_nowait(src)
    for _ in range(workers):
        queue.put_nowait(None)

    start = time.time()

    async def worker() -> None:
        nonlocal processed
        while True:
            src = await queue.get()
            if src is None:
                queue.task_done()
                return
            result = await asyncio.to_thread(process_one_file, src, dataset_root, mode, dry_run)
            stats[result.status] += 1
            if result.status == "collision":
                collisions.append(
                    {
                        "src": result.src,
                        "dst": result.dst or "",
                        "message": result.message or "",
                    }
                )
            elif result.status == "failed":
                failures.append(
                    {
                        "src": result.src,
                        "dst": result.dst or "",
                        "message": result.message or "",
                    }
                )

            processed += 1
            if log_every > 0 and processed % log_every == 0:
                print(
                    f"[{dataset_root.name}] processed {processed}/{len(nested_files)} "
                    f"(moved={stats['move']}, copied={stats['copy']}, "
                    f"duplicates={stats['duplicate_removed'] + stats['duplicate_skipped']}, "
                    f"collisions={stats['collision']}, failed={stats['failed']})"
                )
            queue.task_done()

    tasks = [asyncio.create_task(worker()) for _ in range(workers)]
    await queue.join()
    await asyncio.gather(*tasks)

    cleaned_dirs = 0
    if cleanup_empty_dirs and not dry_run:
        # Remove empty directories bottom-up, but never the dataset root itself.
        for path in sorted(dataset_root.rglob("*"), reverse=True):
            if path.is_dir() and path != dataset_root:
                try:
                    path.rmdir()
                    cleaned_dirs += 1
                except OSError:
                    continue

    elapsed = time.time() - start
    return {
        "dataset": dataset_root.name,
        "root": str(dataset_root),
        "discovered": len(nested_files),
        "stats": dict(stats),
        "collisions": collisions,
        "failures": failures,
        "cleaned_dirs": cleaned_dirs,
        "elapsed_sec": round(elapsed, 3),
    }


def print_summary(report: dict) -> None:
    print(f"\n=== {report['dataset']} ===")
    print(f"root: {report['root']}")
    print(f"discovered nested files: {report['discovered']}")
    for key in sorted(report["stats"]):
        print(f"{key}: {report['stats'][key]}")
    print(f"cleaned empty directories: {report['cleaned_dirs']}")
    print(f"elapsed: {report['elapsed_sec']} sec")
    if report["collisions"]:
        print(f"collisions: {len(report['collisions'])}")
    if report["failures"]:
        print(f"failures: {len(report['failures'])}")


async def main_async() -> int:
    args = parse_args()
    images_root = args.images_root.resolve()

    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")

    reports = []
    for target in args.targets:
        dataset_root = images_root / target
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"Dataset image folder not found: {dataset_root}")
        print(
            f"\n[Start] target={target} mode={args.mode} workers={args.workers} "
            f"dry_run={args.dry_run}"
        )
        report = await flatten_dataset(
            dataset_root=dataset_root,
            workers=args.workers,
            mode=args.mode,
            dry_run=args.dry_run,
            cleanup_empty_dirs=not args.keep_empty_dirs,
            log_every=args.log_every,
        )
        print_summary(report)
        reports.append(report)

    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps({"reports": reports}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nWrote report to {args.report_json}")

    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
