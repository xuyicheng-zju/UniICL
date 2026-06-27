#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic file manifest.")
    parser.add_argument("--root", required=True, help="Root directory to scan.")
    parser.add_argument("--output", required=True, help="Output TSV path.")
    parser.add_argument(
        "--exclude-name",
        action="append",
        default=[".DS_Store"],
        help="File names to exclude. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, int]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name in args.exclude_name:
            continue
        rel = path.relative_to(root).as_posix()
        rows.append((rel, path.stat().st_size))

    rows.sort()

    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["relative_path", "size_bytes"])
        writer.writerows(rows)

    total_bytes = sum(size for _, size in rows)
    print(f"root={root}")
    print(f"files={len(rows)}")
    print(f"bytes={total_bytes}")
    print(f"manifest={output}")


if __name__ == "__main__":
    main()
