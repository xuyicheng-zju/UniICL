#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_manifest(path: Path) -> dict[str, int]:
    records: dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            records[row["relative_path"]] = int(row["size_bytes"])
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two TSV file manifests.")
    parser.add_argument("--local", required=True, help="Local manifest TSV.")
    parser.add_argument("--remote", required=True, help="Remote manifest TSV.")
    parser.add_argument("--show-limit", type=int, default=20, help="How many examples to print per category.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local = read_manifest(Path(args.local))
    remote = read_manifest(Path(args.remote))

    local_keys = set(local)
    remote_keys = set(remote)

    missing_on_remote = sorted(local_keys - remote_keys)
    extra_on_remote = sorted(remote_keys - local_keys)
    size_mismatch = sorted(k for k in (local_keys & remote_keys) if local[k] != remote[k])

    print(f"local_files={len(local)}")
    print(f"remote_files={len(remote)}")
    print(f"missing_on_remote={len(missing_on_remote)}")
    print(f"extra_on_remote={len(extra_on_remote)}")
    print(f"size_mismatch={len(size_mismatch)}")

    limit = args.show_limit
    if missing_on_remote:
        print("\n[missing_on_remote]")
        for item in missing_on_remote[:limit]:
            print(item)
    if extra_on_remote:
        print("\n[extra_on_remote]")
        for item in extra_on_remote[:limit]:
            print(item)
    if size_mismatch:
        print("\n[size_mismatch]")
        for item in size_mismatch[:limit]:
            print(f"{item}\tlocal={local[item]}\tremote={remote[item]}")


if __name__ == "__main__":
    main()
