#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


PLACEHOLDER_PREFIX = "__SET_"


OPEN_SOURCE_ROOT = Path(__file__).resolve().parent
UNIICL_760K_ROOT = OPEN_SOURCE_ROOT / "UniICL-760K"
UNIICL_BENCH_ROOT = OPEN_SOURCE_ROOT / "UniICL-Bench"
UNIICL_ROOT = OPEN_SOURCE_ROOT / "UniICL"
HPSV3_VENDOR_ROOT = UNIICL_BENCH_ROOT / "third_party" / "HPSv3"


# Public release placeholders. Edit these paths for local training/evaluation.
UNIICL_BASE_MODEL = "__SET_UNIICL_BASE_MODEL__"
UNIICL_FINETUNED_MODEL = "__SET_UNIICL_FINETUNED_MODEL__"
UNIICL_TARGET_CHECKPOINT = "__SET_TARGET_CHECKPOINT_DIR__"

UNIWORLD_MODEL = "__SET_UNIWORLD_MODEL__"
QWEN3VL_MODEL = "__SET_QWEN3VL_MODEL__"
QWEN25VL_MODEL = "__SET_QWEN25VL_MODEL__"
INTERNVL_MODEL = "OpenGVLab/InternVL3_5-8B"
NEXUSGEN_MODEL = "__SET_NEXUSGEN_MODEL__"
OVISU1_MODEL = "__SET_OVISU1_MODEL__"
FLUX_MODEL = "__SET_FLUX_MODEL__"
SIGLIP_MODEL = "__SET_SIGLIP_MODEL__"

HPSV3_CONFIG = HPSV3_VENDOR_ROOT / "hpsv3" / "config" / "HPSv3_7B.yaml"
HPSV3_CHECKPOINT = "__SET_HPSV3_CHECKPOINT__"
QALIGN_MODEL = "__SET_QALIGN_MODEL__"
JUDGE_MODEL = "__SET_JUDGE_MODEL__"
JUDGE_API_BASE = "__SET_JUDGE_API_BASE__"
DINO_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"


def _normalize_value(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def is_placeholder(value: object) -> bool:
    return _normalize_value(value).startswith(PLACEHOLDER_PREFIX)


def get_value(name: str) -> str:
    if name not in globals():
        raise KeyError(f"Unknown config key: {name}")
    return _normalize_value(globals()[name])


def to_dict() -> dict[str, str]:
    keys = [
        "OPEN_SOURCE_ROOT",
        "UNIICL_760K_ROOT",
        "UNIICL_BENCH_ROOT",
        "UNIICL_ROOT",
        "HPSV3_VENDOR_ROOT",
        "UNIICL_BASE_MODEL",
        "UNIICL_FINETUNED_MODEL",
        "UNIICL_TARGET_CHECKPOINT",
        "UNIWORLD_MODEL",
        "QWEN3VL_MODEL",
        "QWEN25VL_MODEL",
        "INTERNVL_MODEL",
        "NEXUSGEN_MODEL",
        "OVISU1_MODEL",
        "FLUX_MODEL",
        "SIGLIP_MODEL",
        "HPSV3_CONFIG",
        "HPSV3_CHECKPOINT",
        "QALIGN_MODEL",
        "JUDGE_MODEL",
        "JUDGE_API_BASE",
        "DINO_MODEL",
    ]
    return {key: get_value(key) for key in keys}


def main() -> int:
    parser = argparse.ArgumentParser(description="Print centralized UniICL open-source path settings.")
    parser.add_argument("--get", dest="key", help="Return a single config value by name.")
    parser.add_argument("--json", action="store_true", help="Print all config values as JSON.")
    args = parser.parse_args()

    if args.key:
        print(get_value(args.key))
        return 0

    if args.json:
        print(json.dumps(to_dict(), indent=2, sort_keys=True))
        return 0

    for key, value in to_dict().items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
