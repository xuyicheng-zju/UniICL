#!/usr/bin/env python3
from __future__ import annotations

import socket
from pathlib import Path


ROOT = Path(__file__).resolve().parent

from local_paths import (  # noqa: E402
    HPSV3_CHECKPOINT,
    JUDGE_API_BASE,
    JUDGE_MODEL,
    QALIGN_MODEL,
    UNIICL_760K_ROOT,
    UNIICL_BASE_MODEL,
    UNIICL_BENCH_ROOT,
    UNIICL_FINETUNED_MODEL,
    UNIICL_ROOT as UNIICL_CODE_ROOT,
    is_placeholder,
)


DATASETS_ROOT = Path(UNIICL_760K_ROOT)
BENCHMARK_ROOT = Path(UNIICL_BENCH_ROOT)
UNIICL_ROOT = Path(UNIICL_CODE_ROOT)

IMAGE_DIRS = [
    "images/AIGI-Holmes",
    "images/AVA",
    "images/World-Aware Planning",
    "images/LAION-HR",
    "images/T2I",
    "images/I2I",
    "images/degraded",
    "images/Concept",
    "images/Chain-of-Editing",
]

CONVERTED_OUTPUTS = [
    "Forgery-Detection/forgery_detection_uniicl.jsonl",
    "Aesthetic-Assessment/aesthetic_assessment_uniicl.jsonl",
    "Attribute-Recognition/attribute_recognition_uniicl.jsonl",
    "Style-Aware-Caption/style_aware_caption_uniicl.jsonl",
    "Fast-Concept-Mapping/fast_concept_mapping_uniicl.jsonl",
    "Visual-Grounding/visual_grounding_uniicl.jsonl",
    "World-Aware-Planning/world_aware_planning_uniicl.jsonl",
    "Scene-Reasoning/scene_reasoning_uniicl.jsonl",
    "Analogical-Inference/analogical_inference_uniicl.jsonl",
    "Instructional-Generation/parquet_instructional_generation/parquet_info.json",
    "Image-Manipulation/parquet_image_manipulation/parquet_info.json",
    "Fast-Concept-Generation/parquet_fast_concept_generation/parquet_info.json",
    "Visual-Refinement/parquet_visual_refinement/parquet_info.json",
    "Analogical-Editing/parquet_analogical_editing/parquet_info.json",
]

CONFIG_REQUIREMENTS = {
    "training": {
        "UNIICL_BASE_MODEL": UNIICL_BASE_MODEL,
    },
    "uniicl_eval": {
        "UNIICL_FINETUNED_MODEL": UNIICL_FINETUNED_MODEL,
        "JUDGE_MODEL": JUDGE_MODEL,
    },
    "generation_eval": {
        "HPSV3_CHECKPOINT": HPSV3_CHECKPOINT,
        "QALIGN_MODEL": QALIGN_MODEL,
        "JUDGE_MODEL": JUDGE_MODEL,
    },
}

CONFIG_PATH_KEYS = {
    "UNIICL_BASE_MODEL",
    "UNIICL_FINETUNED_MODEL",
    "HPSV3_CHECKPOINT",
}


def _is_configured(value: str | None) -> bool:
    return bool(value) and not is_placeholder(value)


def _check_path(path: Path) -> bool:
    return path.exists()


def _tcp_reachable(url: str) -> bool:
    if "://" in url:
        url = url.split("://", 1)[1]
    host_port = url.split("/", 1)[0]
    if ":" in host_port:
        host, port_str = host_port.rsplit(":", 1)
        try:
            port = int(port_str)
        except ValueError:
            return False
    else:
        host, port = host_port, 80
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def _print_check(ok: bool, label: str, detail: str = "") -> None:
    prefix = "[OK]" if ok else "[MISSING]"
    line = f"{prefix} {label}"
    if detail:
        line += f": {detail}"
    print(line)


def main() -> int:
    print("UniICL Open Source Setup Check")
    print(f"Root: {ROOT}")
    print(f"UniICL-760K root: {DATASETS_ROOT}")
    print("")

    print("Core layout")
    core_paths = [
        ("UniICL-Bench", BENCHMARK_ROOT),
        ("UniICL-760K", DATASETS_ROOT),
        ("UniICL", UNIICL_ROOT),
    ]
    core_ok = True
    for label, path in core_paths:
        ok = _check_path(path)
        core_ok &= ok
        _print_check(ok, label, str(path))
    print("")

    print("Image directories")
    image_ok = True
    for rel_path in IMAGE_DIRS:
        path = DATASETS_ROOT / rel_path
        ok = _check_path(path)
        image_ok &= ok
        _print_check(ok, rel_path, str(path))
    print("")

    print("Converted training artifacts")
    converted_ok = True
    for rel_path in CONVERTED_OUTPUTS:
        path = DATASETS_ROOT / rel_path
        ok = _check_path(path)
        converted_ok &= ok
        _print_check(ok, rel_path, str(path))
    if not converted_ok:
        print("Hint: run `cd UniICL-760K && bash convert_unified.sh` first.")
    print("")

    print("Model / scorer config")
    env_ok = {}
    for section, config_items in CONFIG_REQUIREMENTS.items():
        section_ok = True
        print(f"- {section}")
        for name, raw_value in config_items.items():
            value = str(raw_value)
            ok = _is_configured(value)
            if ok and name in CONFIG_PATH_KEYS:
                ok = Path(value).exists()
            section_ok &= ok
            _print_check(ok, name, value or "<unset>")
        env_ok[section] = section_ok
    print("")

    judge_api = str(JUDGE_API_BASE)
    judge_api_ok = _tcp_reachable(judge_api) if _is_configured(judge_api) else False
    print("Judge service")
    _print_check(judge_api_ok, "JUDGE_API_BASE", judge_api)
    if not judge_api_ok:
        print("Hint: start a vLLM/OpenAI-compatible judge server or edit JUDGE_API_BASE in local_paths.py.")
    print("")

    training_ready = core_ok and image_ok and converted_ok and env_ok["training"]
    understanding_eval_ready = (
        core_ok and image_ok and env_ok["uniicl_eval"] and judge_api_ok
    )
    generation_eval_ready = (
        core_ok and image_ok and env_ok["uniicl_eval"] and env_ok["generation_eval"] and judge_api_ok
    )

    print("Readiness summary")
    _print_check(training_ready, "Training")
    _print_check(understanding_eval_ready, "UniICL-Bench understanding evaluation")
    _print_check(generation_eval_ready, "UniICL-Bench generation evaluation")

    if training_ready and understanding_eval_ready and generation_eval_ready:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
