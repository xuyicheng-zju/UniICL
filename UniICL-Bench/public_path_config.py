#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


BENCHMARK_ROOT = Path(__file__).resolve().parent
THIRD_PARTY_CONFIG_ROOT = BENCHMARK_ROOT / "third_party_configs"
OPEN_SOURCE_ROOT = BENCHMARK_ROOT.parent
if str(OPEN_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(OPEN_SOURCE_ROOT))

from local_paths import (  # noqa: E402
    DINO_MODEL,
    FLUX_MODEL,
    HPSV3_CHECKPOINT,
    HPSV3_CONFIG,
    HPSV3_VENDOR_ROOT,
    INTERNVL_MODEL,
    JUDGE_API_BASE,
    JUDGE_MODEL,
    NEXUSGEN_MODEL,
    OVISU1_MODEL,
    QALIGN_MODEL,
    QWEN25VL_MODEL,
    QWEN3VL_MODEL,
    SIGLIP_MODEL,
    UNIICL_760K_ROOT,
    UNIICL_BASE_MODEL,
    UNIICL_FINETUNED_MODEL,
    UNIWORLD_MODEL,
)


DATASETS_ROOT = str(UNIICL_760K_ROOT)


def _under_datasets(subdir: str) -> str:
    return str(Path(DATASETS_ROOT) / subdir)


# UniICL-Bench assumes a shared UniICL-760K root:
#   UniICL-760K/
#     images/
#       AIGI-Holmes/
#       AVA/
#       World-Aware Planning/
#       LAION-HR/
#       T2I/
#       I2I/
#       degraded/
#       Concept/
#       Chain-of-Editing/
#
# All released benchmark and training annotations now store image paths relative
# to UniICL-760K/, e.g. images/LAION-HR/000123.jpg or
# images/T2I/sample_000001.png. Evaluators should therefore join paths against
# the UniICL-760K root itself.
DATASETS_IMAGES_ROOT = _under_datasets("images")
GENERATION_ROOT = DATASETS_IMAGES_ROOT
UNICL_GEN_ROOT = GENERATION_ROOT


IMAGE_ROOT_MARKERS = {
    "datasets_root": DATASETS_ROOT,
    "datasets_images_root": DATASETS_IMAGES_ROOT,
    "generation_root": GENERATION_ROOT,
}


CANONICAL_TASK_ORDER = [
    "visual_grounding",
    "attribute_recognition",
    "scene_reasoning",
    "style_aware_caption",
    "instructional_generation",
    "image_manipulation",
    "aesthetic_assessment",
    "forgery_detection",
    "visual_refinement",
    "fast_concept_mapping",
    "fast_concept_generation",
    "world_aware_planning",
    "chain_of_editing",
    "analogical_editing",
    "analogical_inference",
]

LEGACY_TASK_ALIASES = {
    "grounding": "visual_grounding",
    "attr_rec": "attribute_recognition",
    "vqa": "scene_reasoning",
    "caption": "style_aware_caption",
    "t2i": "instructional_generation",
    "i2i": "image_manipulation",
    "aesthetic": "aesthetic_assessment",
    "authenticity": "forgery_detection",
    "perfection": "visual_refinement",
    "fcb": "fast_concept_mapping",
    "fci": "fast_concept_generation",
    "planning": "world_aware_planning",
    "chain_edit": "chain_of_editing",
    "visualcloze_g": "analogical_editing",
    "visualcloze-g": "analogical_editing",
    "visualcloze_u": "analogical_inference",
    "visualcloze-u": "analogical_inference",
    "visualcloze": "analogical_editing",
}

TASK_CLI_CHOICES = CANONICAL_TASK_ORDER + ["all"]


def normalize_task_name(task: str, allow_all: bool = False) -> str:
    normalized = str(task).strip().lower().replace("-", "_")
    normalized = LEGACY_TASK_ALIASES.get(normalized, normalized)
    if allow_all and normalized == "all":
        return normalized
    if normalized not in CANONICAL_TASK_ORDER:
        raise ValueError(
            f"Unknown task: {task}. Expected one of: {', '.join(CANONICAL_TASK_ORDER)}"
        )
    return normalized


TASK_DATA_REL_PATHS = {
    "visual_grounding": "Visual-Grounding/visual_grounding_benchmark.jsonl",
    "attribute_recognition": "Attribute-Recognition/attribute_recognition_benchmark.jsonl",
    "scene_reasoning": "Scene-Reasoning/scene_reasoning_benchmark.jsonl",
    "style_aware_caption": "Style-Aware-Caption/style_aware_caption_benchmark.jsonl",
    "instructional_generation": "Instructional-Generation/instructional_generation_benchmark.jsonl",
    "image_manipulation": "Image-Manipulation/image_manipulation_benchmark.jsonl",
    "aesthetic_assessment": "Aesthetic-Assessment/aesthetic_assessment_benchmark.jsonl",
    "forgery_detection": "Forgery-Detection/forgery_detection_benchmark.jsonl",
    "visual_refinement": "Visual-Refinement/visual_refinement_benchmark.jsonl",
    "fast_concept_mapping": "Fast-Concept-Mapping/fast_concept_mapping_benchmark.json",
    "fast_concept_generation": "Fast-Concept-Generation/fast_concept_generation_benchmark.json",
    "world_aware_planning": "World-Aware-Planning/world_aware_planning_benchmark.json",
    "chain_of_editing": "Chain-of-Editing/chain_of_editing_benchmark.json",
    "analogical_editing": "Analogical-Editing/analogical_editing_benchmark.json",
    "analogical_inference": "Analogical-Inference/analogical_inference_benchmark.jsonl",
}


TASK_IMAGE_ROOT_KEYS = {
    "visual_grounding": "datasets_root",
    "attribute_recognition": "datasets_root",
    "scene_reasoning": "datasets_root",
    "style_aware_caption": "datasets_root",
    "analogical_inference": "datasets_root",
    "instructional_generation": "datasets_root",
    "image_manipulation": "datasets_root",
    "visual_refinement": "datasets_root",
    "analogical_editing": "datasets_root",
    "aesthetic_assessment": "datasets_root",
    "forgery_detection": "datasets_root",
    "world_aware_planning": "datasets_root",
    "fast_concept_mapping": "datasets_root",
    "fast_concept_generation": "datasets_root",
    "chain_of_editing": "datasets_root",
}


TASK_IMAGE_ROOT_NOTES = {
    "visual_grounding": "Reads root-relative paths like images/LAION-HR/... under UniICL-760K/.",
    "attribute_recognition": "Reads root-relative paths like images/LAION-HR/... under UniICL-760K/.",
    "scene_reasoning": "Reads root-relative paths like images/LAION-HR/... under UniICL-760K/.",
    "style_aware_caption": "Reads root-relative paths like images/LAION-HR/... under UniICL-760K/.",
    "analogical_inference": "Reads root-relative paths like images/LAION-HR/... under UniICL-760K/.",
    "instructional_generation": "Reads root-relative paths like images/T2I/... under UniICL-760K/.",
    "image_manipulation": "Reads source paths under images/T2I/... and target paths under images/I2I/... from UniICL-760K/.",
    "visual_refinement": "Reads images/degraded/... and images/T2I/... under UniICL-760K/.",
    "analogical_editing": "Reads paired paths under images/T2I/... and images/I2I/... from UniICL-760K/.",
    "aesthetic_assessment": "Reads root-relative paths like images/AVA/... under UniICL-760K/.",
    "forgery_detection": "Reads root-relative paths like images/AIGI-Holmes/... under UniICL-760K/.",
    "world_aware_planning": "Reads root-relative environment frames like images/World-Aware Planning/... under UniICL-760K/.",
    "fast_concept_mapping": "Reads root-relative concept images like images/Concept/... under UniICL-760K/.",
    "fast_concept_generation": "Reads root-relative concept images like images/Concept/... under UniICL-760K/.",
    "chain_of_editing": "Reads root-relative reference images like images/Chain-of-Editing/... under UniICL-760K/.",
}


def get_task_image_dir(task: str, benchmark_root: str | Path | None = None) -> str:
    del benchmark_root
    normalized = normalize_task_name(task)
    root_key = TASK_IMAGE_ROOT_KEYS[normalized]
    return IMAGE_ROOT_MARKERS[root_key]


def get_task_data_path(task: str, benchmark_root: str | Path | None = None) -> str:
    normalized = normalize_task_name(task)
    root = Path(benchmark_root) if benchmark_root is not None else BENCHMARK_ROOT
    return str(root / TASK_DATA_REL_PATHS[normalized])


# Model / scorer path placeholders for benchmark evaluation code. These are
# intentionally environment-agnostic and should be overridden by CLI args or
# environment variables.
DEFAULT_UNIICL_BASE_MODEL = str(UNIICL_BASE_MODEL)
DEFAULT_UNIICL_FINETUNED_MODEL = str(UNIICL_FINETUNED_MODEL)
DEFAULT_BAGEL_BASE_MODEL = DEFAULT_UNIICL_BASE_MODEL
DEFAULT_BAGEL_FINETUNED_MODEL = DEFAULT_UNIICL_FINETUNED_MODEL
DEFAULT_UNIWORLD_MODEL = str(UNIWORLD_MODEL)
DEFAULT_QWEN3VL_MODEL = str(QWEN3VL_MODEL)
DEFAULT_QWEN25VL_MODEL = str(QWEN25VL_MODEL)
DEFAULT_INTERNVL_MODEL = str(INTERNVL_MODEL)
DEFAULT_NEXUSGEN_MODEL = str(NEXUSGEN_MODEL)
DEFAULT_OVISU1_MODEL = str(OVISU1_MODEL)
DEFAULT_FLUX_MODEL = str(FLUX_MODEL)
DEFAULT_SIGLIP_MODEL = str(SIGLIP_MODEL)
DEFAULT_HPSV3_VENDOR_ROOT = str(HPSV3_VENDOR_ROOT)
DEFAULT_HPSV3_CONFIG = str(HPSV3_CONFIG)
DEFAULT_HPSV3_CHECKPOINT = str(HPSV3_CHECKPOINT)
DEFAULT_QALIGN_MODEL = str(QALIGN_MODEL)
DEFAULT_JUDGE_MODEL = str(JUDGE_MODEL)
DEFAULT_JUDGE_API_BASE = str(JUDGE_API_BASE)
DEFAULT_DINO_MODEL = str(DINO_MODEL)
