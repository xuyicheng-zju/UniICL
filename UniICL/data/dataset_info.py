# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

from .interleave_datasets import UnifiedEditIterableDataset, ICLGenIterableDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset
from .vlm_dataset_gen import SftJSONLIterableDatasetWithGen


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OPEN_SOURCE_ROOT = PACKAGE_ROOT.parent
if str(OPEN_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(OPEN_SOURCE_ROOT))

from local_paths import UNIICL_760K_ROOT  # noqa: E402


DATASETS_ROOT = Path(UNIICL_760K_ROOT)


def _dataset_path(rel_path: str) -> str:
    return str(DATASETS_ROOT / rel_path)


DATASET_REGISTRY = {
    "t2i_pretrain": T2IIterableDataset,
    "vlm_sft": SftJSONLIterableDataset,
    "vlm_sft_icl": SftJSONLIterableDatasetWithGen,
    "unified_edit": UnifiedEditIterableDataset,
    "icl_gen": ICLGenIterableDataset,
}


DATASET_INFO = {
    "icl_gen": {
        "instructional_generation_icl": {
            "data_dir": _dataset_path("Instructional-Generation/parquet_instructional_generation"),
            "num_total_samples": 60990,
            "parquet_info_path": _dataset_path("Instructional-Generation/parquet_instructional_generation/parquet_info.json"),
        },
        "image_manipulation_icl": {
            "data_dir": _dataset_path("Image-Manipulation/parquet_image_manipulation"),
            "num_total_samples": 39201,
            "parquet_info_path": _dataset_path("Image-Manipulation/parquet_image_manipulation/parquet_info.json"),
        },
        "fast_concept_generation": {
            "data_dir": _dataset_path("Fast-Concept-Generation/parquet_fast_concept_generation"),
            "num_total_samples": 50000,
            "parquet_info_path": _dataset_path("Fast-Concept-Generation/parquet_fast_concept_generation/parquet_info.json"),
        },
        "analogical_editing": {
            "data_dir": _dataset_path("Analogical-Editing/parquet_analogical_editing"),
            "num_total_samples": 18710,
            "parquet_info_path": _dataset_path("Analogical-Editing/parquet_analogical_editing/parquet_info.json"),
        },
        "visual_refinement": {
            "data_dir": _dataset_path("Visual-Refinement/parquet_visual_refinement"),
            "num_total_samples": 27996,
            "parquet_info_path": _dataset_path("Visual-Refinement/parquet_visual_refinement/parquet_info.json"),
        },
    },
    "vlm_sft": {
        "style_aware_caption_icl": {
            "data_dir": str(DATASETS_ROOT),
            "jsonl_path": _dataset_path("Style-Aware-Caption/style_aware_caption_uniicl.jsonl"),
            "num_total_samples": 67225,
        },
        "scene_reasoning_icl": {
            "data_dir": str(DATASETS_ROOT),
            "jsonl_path": _dataset_path("Scene-Reasoning/scene_reasoning_uniicl.jsonl"),
            "num_total_samples": 66074,
        },
        "visual_grounding_icl": {
            "data_dir": str(DATASETS_ROOT),
            "jsonl_path": _dataset_path("Visual-Grounding/visual_grounding_uniicl.jsonl"),
            "num_total_samples": 66347,
        },
        "attribute_recognition_icl": {
            "data_dir": str(DATASETS_ROOT),
            "jsonl_path": _dataset_path("Attribute-Recognition/attribute_recognition_uniicl.jsonl"),
            "num_total_samples": 64338,
        },
        "forgery_detection_icl": {
            "data_dir": str(DATASETS_ROOT),
            "jsonl_path": _dataset_path("Forgery-Detection/forgery_detection_uniicl.jsonl"),
            "num_total_samples": 40661,
        },
        "aesthetic_assessment_icl": {
            "data_dir": str(DATASETS_ROOT),
            "jsonl_path": _dataset_path("Aesthetic-Assessment/aesthetic_assessment_uniicl.jsonl"),
            "num_total_samples": 80481,
        },
        "fast_concept_mapping_icl": {
            "data_dir": str(DATASETS_ROOT),
            "jsonl_path": _dataset_path("Fast-Concept-Mapping/fast_concept_mapping_uniicl.jsonl"),
            "num_total_samples": 50000,
        },
        "world_aware_planning_icl": {
            "data_dir": str(DATASETS_ROOT),
            "jsonl_path": _dataset_path("World-Aware-Planning/world_aware_planning_uniicl.jsonl"),
            "num_total_samples": 63964,
        },
        "analogical_inference_icl": {
            "data_dir": str(DATASETS_ROOT),
            "jsonl_path": _dataset_path("Analogical-Inference/analogical_inference_uniicl.jsonl"),
            "num_total_samples": 51028,
        },
    },
}

# Backward-compatible aliases for older configs.
DATASET_INFO["icl_gen"]["t2i_icl"] = DATASET_INFO["icl_gen"]["instructional_generation_icl"]
DATASET_INFO["icl_gen"]["i2i_icl"] = DATASET_INFO["icl_gen"]["image_manipulation_icl"]
DATASET_INFO["icl_gen"]["fci"] = DATASET_INFO["icl_gen"]["fast_concept_generation"]
DATASET_INFO["icl_gen"]["visualcloze-g"] = DATASET_INFO["icl_gen"]["analogical_editing"]
DATASET_INFO["icl_gen"]["perfection"] = DATASET_INFO["icl_gen"]["visual_refinement"]
DATASET_INFO["vlm_sft"]["caption_icl"] = DATASET_INFO["vlm_sft"]["style_aware_caption_icl"]
DATASET_INFO["vlm_sft"]["stylized_caption_icl"] = DATASET_INFO["vlm_sft"]["style_aware_caption_icl"]
DATASET_INFO["vlm_sft"]["vqa_icl"] = DATASET_INFO["vlm_sft"]["scene_reasoning_icl"]
DATASET_INFO["vlm_sft"]["grounding_icl"] = DATASET_INFO["vlm_sft"]["visual_grounding_icl"]
DATASET_INFO["vlm_sft"]["attr_rec_icl"] = DATASET_INFO["vlm_sft"]["attribute_recognition_icl"]
DATASET_INFO["vlm_sft"]["aigi_icl"] = DATASET_INFO["vlm_sft"]["forgery_detection_icl"]
DATASET_INFO["vlm_sft"]["ava_icl"] = DATASET_INFO["vlm_sft"]["aesthetic_assessment_icl"]
DATASET_INFO["vlm_sft"]["fcb"] = DATASET_INFO["vlm_sft"]["fast_concept_mapping_icl"]
DATASET_INFO["vlm_sft"]["planning"] = DATASET_INFO["vlm_sft"]["world_aware_planning_icl"]
DATASET_INFO["vlm_sft"]["visualcloze-u"] = DATASET_INFO["vlm_sft"]["analogical_inference_icl"]

# Alias for ICL-formatted JSONL datasets (same paths, different parser).
DATASET_INFO["vlm_sft_icl"] = DATASET_INFO["vlm_sft"]
