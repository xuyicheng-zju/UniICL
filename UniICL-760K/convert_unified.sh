#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS_ROOT="${SCRIPT_DIR}"
NUM_WORKERS="${UNICL_CONVERT_NUM_WORKERS:-16}"
ROWS_PER_FILE="${UNICL_CONVERT_ROWS_PER_FILE:-500}"
SELECTED_TASKS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks)
            SELECTED_TASKS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--tasks task1,task2,...]"
            exit 1
            ;;
    esac
done

SHOTS_STANDARD="0,1,2,4"
WEIGHTS_STANDARD="10,25,40,25"
SHOTS_INTENT="2,3,4"
WEIGHTS_INTENT="40,20,40"

UND_TASKS=(
    "style_aware_caption|Style-Aware-Caption/style_aware_caption_train_icl.jsonl|Style-Aware-Caption/style_aware_caption_uniicl.jsonl|UND|default"
    "scene_reasoning|Scene-Reasoning/scene_reasoning_train_icl.jsonl|Scene-Reasoning/scene_reasoning_uniicl.jsonl|UND|default"
    "visual_grounding|Visual-Grounding/visual_grounding_train_icl.jsonl|Visual-Grounding/visual_grounding_uniicl.jsonl|UND|default"
    "attribute_recognition|Attribute-Recognition/attribute_recognition_train_icl.jsonl|Attribute-Recognition/attribute_recognition_uniicl.jsonl|UND|default"
    "forgery_detection|Forgery-Detection/forgery_detection_train_icl.jsonl|Forgery-Detection/forgery_detection_uniicl.jsonl|UND|default"
    "aesthetic_assessment|Aesthetic-Assessment/aesthetic_assessment_train_icl.jsonl|Aesthetic-Assessment/aesthetic_assessment_uniicl.jsonl|UND|default"
    "fast_concept_mapping|Fast-Concept-Mapping/fast_concept_mapping_train_icl.jsonl|Fast-Concept-Mapping/fast_concept_mapping_uniicl.jsonl|UND|intent"
    "analogical_inference|Analogical-Inference/analogical_inference_train_icl.jsonl|Analogical-Inference/analogical_inference_uniicl.jsonl|UND|intent"
    "world_aware_planning|World-Aware-Planning/world_aware_planning_train_icl.json|World-Aware-Planning/world_aware_planning_uniicl.jsonl|PLANNING|default"
)

GEN_TASKS=(
    "instructional_generation|Instructional-Generation/instructional_generation_train_icl.jsonl|Instructional-Generation/parquet_instructional_generation|instructional_generation|default"
    "image_manipulation|Image-Manipulation/image_manipulation_train_icl.jsonl|Image-Manipulation/parquet_image_manipulation|image_manipulation|default"
    "visual_refinement|Visual-Refinement/visual_refinement_train_icl.jsonl|Visual-Refinement/parquet_visual_refinement|visual_refinement|default"
    "analogical_editing|Analogical-Editing/analogical_editing_train_icl.json|Analogical-Editing/parquet_analogical_editing|analogical_editing|intent"
    "fast_concept_generation|Fast-Concept-Generation/fast_concept_generation_train_icl.jsonl|Fast-Concept-Generation/parquet_fast_concept_generation|fast_concept_generation|intent"
)

should_process_task() {
    local task_name="$1"
    if [[ -z "$SELECTED_TASKS" ]]; then
        return 0
    fi
    [[ ",$SELECTED_TASKS," == *",$task_name,"* ]]
}

shots_weights_for() {
    local config="$1"
    case "$config" in
        default) echo "${SHOTS_STANDARD}|${WEIGHTS_STANDARD}" ;;
        intent) echo "${SHOTS_INTENT}|${WEIGHTS_INTENT}" ;;
        *) echo "${SHOTS_STANDARD}|${WEIGHTS_STANDARD}" ;;
    esac
}

echo "========================================"
echo "UniICL public dataset conversion"
echo "========================================"
echo "UniICL-760K root: ${DATASETS_ROOT}"
if [[ -n "$SELECTED_TASKS" ]]; then
    echo "Selected tasks: ${SELECTED_TASKS}"
fi
echo

echo "[1/2] Converting understanding-style tasks to UniICL JSONL"
for entry in "${UND_TASKS[@]}"; do
    IFS='|' read -r task_name input_rel output_rel task_type shot_config <<< "$entry"
    if ! should_process_task "$task_name"; then
        continue
    fi

    sw="$(shots_weights_for "$shot_config")"
    shots="${sw%%|*}"
    weights="${sw#*|}"
    input_path="${SCRIPT_DIR}/${input_rel}"
    output_path="${SCRIPT_DIR}/${output_rel}"

    echo "  - ${task_name}"
    python3 "${SCRIPT_DIR}/convert_icl_to_uniicl.py" \
        --input_jsonl "${input_path}" \
        --output_jsonl "${output_path}" \
        --task_type "${task_type}" \
        --image_base_path "${DATASETS_ROOT}" \
        --shots "${shots}" \
        --weights "${weights}"
done

echo
echo "[2/2] Converting generation-style tasks to Parquet"
for entry in "${GEN_TASKS[@]}"; do
    IFS='|' read -r task_name input_rel output_rel gen_type shot_config <<< "$entry"
    if ! should_process_task "$task_name"; then
        continue
    fi

    sw="$(shots_weights_for "$shot_config")"
    shots="${sw%%|*}"
    weights="${sw#*|}"
    input_path="${SCRIPT_DIR}/${input_rel}"
    output_path="${SCRIPT_DIR}/${output_rel}"

    echo "  - ${task_name}"
    python3 "${SCRIPT_DIR}/convert_icl_to_parquet.py" \
        --input "${input_path}" \
        --output "${output_path}" \
        --type "${gen_type}" \
        --image_base_path "${DATASETS_ROOT}" \
        --rows_per_file "${ROWS_PER_FILE}" \
        --num_workers "${NUM_WORKERS}" \
        --shots "${shots}" \
        --weights "${weights}"
done

echo
echo "Done."
