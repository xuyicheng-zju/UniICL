#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
K_SHOT="${K_SHOT:-2}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${BENCH_DIR}/taxonomy_capm_analysis_8gpu_${K_SHOT}shot}"
SHARD_ROOT="${OUTPUT_DIR}/shards"
LOG_DIR="${OUTPUT_DIR}/logs"

EXTRA_ARGS=("$@")

declare -A GPU_TASKS=(
  [0]="visual_grounding attribute_recognition"
  [1]="scene_reasoning style_aware_caption"
  [2]="instructional_generation image_manipulation"
  [3]="fast_concept_mapping fast_concept_generation"
  [4]="world_aware_planning chain_of_editing"
  [5]="analogical_inference analogical_editing"
  [6]="aesthetic_assessment forgery_detection"
  [7]="visual_refinement"
)

echo "Running taxonomy CAPM-only extraction"
echo "  Python: ${PYTHON_BIN}"
echo "  K-shot: ${K_SHOT}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Logs: ${LOG_DIR}"
echo

mkdir -p "${SHARD_ROOT}" "${LOG_DIR}"
cd "${BENCH_DIR}"

PIDS=()
SHARD_ARGS=()

for GPU in 0 1 2 3 4 5 6 7; do
  TASKS="${GPU_TASKS[$GPU]}"
  SHARD_DIR="${SHARD_ROOT}/gpu${GPU}"
  LOG_FILE="${LOG_DIR}/gpu${GPU}.log"
  TASK_ARRAY=()
  read -r -a TASK_ARRAY <<< "${TASKS}"

  SHARD_ARGS+=(--shard-dir "${SHARD_DIR}")

  env HIP_VISIBLE_DEVICES="${GPU}" CUDA_VISIBLE_DEVICES="${GPU}" \
    "${PYTHON_BIN}" "${BENCH_DIR}/analyze_taxonomy_capm_only.py" \
    --task "${TASK_ARRAY[@]}" \
    --k-shot "${K_SHOT}" \
    --output-dir "${SHARD_DIR}" \
    --skip-analysis \
    "${EXTRA_ARGS[@]}" >"${LOG_FILE}" 2>&1 &
  PIDS+=($!)
done

FAIL=0
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then
    FAIL=1
  fi
done

if [[ "${FAIL}" -ne 0 ]]; then
  echo "CAPM-only shard extraction finished with failures." >&2
  exit 1
fi

"${PYTHON_BIN}" "${BENCH_DIR}/merge_taxonomy_capm_shards.py" \
  "${SHARD_ARGS[@]}" \
  --output-dir "${OUTPUT_DIR}"

echo
echo "Merged analysis ready at: ${OUTPUT_DIR}"
