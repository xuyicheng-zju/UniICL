#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-5000}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${BENCH_DIR}/taxonomy_capm_train_understanding_${SAMPLES_PER_TASK}}"
SHARD_ROOT="${OUTPUT_DIR}/shards"
LOG_DIR="${OUTPUT_DIR}/logs"

declare -A GPU_TASKS=(
  [0]="visual_grounding"
  [1]="style_aware_caption"
  [2]="fast_concept_mapping"
  [3]="world_aware_planning"
  [4]="analogical_inference"
  [5]="aesthetic_assessment"
)

EXTRA_ARGS=("$@")

echo "Running CAPM-only taxonomy extraction on UniICL-760K understanding training tasks"
echo "  Python: ${PYTHON_BIN}"
echo "  Samples per task: ${SAMPLES_PER_TASK}"
echo "  Sample seed: ${SAMPLE_SEED}"
echo "  Output: ${OUTPUT_DIR}"
echo

mkdir -p "${SHARD_ROOT}" "${LOG_DIR}"
cd "${BENCH_DIR}"

PIDS=()
SHARD_ARGS=()

for GPU in 0 1 2 3 4 5; do
  TASK="${GPU_TASKS[$GPU]}"
  SHARD_DIR="${SHARD_ROOT}/${TASK}"
  LOG_FILE="${LOG_DIR}/${TASK}.log"
  SHARD_ARGS+=(--shard-dir "${SHARD_DIR}")

  env HIP_VISIBLE_DEVICES="${GPU}" CUDA_VISIBLE_DEVICES="${GPU}" \
    "${PYTHON_BIN}" "${BENCH_DIR}/analyze_taxonomy_capm_train_understanding.py" \
    --task "${TASK}" \
    --samples-per-task "${SAMPLES_PER_TASK}" \
    --sample-seed "${SAMPLE_SEED}" \
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
  echo "Training-set CAPM-only extraction finished with failures." >&2
  exit 1
fi

"${PYTHON_BIN}" "${BENCH_DIR}/merge_taxonomy_capm_shards.py" \
  "${SHARD_ARGS[@]}" \
  --output-dir "${OUTPUT_DIR}"

echo
echo "Merged analysis ready at: ${OUTPUT_DIR}"
