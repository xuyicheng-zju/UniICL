#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${BENCH_DIR}/.." && pwd)"

cd "${BENCH_DIR}"

if [[ ! -e "${ROOT_DIR}/UniICL-760K/images" ]]; then
  if [[ -d "${ROOT_DIR}/images" ]]; then
    ln -s ../images "${ROOT_DIR}/UniICL-760K/images"
    echo "Created image symlink: UniICL-760K/images -> ../images"
  else
    echo "Missing image directory: ${ROOT_DIR}/UniICL-760K/images" >&2
    echo "Expected either UniICL-760K/images or top-level images/." >&2
    exit 1
  fi
fi

"${PYTHON_BIN}" - <<'PY'
import socket
import sys

sock = socket.socket()
sock.settimeout(1.0)
try:
    sock.connect(("127.0.0.1", 8000))
except OSError:
    print(
        "Warning: judge/VLM API is not listening on 127.0.0.1:8000. "
        "GPU 7 is reserved for that service; start it before tasks that need judge scoring.",
        file=sys.stderr,
    )
finally:
    sock.close()
PY

if [[ $# -gt 0 ]]; then
  TAU_SPECS=("$@")
elif [[ -n "${TAU_VALUES:-}" ]]; then
  read -r -a TAU_SPECS <<< "${TAU_VALUES}"
else
  TAU_SPECS=(0.1 0.4 0.7)
fi

declare -A GPU_TASKS=(
  [0]="visual_grounding attribute_recognition"
  [1]="scene_reasoning style_aware_caption"
  [2]="instructional_generation image_manipulation"
  [3]="fast_concept_mapping fast_concept_generation"
  [5]="analogical_inference analogical_editing"
  [6]="aesthetic_assessment forgery_detection visual_refinement"
)

for K_SHOT in 1 2 4 8; do
  for TAU_SPEC in "${TAU_SPECS[@]}"; do
    if [[ "${TAU_SPEC}" == "adaptive" ]]; then
      LABEL="adaptive"
      FIXED_TAU_ARGS=()
    else
      LABEL="tau_${TAU_SPEC//./p}"
      FIXED_TAU_ARGS=(--capm-fixed-tau "${TAU_SPEC}")
    fi

    OUTPUT_DIR="${BENCH_DIR}/eval_results_tau_ablation_${K_SHOT}shot/${LABEL}"
    LOG_DIR="${BENCH_DIR}/logs_tau_ablation_${K_SHOT}shot/${LABEL}"

    echo "Running UniICL tau ablation"
    echo "  Tau mode: ${LABEL}"
    echo "  K-shot: ${K_SHOT}"
    echo "  Evaluation GPUs: 0-6"
    echo "  Reserved GPU: 7 for judge/VLM API"
    echo "  Skipped 0-shot-only tasks: world_aware_planning, chain_of_editing"
    echo "  Python: ${PYTHON_BIN}"
    echo "  Output: ${OUTPUT_DIR}"
    echo "  Logs: ${LOG_DIR}"
    echo

    mkdir -p "${LOG_DIR}/uniicl"
    PIDS=()

    for GPU in 0 1 2 3 5 6; do
      TASKS="${GPU_TASKS[$GPU]}"
      LOG_FILE="${LOG_DIR}/uniicl/gpu${GPU}_queue.log"
      TASK_ARRAY=()
      read -r -a TASK_ARRAY <<< "${TASKS}"

      env HIP_VISIBLE_DEVICES="${GPU}" CUDA_VISIBLE_DEVICES="${GPU}" \
        "${PYTHON_BIN}" run_eval.py \
        --model uniicl \
        --task "${TASK_ARRAY[@]}" \
        --gpu "${GPU}" \
        --k-shot "${K_SHOT}" \
        --benchmark-dir "${BENCH_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --log-dir "${LOG_DIR}" \
        "${FIXED_TAU_ARGS[@]}" >"${LOG_FILE}" 2>&1 &
      PIDS+=($!)
    done

    FAIL=0
    for PID in "${PIDS[@]}"; do
      if ! wait "${PID}"; then
        FAIL=1
      fi
    done

    if [[ "${FAIL}" -ne 0 ]]; then
      echo "Tau mode ${LABEL} for ${K_SHOT}-shot finished with failures." >&2
      exit 1
    fi

    echo
  done
done
