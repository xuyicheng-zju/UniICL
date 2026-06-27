#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-${BENCH_DIR}/eval_results_capm_component}"
SUMMARY_PATH="${SUMMARY_PATH:-${RESULTS_ROOT}/rescored_gen_tasks_summary.json}"
TASKS=()
MODES=()

usage() {
  cat <<'EOF'
Usage:
  run_rescore_capm_component_gen_tasks.sh [options] [mode...]

Options:
  --task TASK                instructional_generation | visual_refinement
                             Can be repeated. Default: run both.
  --mode MODE                Restrict rescoring to specific CAPM ablation mode.
                             Can be repeated.
  --gpu-id ID                Visible GPU to use for scoring. Default: 0.
  --results-root PATH        Results root. Default: eval_results_capm_component.
  --summary-path PATH        Summary JSON path.
  -h, --help                 Show this help message.

Notes:
  This wrapper uses the current terminal Python environment only.
  Activate the target environment yourself before running it.
  If HPSv3 and Q-Align require different environments, run the two tasks separately.

Examples:
  run_rescore_capm_component_gen_tasks.sh --task instructional_generation
  run_rescore_capm_component_gen_tasks.sh --task visual_refinement
  run_rescore_capm_component_gen_tasks.sh --task visual_refinement --mode no_adaptive_routing
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --task)
      TASKS+=("${2//-/_}")
      shift 2
      ;;
    --mode)
      MODES+=("$2")
      shift 2
      ;;
    --gpu-id)
      GPU_ID="$2"
      shift 2
      ;;
    --results-root)
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --summary-path)
      SUMMARY_PATH="$2"
      shift 2
      ;;
    *)
      MODES+=("$1")
      shift
      ;;
  esac
done

if [[ ${#TASKS[@]} -eq 0 ]]; then
  TASKS=(instructional_generation visual_refinement)
fi

if [[ ${#TASKS[@]} -gt 1 ]]; then
  echo "This wrapper uses the current terminal Python environment only." >&2
  echo "If HPSv3 and Q-Align need different environments, activate each env and run one task at a time." >&2
  echo "Example:" >&2
  echo "  run_rescore_capm_component_gen_tasks.sh --task instructional_generation" >&2
  echo "  run_rescore_capm_component_gen_tasks.sh --task visual_refinement" >&2
  echo >&2
fi

MODE_ARGS=()
if [[ ${#MODES[@]} -gt 0 ]]; then
  for MODE in "${MODES[@]}"; do
    MODE_ARGS+=(--mode "${MODE}")
  done
fi

run_one() {
  local task="$1"
  local summary_path="$2"

  echo "Rescoring CAPM component generation task"
  echo "  Task: ${task}"
  echo "  Results root: ${RESULTS_ROOT}"
  echo "  GPU: ${GPU_ID}"
  echo "  Python: ${PYTHON_BIN}"
  echo "  Summary: ${summary_path}"
  if [[ ${#MODE_ARGS[@]} -gt 0 ]]; then
    echo "  Restricted modes: ${MODES[*]}"
  else
    echo "  Restricted modes: auto-discover all"
  fi
  echo

  cd "${BENCH_DIR}"
  env HIP_VISIBLE_DEVICES="${GPU_ID}" CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${PYTHON_BIN}" "${BENCH_DIR}/rescore_capm_component_gen_tasks.py" \
    --results-root "${RESULTS_ROOT}" \
    --device "cuda:0" \
    --summary-path "${summary_path}" \
    --task "${task}" \
    "${MODE_ARGS[@]}"
}

if [[ ${#TASKS[@]} -eq 2 ]] && [[ "${TASKS[*]}" == "instructional_generation visual_refinement" ]]; then
  echo "Rescoring CAPM component generation tasks"
  echo "  Tasks: ${TASKS[*]}"
  echo "  Results root: ${RESULTS_ROOT}"
  echo "  GPU: ${GPU_ID}"
  echo "  Python: ${PYTHON_BIN}"
  echo "  Summary: ${SUMMARY_PATH}"
  if [[ ${#MODE_ARGS[@]} -gt 0 ]]; then
    echo "  Restricted modes: ${MODES[*]}"
  else
    echo "  Restricted modes: auto-discover all"
  fi
  echo

  cd "${BENCH_DIR}"
  env HIP_VISIBLE_DEVICES="${GPU_ID}" CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${PYTHON_BIN}" "${BENCH_DIR}/rescore_capm_component_gen_tasks.py" \
    --results-root "${RESULTS_ROOT}" \
    --device "cuda:0" \
    --summary-path "${SUMMARY_PATH}" \
    --task instructional_generation \
    --task visual_refinement \
    "${MODE_ARGS[@]}"
  exit 0
fi

for TASK in "${TASKS[@]}"; do
  case "${TASK}" in
    instructional_generation)
      TASK_SUMMARY="${SUMMARY_PATH%.json}_instructional_generation.json"
      run_one "${TASK}" "${TASK_SUMMARY}"
      ;;
    visual_refinement)
      TASK_SUMMARY="${SUMMARY_PATH%.json}_visual_refinement.json"
      run_one "${TASK}" "${TASK_SUMMARY}"
      ;;
    *)
      echo "Unknown task: ${TASK}" >&2
      exit 1
      ;;
  esac
done
