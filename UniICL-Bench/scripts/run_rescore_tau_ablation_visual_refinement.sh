#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_ROOT=""
SUMMARY_PATH="${SUMMARY_PATH:-${BENCH_DIR}/rescore_tau_ablation_visual_refinement_summary.json}"
SHOTS=()
MODES=()

usage() {
  cat <<'EOF'
Usage:
  run_rescore_tau_ablation_visual_refinement.sh [options]

Options:
  --shot K                  Restrict rescoring to a specific shot. Can be repeated.
                            Default: 1 2 4 8
  --mode MODE               Restrict rescoring to a specific tau mode. Can be repeated.
                            Default: auto-discover all modes under each shot root.
  --results-root PATH       Explicit tau ablation shot root, e.g.
                            eval_results_tau_ablation_1shot
                            If set, --shot is ignored.
  --gpu-id ID               Visible GPU to use for Q-Align scoring. Default: 0.
  --summary-path PATH       Output JSON summary path.
  -h, --help                Show this help message.

Notes:
  Activate the Q-Align-compatible environment yourself before running.
  This wrapper only rescoring visual_refinement for tau ablation outputs.

Examples:
  run_rescore_tau_ablation_visual_refinement.sh
  run_rescore_tau_ablation_visual_refinement.sh --shot 1 --shot 2
  run_rescore_tau_ablation_visual_refinement.sh --mode tau_0p1
  run_rescore_tau_ablation_visual_refinement.sh --results-root /path/to/eval_results_tau_ablation_1shot
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --shot)
      SHOTS+=("$2")
      shift 2
      ;;
    --mode)
      MODES+=("$2")
      shift 2
      ;;
    --results-root)
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --gpu-id)
      GPU_ID="$2"
      shift 2
      ;;
    --summary-path)
      SUMMARY_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ${#SHOTS[@]} -eq 0 ]]; then
  SHOTS=(1 2 4 8)
fi

ARGS=(
  --bench-dir "${BENCH_DIR}"
  --device "cuda:0"
  --summary-path "${SUMMARY_PATH}"
)

if [[ -n "${RESULTS_ROOT}" ]]; then
  ARGS+=(--results-root "${RESULTS_ROOT}")
else
  ARGS+=(--shot "${SHOTS[@]}")
fi

if [[ ${#MODES[@]} -gt 0 ]]; then
  for MODE in "${MODES[@]}"; do
    ARGS+=(--mode "${MODE}")
  done
fi

echo "Rescoring tau ablation visual_refinement"
echo "  Bench dir: ${BENCH_DIR}"
if [[ -n "${RESULTS_ROOT}" ]]; then
  echo "  Results root: ${RESULTS_ROOT}"
else
  echo "  Shots: ${SHOTS[*]}"
fi
if [[ ${#MODES[@]} -gt 0 ]]; then
  echo "  Modes: ${MODES[*]}"
else
  echo "  Modes: auto-discover"
fi
echo "  GPU: ${GPU_ID}"
echo "  Python: ${PYTHON_BIN}"
echo "  Summary: ${SUMMARY_PATH}"
echo

cd "${BENCH_DIR}"
env HIP_VISIBLE_DEVICES="${GPU_ID}" CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  "${PYTHON_BIN}" "${BENCH_DIR}/rescore_tau_ablation_visual_refinement.py" \
  "${ARGS[@]}"
