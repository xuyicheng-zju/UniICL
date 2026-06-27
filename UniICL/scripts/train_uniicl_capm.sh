#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNICL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OPEN_SOURCE_ROOT="$(cd "${UNICL_ROOT}/.." && pwd)"
export PYTHONPATH="${UNICL_ROOT}:${PYTHONPATH:-}"

MODEL_PATH="$(python3 "${OPEN_SOURCE_ROOT}/local_paths.py" --get UNIICL_BASE_MODEL)"
DEFAULT_RESUME_FROM="${MODEL_PATH}"
OUTPUT_DIR="${UNICL_RESULTS_DIR:-${UNICL_ROOT}/results/uniicl_unified_icl_capm}"
CHECKPOINT_DIR="${UNICL_CHECKPOINT_DIR:-${OUTPUT_DIR}/checkpoints}"
DATASET_CONFIG="${UNICL_DATASET_CONFIG:-${UNICL_ROOT}/data/configs/unified_icl.yaml}"

NUM_NODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
WORLD_SIZE="$((NUM_NODES * NPROC_PER_NODE))"
NUM_REPLICATE="${NUM_REPLICATE:-1}"
NUM_SHARD="${NUM_SHARD:-${WORLD_SIZE}}"
SHARDING_STRATEGY="${SHARDING_STRATEGY:-HYBRID_SHARD}"

TOTAL_STEPS="${TOTAL_STEPS:-10000}"
LR="${LR:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
SAVE_EVERY="${SAVE_EVERY:-2500}"
LOG_EVERY="${LOG_EVERY:-10}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EXPECTED_NUM_TOKENS="${EXPECTED_NUM_TOKENS:-24576}"
MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-28672}"
MAX_NUM_TOKENS_PER_SAMPLE="${MAX_NUM_TOKENS_PER_SAMPLE:-24576}"
SEED="${SEED:-42}"

WANDB_PROJECT="${WANDB_PROJECT:-uniicl_unified_icl_capm}"
WANDB_NAME="${WANDB_NAME:-uniicl_unified_icl_capm_$(date +%Y%m%d_%H%M%S)}"
WANDB_OFFLINE="${WANDB_OFFLINE:-False}"
AUTO_RESUME="${UNICL_AUTO_RESUME:-False}"
RESUME_FROM="${UNICL_RESUME_FROM:-${DEFAULT_RESUME_FROM}}"
RESUME_MODEL_ONLY="${UNICL_RESUME_MODEL_ONLY:-True}"
FINETUNE_FROM_EMA="${UNICL_FINETUNE_FROM_EMA:-True}"
CAPM_APPLY_TO_DEMO_TOKENS="${CAPM_APPLY_TO_DEMO_TOKENS:-False}"
CE_WEIGHT="${CE_WEIGHT:-1.0}"
MSE_WEIGHT="${MSE_WEIGHT:-1.0}"

mkdir -p "${OUTPUT_DIR}" "${CHECKPOINT_DIR}"

CMD=(
  torchrun
  --nnodes="${NUM_NODES}"
  --node_rank="${NODE_RANK}"
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_addr="${MASTER_ADDR}"
  --master_port="${MASTER_PORT}"
  train/pretrain_unified_navit.py
  --dataset_config_file "${DATASET_CONFIG}"
  --model_path "${MODEL_PATH}"
  --layer_module Qwen2MoTDecoderLayer
  --max_latent_size 64
  --finetune_from_hf True
  --num_replicate "${NUM_REPLICATE}"
  --num_shard "${NUM_SHARD}"
  --sharding_strategy "${SHARDING_STRATEGY}"
  --auto_resume "${AUTO_RESUME}"
  --resume-model-only "${RESUME_MODEL_ONLY}"
  --finetune-from-ema "${FINETUNE_FROM_EMA}"
  --results_dir "${OUTPUT_DIR}"
  --checkpoint_dir "${CHECKPOINT_DIR}"
  --total_steps "${TOTAL_STEPS}"
  --lr "${LR}"
  --warmup_steps "${WARMUP_STEPS}"
  --save_every "${SAVE_EVERY}"
  --log_every "${LOG_EVERY}"
  --num_workers "${NUM_WORKERS}"
  --expected_num_tokens "${EXPECTED_NUM_TOKENS}"
  --max_num_tokens "${MAX_NUM_TOKENS}"
  --max_num_tokens_per_sample "${MAX_NUM_TOKENS_PER_SAMPLE}"
  --data_seed "${SEED}"
  --visual_gen True
  --visual_und True
  --wandb_project "${WANDB_PROJECT}"
  --wandb_name "${WANDB_NAME}"
  --wandb_offline "${WANDB_OFFLINE}"
  --use_capm True
  --capm_d_capm 768
  --capm_num_probes 32
  --capm_num_inject_layers 28
  --capm_cross_attn_heads 8
  --capm_operator_rank 64
  --capm_op_gain 0.1
  --capm_apply_to_demo_tokens "${CAPM_APPLY_TO_DEMO_TOKENS}"
  --ce_weight "${CE_WEIGHT}"
  --mse_weight "${MSE_WEIGHT}"
)

if [[ -n "${RESUME_FROM}" ]]; then
  CMD+=(--resume-from "${RESUME_FROM}")
fi

echo "=========================================="
echo "UniICL Unified ICL Training (CAPM)"
echo "=========================================="
echo "UniICL Root: ${UNICL_ROOT}"
echo "UniICL-760K Config: ${DATASET_CONFIG}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Model Path: ${MODEL_PATH}"
echo "Auto Resume: ${AUTO_RESUME}"
echo "Resume From: ${RESUME_FROM}"
echo "Resume Model Only: ${RESUME_MODEL_ONLY}"
echo "Finetune From EMA: ${FINETUNE_FROM_EMA}"
echo "Num Replicate: ${NUM_REPLICATE}"
echo "Num Shard: ${NUM_SHARD}"
echo "Sharding Strategy: ${SHARDING_STRATEGY}"
echo "=========================================="

cd "${UNICL_ROOT}"
"${CMD[@]}"
