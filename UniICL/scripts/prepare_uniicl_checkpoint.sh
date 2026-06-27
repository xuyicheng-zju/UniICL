#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPEN_SOURCE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PRETRAINED_MODEL="$(python3 "${OPEN_SOURCE_ROOT}/local_paths.py" --get UNIICL_BASE_MODEL)"
CHECKPOINT_DIR="$(python3 "${OPEN_SOURCE_ROOT}/local_paths.py" --get UNIICL_TARGET_CHECKPOINT)"

echo "Preparing UniICL checkpoint for evaluation..."
echo "Source: ${PRETRAINED_MODEL}"
echo "Target: ${CHECKPOINT_DIR}"

mkdir -p "${CHECKPOINT_DIR}"
cd "${CHECKPOINT_DIR}"

for file in vocab.json merges.txt tokenizer.json tokenizer_config.json llm_config.json vit_config.json ae.safetensors; do
    if [[ -f "${PRETRAINED_MODEL}/${file}" && ! -f "${CHECKPOINT_DIR}/${file}" ]]; then
        cp "${PRETRAINED_MODEL}/${file}" .
    fi
done

echo "Prepared checkpoint contents:"
ls -lh "${CHECKPOINT_DIR}"
