#!/bin/bash
# Qwen3-8B LoRA SFT。
set -euo pipefail
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python qwen3_8b_lora_sft.py --config "${CONFIG:-qwen3_8b_lora_sft.yml}" 2>&1 | tee -a lora_sft.log
