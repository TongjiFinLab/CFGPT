#!/bin/bash

# --- 0. 激活正确的环境 ---
source /root/miniconda3/etc/profile.d/conda.sh
conda activate train

# --- 1. 路径重定向配置 (保持开发机整洁) ---
BASE_CACHE="/root/cache"
mkdir -p $BASE_CACHE/huggingface $BASE_CACHE/datasets $BASE_CACHE/triton $BASE_CACHE/tmp

export CUDA_HOME="/root/cudahome"
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export HF_HOME="$BASE_CACHE/huggingface"
export HF_DATASETS_CACHE="$BASE_CACHE/datasets"
export TRITON_CACHE_DIR="$BASE_CACHE/triton"
export TMPDIR="$BASE_CACHE/tmp"

# 注意：这里已经去掉了废弃的 CUDA_ 字样，防止弹出警告
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export TORCH_CUDNN_V8_API_ENABLED=1

echo "环境初始化完成，正在启动训练..."




# 确保显卡可见
export CUDA_VISIBLE_DEVICES=0,1
# 启动
accelerate launch \
    --num_processes 2 \
    --mixed_precision bf16 \
    qwen3_8b_sft.py >out.log 2>&1 &
echo "训练已转入后台！请运行 'tail -f out.log' 实时查看日志。"