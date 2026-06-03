#!/bin/bash

# --- 0. 激活正确的环境 ---
source /root/miniconda3/etc/profile.d/conda.sh
conda activate train

export CUDA_HOME="/root/cudahome"
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# --- 1. 指定所有缓存位置 (新增部分) ---
# 确保目录存在
mkdir -p /root/cache
# 数据集处理产生的 Arrow 缓存（最占空间的地方）
export HF_DATASETS_CACHE="/root/cache/datasets"
# 下载的模型权重、分词器缓存
export HUGGINGFACE_HUB_CACHE="/root/cache/hub"
# 某些旧版库可能使用的通配路径
export HF_HOME="/root/cache"
export TRITON_CACHE_DIR="/tmp/triton_cache"

# --- 2. 线程与算力分配 (针对 64 核 CPU) ---
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

# --- 3. H100/H200 Hopper 架构深度优化 ---
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export TORCH_CUDNN_V8_API_ENABLED=1

# --- 6. 启动训练 (4 卡并行) ---
echo "正在启动训练..."

# 使用 python -m 调用环境内的分布式模块
nohup python -m torch.distributed.run \
    --nproc_per_node=4 \
    qwen3_8b_pt_train.py > out.log 2>&1 &

echo "训练已转入后台！请运行 'tail -f out.log' 实时查看日志。"