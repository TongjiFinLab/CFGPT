import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from transformers.trainer_utils import get_last_checkpoint

# ==========================================
# 1. 核心路径配置
# ==========================================
MODEL_PATH = "/root/yang/qwen3-8b"
#DATA_PATH = "/root/yang/Financial_qwen3_cpt/2/"
DATA_PATH = "/tmp/data/2/"
OUTPUT_DIR = "./saves"

# ==========================================
# 2. DeepSpeed ZeRO-3 配置字典 (极简安全版)
# ==========================================
ds_config = {
    "bf16": {
        "enabled": "auto" 
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "total_num_steps": "auto",
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": "auto"
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "steps_per_print": 10
}

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print("🚀 [Rank 0] 正在初始化正式训练参数...")

    # ==========================================
    # 3. 训练超参数配置 (正式全量跑)
    # ==========================================
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        
        # --- 批次与步数设置 ---
        # H200 141G，单卡 bs=16 毫无压力 (16 * 2048 tokens)。若遇到OOM，请降回 8。
        per_device_train_batch_size=16, 
        gradient_accumulation_steps=8,  # 全局 BS = 4 * 16 * 8 = 512 chunks (约 104 万 tokens/step)
        num_train_epochs=1,             # 跑完整个 1300万 chunks 数据 1 遍
        
        # --- 学习率与优化器 ---
        learning_rate=2e-5,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.95,
        lr_scheduler_type="cosine",
        warmup_ratio=0.02,              # 使用比例替代固定步数，前 2% 的进度用来预热
        
        # --- 精度与硬件加速 ---
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        
        # --- 日志与保存 (为长线训练保驾护航) ---
        logging_steps=10,               # 每 10 步打印一次日志
        save_strategy="steps",
        save_steps=500,                 # 每 500 步 (约消化 5.2 亿 tokens) 保存一次权重
        save_total_limit=3,             # 保留最近的 5 个权重，防止磁盘被撑爆
        report_to=["tensorboard"],      # 开启 TensorBoard 可视化监控
        
        # --- DeepSpeed 集成 ---
        deepspeed=ds_config,
        
        # --- 数据流向控制与 CPU 优化 ---
        remove_unused_columns=False,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
    )

    # ==========================================
    # 4. 加载模型与 Tokenizer
    # ==========================================
    if local_rank == 0:
        print(f"📦 [Rank 0] 正在从 {MODEL_PATH} 加载...")
        
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    
    # ==========================================
    # 5. 加载数据集
    # ==========================================
    if local_rank == 0:
        print(f"📖 [Rank 0] 正在从磁盘加载处理好的数据集: {DATA_PATH}...")
    dataset = load_from_disk(DATA_PATH)
    
    # ==========================================
    # 6. 初始化 Trainer 与 断点续训检测
    # ==========================================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer, 
        data_collator=default_data_collator, 
    )

    # 自动检测是否中断过，如果有，就从最近的 Checkpoint 恢复
#    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
#    if last_checkpoint is not None and local_rank == 0:
#        print(f"🔄 [Rank 0] 检测到历史检查点，将从 {last_checkpoint} 恢复断点续训！")

    # ==========================================
    # 7. 开始训练
    # ==========================================
    if local_rank == 0:
        print("🔥 [Rank 0] 开始全量分布式训练...")
        
    trainer.train()
    
    # ==========================================
    # 8. 保存最终模型
    # ==========================================
    if local_rank == 0:
        print(f"💾 [Rank 0] 训练完成，正在保存最终权重至 {OUTPUT_DIR}...")
        trainer.save_model()
        trainer.save_state()

if __name__ == "__main__":
    main()