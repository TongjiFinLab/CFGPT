import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

# ================= 1. 路径配置 =================
model_id = "/root/yang/qwen3_8b_cpt/2/saves/qwen3_8b_cpt"
base_data_path = "/root/yang/Financial_qwen3_sft"

# 包含所有修复后的核心训练文件
train_files = [
    os.path.join(base_data_path, f) 
    for f in [
        "fin_nlp.jsonl", 
        "fin_conv.jsonl", 
        "fin_exam.jsonl", 
        "moss.jsonl", 
        "cfgpt_identity.jsonl" # 使用修复后的身份文件
    ]
]
test_files = [os.path.join(base_data_path, "cfgpt_eval.jsonl")]

data_files = {"train": train_files, "test": test_files}

# ================= 2. 加载分词器与模型 =================
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16, # H200 硬件加速
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    device_map=None           # 多卡训练必须设为 None
)

# ================= 3. 数据预处理逻辑 =================
def format_instruction(sample):
    """
    统一处理 'messages' 格式的数据
    """
    if "messages" in sample and isinstance(sample["messages"], list):
        # 直接使用消息列表，apply_chat_template 会处理 role 和 content
        prompt = tokenizer.apply_chat_template(
            sample["messages"], 
            tokenize=False, 
            add_generation_prompt=False
        )
        return {"text": prompt}
    return {"text": ""}

# 加载并转换
dataset = load_dataset("json", data_files=data_files)
column_names = dataset["train"].column_names
dataset = dataset.map(
    format_instruction,
    remove_columns=column_names,
    desc="Converting to ChatML format"
)

# 过滤掉可能的空行
dataset = dataset.filter(lambda x: len(x["text"]) > 0)

from trl import SFTTrainer, SFTConfig  # 确保导入了 SFTConfig

# ================= 4. 训练配置 (适配最新版 TRL) =================
# 基于你提供的参数列表进行匹配
sft_config = SFTConfig(
    output_dir="./qwen3_8b_sft_output",
    
    # 基础训练参数 (继承自 TrainingArguments)
    eval_strategy="steps",
    eval_steps=200,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_train_epochs=3,
    bf16=True,
    gradient_checkpointing=True,
    warmup_steps=100,
    ddp_find_unused_parameters=False,
    logging_steps=10,
    deepspeed="/root/yang/ds_z2_config.json",
    
    # SFTConfig 特有参数 (对应你列表中的参数名)
    packing=True,
    dataset_text_field="text", # 对应 format_instruction 返回的 key
    max_length=2048,           # 你的列表里是 max_length，不是 max_seq_length
)

# ================= 5. 初始化 Trainer =================
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"], # 必须传入 map 后的 dataset 对象
    eval_dataset=dataset["test"],   # 必须传入 map 后的 dataset 对象
    args=sft_config,
)

print("--- 🚀 CFGPT 8B SFT 任务正式启动 ---")
trainer.train()


# ================= 6. 保存最终模型 =================
trainer.save_model("./qwen3_8b_sft_final")
tokenizer.save_pretrained("./qwen3_8b_sft_final")
print("--- ✅ 训练完成，模型已保存 ---")