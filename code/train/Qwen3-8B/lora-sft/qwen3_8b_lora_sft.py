"""
Qwen3-8B LoRA SFT。
只更新 LoRA。模板与 qwen3_nothink 一致：ChatML，不插入 think 块。
只对 assistant 回答（含 <|im_end|>）计算损失，不 packing。
"""

import argparse
import os

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


ROLE = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "chatbot": "assistant",
    "assistant": "assistant",
    "system": "system",
}

TOKENIZER = None
CUTOFF_LEN = 4096


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="qwen3_8b_lora_sft.yml")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key, value in cfg.items():
        print(f"{key}: {value}")
    return cfg


def encode(text):
    return TOKENIZER.encode(text, add_special_tokens=False)


def extend_piece(cursor, piece, learn, input_ids, labels):
    """按累积文本取新增 token，避免分段编码时 BPE 在边界被切开。"""
    prev = encode(cursor)
    cursor = cursor + piece
    full = encode(cursor)
    if full[: len(prev)] != prev:
        added = encode(piece)
    else:
        added = full[len(prev) :]
    input_ids.extend(added)
    labels.extend(added if learn else [-100] * len(added))
    return cursor


def to_messages(example):
    if isinstance(example.get("messages"), list):
        raw = example["messages"]
    elif isinstance(example.get("conversations"), list):
        raw = example["conversations"]
    else:
        return None

    messages = []
    for turn in raw:
        if "role" in turn and "content" in turn:
            role = ROLE.get(turn["role"])
            content = turn["content"]
        elif "from" in turn and "value" in turn:
            role = ROLE.get(turn["from"])
            content = turn["value"]
        else:
            return None
        if role is None or not isinstance(content, str) or not content.strip():
            return None
        messages.append({"role": role, "content": content})
    if not any(message["role"] == "assistant" for message in messages):
        return None
    return messages


def supervised_example(messages):
    cursor = ""
    input_ids = []
    labels = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "assistant":
            cursor = extend_piece(cursor, "<|im_start|>assistant\n", False, input_ids, labels)
            cursor = extend_piece(cursor, content + "<|im_end|>\n", True, input_ids, labels)
        else:
            piece = f"<|im_start|>{role}\n{content}<|im_end|>\n"
            cursor = extend_piece(cursor, piece, False, input_ids, labels)
    input_ids = input_ids[:CUTOFF_LEN]
    labels = labels[:CUTOFF_LEN]
    if not any(label != -100 for label in labels):
        return None
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def encode_example(example):
    messages = to_messages(example)
    encoded = supervised_example(messages) if messages else None
    if encoded is None:
        return {"input_ids": [], "attention_mask": [], "labels": []}
    return encoded


def load_model(model_name):
    kwargs = {"trust_remote_code": True}
    for attn in ("flash_attention_2", "sdpa"):
        try:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, dtype=torch.bfloat16, attn_implementation=attn, **kwargs
                )
            except TypeError:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=torch.bfloat16, attn_implementation=attn, **kwargs
                )
            print(f"attn_implementation: {attn}")
            return model
        except Exception as exc:
            print(f"skip attn {attn}: {exc}")
    raise RuntimeError(f"无法加载模型: {model_name}")


def build_trainer(cfg, model, dataset):
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        num_train_epochs=float(cfg["num_train_epochs"]),
        lr_scheduler_type=cfg["lr_scheduler_type"],
        lr_scheduler_kwargs={"min_lr_rate": float(cfg["min_lr_rate"])},
        warmup_ratio=float(cfg["warmup_ratio"]),
        bf16=bool(cfg["bf16"]),
        gradient_checkpointing=bool(cfg["gradient_checkpointing"]),
        logging_steps=cfg["logging_steps"],
        save_strategy=cfg["save_strategy"],
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        seed=cfg["seed"],
        dataloader_num_workers=cfg["dataloader_num_workers"],
        remove_unused_columns=False,
        report_to="none",
        ddp_find_unused_parameters=False,
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer=TOKENIZER,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=TOKENIZER,
    )


def main():
    global TOKENIZER, CUTOFF_LEN
    cfg = parse_args()
    if cfg.get("packing"):
        raise ValueError("本脚本不支持 packing，请将 packing 设为 false。")
    if cfg.get("enable_thinking"):
        raise ValueError("本脚本使用 qwen3_nothink 模板，enable_thinking 必须为 false。")
    train_file = cfg["train_file"]
    if not os.path.isfile(train_file):
        raise FileNotFoundError(
            f"找不到训练数据: {train_file}\n"
            "请把 yml 里的 train_file 改成本地 jsonl（训练数据暂不公开）。"
        )

    set_seed(cfg["seed"])
    CUTOFF_LEN = int(cfg["cutoff_len"])
    TOKENIZER = AutoTokenizer.from_pretrained(
        cfg["model_name_or_path"], trust_remote_code=bool(cfg["trust_remote_code"])
    )
    if TOKENIZER.pad_token is None:
        TOKENIZER.pad_token = TOKENIZER.eos_token
    TOKENIZER.padding_side = "right"

    model = load_model(cfg["model_name_or_path"])
    model.config.use_cache = False
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    if cfg["gradient_checkpointing"]:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    raw = load_dataset("json", data_files=train_file, split="train")
    dataset = raw.map(
        encode_example,
        remove_columns=raw.column_names,
        num_proc=cfg["preprocessing_num_workers"] or None,
        desc="tokenize",
    )
    before = len(dataset)
    dataset = dataset.filter(lambda row: len(row["input_ids"]) > 0)
    print(f"samples: {len(dataset)} / {before}")
    if len(dataset) == 0:
        raise RuntimeError("没有可用样本。请检查 jsonl 是否为 conversations 或 messages。")

    os.makedirs(cfg["output_dir"], exist_ok=True)
    trainer = build_trainer(cfg, model, dataset)
    resume_from = None
    if cfg.get("resume", True):
        resume_from = get_last_checkpoint(cfg["output_dir"])
        if resume_from:
            print(f"resume: {resume_from}")
    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(cfg["output_dir"])
    TOKENIZER.save_pretrained(cfg["output_dir"])
    trainer.save_state()


if __name__ == "__main__":
    main()
