import os
import glob
from itertools import chain
from datasets import load_dataset
from transformers import AutoTokenizer

# --- 0. 环境变量与缓存终极防护 ---
# 彻底杜绝往系统盘 ( /root/.cache ) 写数据的可能
CACHE_DIR = "/root/yang/cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_DIR, "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(CACHE_DIR, "hub")
os.environ["HF_HOME"] = CACHE_DIR

# 防止底层 C 库抢占 CPU 导致 Python 多进程死锁
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# --- 1. 核心参数 (针对 64核 + 256G内存 深度优化) ---
MODEL_PATH = "/root/yang/qwen3-8b"
DATA_DIRS = ["/data/data/Financial", "/data/data/General"]
SAVE_PATH = "/root/yang/Financial_qwen3_cpt"

BLOCK_SIZE = 2048
# 留 4 个核给系统内核和磁盘 I/O 调度，其余 60 个核全部拉满
NUM_PROC = 60 
# 有了 256G 内存，Batch Size 直接拉高 5 倍，大幅减少 I/O 频次
BATCH_SIZE = 5000 

def preprocess():
    print(f"启动极速预处理模式 | 核心数: {NUM_PROC} | 内存: 256GB+")
    
    # 2. Tokenizer 初始化
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 获取所有文件
    data_files = []
    for d in DATA_DIRS:
        data_files.extend(glob.glob(os.path.join(d, "*.jsonl")))
    print(f"找到 {len(data_files)} 个 JSONL 文件，开始加载...")

    # 3. 高速加载阶段
    # 将 num_proc 引入 load_dataset，让 JSON 解析阶段也多核并行
    raw_datasets = load_dataset("json", data_files=data_files, split="train", num_proc=NUM_PROC)

    # 4. Tokenize 与 注入 EOS
    def tokenize_function(examples):
        # 核心修复：强制在每一条文本末尾拼接 EOS 符号
        texts = [t + tokenizer.eos_token for t in examples["text"]]
        return tokenizer(texts, truncation=False, add_special_tokens=False)

    print("\nStep 1: 正在注入 EOS 并进行分词 (Tokenization)...")
    tokenized_ds = raw_datasets.map(
        tokenize_function,
        batched=True,
        batch_size=BATCH_SIZE,
        writer_batch_size=BATCH_SIZE, # 调大落盘块大小，提升写盘速度
        num_proc=NUM_PROC,
        remove_columns=raw_datasets.column_names,
        keep_in_memory=False, # 依然保持 False，借助 Arrow 的 mmap 机制最高效
        desc="Tokenizing & Adding EOS"
    )

    # 5. Packing 拼接机制 (严格切分 2048 长度)
    def group_texts(examples):
        # 展开所有序列并首尾相连
        concatenated_examples = {k: list(chain(*v)) for k, v in examples.items()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # 抛弃最后不足 2048 的零头
        if total_length >= BLOCK_SIZE:
            total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
        
        # 精准切块
        result = {
            k: [v[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
            for k, v in concatenated_examples.items()
        }
        # 语言模型自回归训练的特性：labels 与 input_ids 一致
        result["labels"] = result["input_ids"].copy()
        return result

    print(f"\nStep 2: 正在执行 Packing 数据拼接 (Target Length: {BLOCK_SIZE})...")
    lm_dataset = tokenized_ds.map(
        group_texts,
        batched=True,
        batch_size=BATCH_SIZE,
        writer_batch_size=BATCH_SIZE,
        num_proc=NUM_PROC,
        desc=f"Packing into {BLOCK_SIZE}"
    )

    # 6. 保存最终产物
    print(f"\nStep 3: 正在将处理好的 2048 块高速存入磁盘...")
    lm_dataset.save_to_disk(SAVE_PATH, num_proc=16) # 保存时 16 线程写盘最均衡
    print(f"数据集已保存在: {SAVE_PATH}")

if __name__ == "__main__":
    preprocess()