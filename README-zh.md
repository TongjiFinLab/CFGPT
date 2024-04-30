<div style="text-align:center">
<!-- <img src="https://big-cheng.com/k2/k2.png" alt="k2-logo" width="200"/> -->
<h2>📈 CFGPT: Chinese Financial Assistant with Large Language Model</h2>
</div>

<a href='https://arxiv.org/abs/2309.10654'><img src='https://img.shields.io/badge/Paper-ArXiv-C71585'></a> 
<a href='https://huggingface.co/TongjiFinLab/CFGPT1-pt-7B'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging Face-CFGPT(pt)-red'></a> 
<a href='https://huggingface.co/TongjiFinLab/CFGPT1-sft-7B-LoRA'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging Face-CFGPT(sft%20LoRA)-red'></a> 
<a href='https://huggingface.co/TongjiFinLab/CFGPT1-sft-7B-Full'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging Face-CFGPT(sft%20Full)-red'></a> 

[English](README.md) | 简体中文

# 简介

**CFGPT**是一个开源的语言模型，首先通过在收集和清理的中国金融文本数据（CFData-pt）上进行继续预训练，包括金融领域特定数据（公告、金融文章、金融考试、金融新闻、金融研究论文）和通用数据（维基百科），然后使用知识密集的指导调整数据（CFData-sft）进行微调。
我们使用CFBenchmark-Basic进行初步评估。与几个具有相似参数的基线模型相比，CFGPT在识别，分类和生成任务上表现优越。

- 我们开发了CFGPT2 (13B)，这不仅是一个功能更加强大的中文金融大模型，同时还融合了检索增强模块，事实检测模块，合规检查模块和风险监测模块，在提升金融大模型服务的实时性与准确性的同时，有效监测与管控金融风险。
- 我们将CFGPT1 (7B) 分成三个部分发布：
    - [Pretrained Model](https://huggingface.co/TongjiFinLab/CFGPT1-pt-7B): 在中国金融文本语料库上进行进一步预训练且符合InternLM模型许可的完整模型权重。
    - [Supervised Finetuned Model (Lora)](https://huggingface.co/TongjiFinLab/CFGPT1-sft-7B-LoRA): 基于我们继续预训练模型的由PEFT（LoRA）训练的适配器模型权重。
    - [Supervised Finetuned Model (Full)](https://huggingface.co/TongjiFinLab/CFGPT1-sft-7B-Full): 基于我们继续预训练模型的进一步全参数微调的完整模型训练权重。

- 我们还发布了[CFBenchmark](https://github.com/TongjiFinLab/CFBenchmark)，这是一个针对大语言模型的中文金融基准测试。基础版本的CFBenchmark包括3917个金融文本，涵盖三个方面和八个任务，用于评估中文金融市场中LLM（大型语言模型）的金融文本处理能力。

- 我们还发布了CFGPT的进一步预训练和指导微调的代码。

- 我们还提供了CFData-sft的相关示例数据，以方便使用者理解我们的训练过程

***以下是训练 CFGPT 的流程概览图：***

<div align="center">
<img align="center" src=./figs/CFGPT-Training.svg width="100%"/>
</div>

# 目录

- [快速使用](#快速使用)
- [典型使用案例](#典型使用案例)
- [数据集](#数据集)
- [代码](#代码)
- [评测](#评测)
- [致谢](#致谢)
- [未来工作](#未来工作)
- [使用许可](#使用许可)
- [引用](#引用)

# 快速使用

**1. 准备代码和环境**

克隆我们的仓库，创建一个Python环境，并通过以下命令激活它：
```bash
git clone https://github.com/TongjiFinLab/CFGPT.git
cd CFGPT
conda create -n env_name python=3.10   
source activate env_name 
pip install -r requirements.txt
```

**2. 准备预训练的 CFGPT1**

CFGPT1由三个部分组成：一个继续预训练模型，将InternLM-7B在我们的CFData-pt上继续预训练，一个LoRA模型（通过PEFT在我们的CFData-sft上训练），以及基于继续预训练模型监督微调训练的全精调模型。

|Pretrain model|Adapter model|Full SFT Model|
|:-:|:-:|:-:|
 [CFGPT1-pt-7B](https://huggingface.co/TongjiFinLab/CFGPT1-pt-7B)|[CFGPT1-sft-7B-LoRA](https://huggingface.co/TongjiFinLab/CFGPT1-sft-7B-LoRA)|[CFGPT1-sft-7B-Full](https://huggingface.co/TongjiFinLab/CFGPT1-sft-7B-Full)|

**3. 使用 CFGPT1-sft-7B-LoRA**

```python
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
base_model = 'TongjiFinLab/CFGPT1-pt-7B'
lora_weights = 'TongjiFinLab/CFGPT1-sft-7B-LoRA'
device_map = 'cuda:0'
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = AutoModel.from_pretrained(
    base_model,
    trust_remote_code=True,
    device_map=device_map,
    torch_dtype=torch.bfloat16
)
model = PeftModel.from_pretrained(
    model,
    lora_weights,
    device_map=device_map,
)
model = model.eval()
inputs = tokenizer("""你是一名金融从业者，请对这篇新闻进行情感分析。请从（中性、积极、消极）中选取答案。新闻内容：挖贝快讯：特步国际发布2023年第二季度中国内地业务营运状况，披露截至2023年6月30日止3个月零售销售实现高双位数同比增长(包括线上线下渠道)，零售折扣水平约七五折。同时，2022年7月MSCI首次予以特步ESG评级，一年后评级表现即迎来提升。明晟MSCI上调特步ESG评级，由“BB”升至“BBB”。\n回答：""", return_tensors='pt').to(device_map)
pred = model.generate(**inputs, max_new_tokens=64, do_sample=False, repetition_penalty=1.0)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True).split('回答：')[1])
```

**4. 使用 CFGPT1-sft-7B-Full**

```python
from transformers import AutoModel, AutoTokenizer
base_model = 'TongjiFinLab/CFGPT1-sft-7B-Full'
device_map = 'cuda:0'
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = AutoModel.from_pretrained(
    base_model,
    trust_remote_code=True,
    device_map=device_map,
    torch_dtype=torch.bfloat16
)
model = model.eval()
inputs = tokenizer("""你是一名金融从业者，请对这篇新闻进行情感分析。请从（中性、积极、消极）中选取答案。新闻内容：挖贝快讯：特步国际发布2023年第二季度中国内地业务营运状况，披露截至2023年6月30日止3个月零售销售实现高双位数同比增长(包括线上线下渠道)，零售折扣水平约七五折。同时，2022年7月MSCI首次予以特步ESG评级，一年后评级表现即迎来提升。明晟MSCI上调特步ESG评级，由“BB”升至“BBB”。\n回答：""", return_tensors='pt').to(device_map)
pred = model.generate(**inputs, max_new_tokens=64, do_sample=False, repetition_penalty=1.0)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True).split('回答：')[1])
```

- **更多使用细节在 `./code/test`**

# 典型使用案例

- [CFGPT-v2-13B 银行业场景使用案例](cases/case_bank.md)
- [CFGPT-v2-13B 算法交易场景使用案例](cases/case_algor_trading.md)
- [CFGPT-v2-13B 第一产业场景使用案例](cases/case_primary_industry.md)
- [CFGPT-v1-7B 典型数据案例介绍](cases/case_CFGPTv1.md)

# 数据集

在这个存储库中，我们分享了CFData的样本：
- CFData：`./data`

    CFData包括一个预训练数据集（CFData-pt）和一个监督微调数据集（CFData-sft），其中预训练数据集汇集了中国金融数据和分析，以及一个较小的通用文本子集，总共有5.84亿个文档和1410亿个token，监督微调数据集专为六种不同的金融任务量身定制，涵盖了金融分析和决策制定的各个方面，共有150万个指令对和150亿个token。


## 继续预训练

预训练数据集包括 5.91 亿份文档和 1930 亿个token，包括六个子数据集：

* CFData-CP（6.24%）：包括 3,900 份公司招股说明书，共计 130 亿个token；
* CFData-CA（12.28%）：包括 600 万份公司公告，共计 170 亿个token；
* CFData-RR（2.51%）：包括 39.2 万份研究报告，共计 30 亿个token；
* CFData-FN（18.70%）：包括 8,200 万份财经新闻，共计 260 亿个token；
* CFData-SM（60.15%）：包括 4.95 亿份社交媒体内容，共计 840 亿个token；
* CFData-Wiki（0.09%）：包括 25.5 万份维基百科内容，共计 1.37 亿个token。

我们从CFData-pt中抽取了一个财经文本子语料库，以便在InternLM-7B上进行进一步的预训练。该子语料库包含了来自大量中国财经数据和分析以及少量通用文本的共计约137亿个token，这些通用文本包括公告、研究报告、社交媒体内容、财经新闻文章和维基百科等，而这些数据主要由我们自行收集。

## 有监督微调

监督微调数据集包括160万条指令对和15亿个标记，其中包括六个金融任务：
* CFData-SA（5.69%）：12万个实例，8600万标记用于情感分析；
* CFData-RS（50.60%）：36.9万个实例，7.65亿标记用于报告摘要；
* CFData-ED（22.69%）：49万个实例，3.43亿标记用于事件检测；
* CFData-TD（12.37%）：36.9万个实例，1.87亿标记用于主题分解；
* CFData-QA（0.39%）：1.2万个实例，600万标记用于问答；
* CFData-SP（8.27%）：21.2万个实例，1.25亿标记用于股票价格预测。

我们利用高质量的领域特定数据，通过有监督的微调来实现金融领域的适应性。该数据集包括六个金融数据集，以反映金融分析和决策的不同方面，包括情感分析、事件检测、报告摘要、主题分解、问题回答和股票走势预测。

CFData-sft提供了大量金融领域的文本信息，使FinLLM能够从不同的信息源中学习。

考虑到实际需求，我们将这些金融有监督微调数据集重组成十个任务。

以下是详细信息：
| 任务 | 任务描述 | 数据集 | 大小 |
| - | - | - | - |
| Sentiment | 识别与财务文件相关的情感 | CFData-SA | 13K |
| Summary | 基于提供的财务文件生成内容摘要 | CFData-RS | 18K |
| Risk | 基于提供的财务文件生成风险警报 | CFData-RS | 20K |
| Suggestion | 基于提供的财务文件生成投资建议 | CFData-RS | 18K |
| Event | 识别与财务文件相关的事件类别 | CFData-ED | 12K |
| Industry | 识别与财务文件相关的行业类别 | CFData-ED | 14K |
| Company | 识别与财务文件相关的公司名称 | CFData-ED | 12K |
| Product | 识别与财务文件相关的产品名称 | CFData-ED | 21K |
| Exam | 回答与财务问题相关的是非问题 | CFData-QA | 16K |
| Stock | 预测股票未来走势 | CFData-SP | 15K |

因为数据的许可问题, 我们不能公开发布完整版的CFData. 
研究人员可以参考我们[CFData](./data)的一些示例数据

# 代码

## 继续预训练

训练脚本在 **`./code/train/pretrain`**

```bash
deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 60002 bf_16_parallel_train.py --config bf_16_parallel_train.yml > bf_16_parallel_train.log 2>&1
```

<div align="center">
<img align="center" src=./figs/CFGPT-Training-loss.svg width="100%"/>
</div>

trainer的训练参数在 **`./code/train/pretrain/bf_16_parallel_train.yml`**: 
```
# basic setting
model_name: path/of/your/further/pretrain/model
dataset: path/to/your/further/pretrain/dataset
deepspeed: ./ds_config.json
seed: 42
max_seq_length: 2048

# train setting 
output_dir: ./bf_16_parallel_train
logging_steps: 10
num_train_epochs: 1
per_device_train_batch_size: 2
gradient_accumulation_steps: 16
learning_rate: 2.0e-4
weight_decay: 0.01
warmup_steps: 1000
save_steps: 1000
fp16: 0
bf16: 1
torch_compile: 0
save_strategy: steps
remove_unused_columns: 0
```

Deepspeed 训练参数在 **`./code/train/pretrain/ds_config.json`**: 
```
{
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false,
    "optimizer": {
        "type": "AdamW",
        "params": {
          "lr": "auto",
          "betas": "auto",
          "eps": "auto",
          "weight_decay": 0.01
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
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 1,
        "reduce_bucket_size": 5e8
    }
}
```

## 有监督微调

训练脚本位于 **`./code/train/lora`** 目录下。在这里，我们以 lora-bf16 作为示例。

```bash
deepspeed --include localhost:6,7 --master_port 60005 lora_bf_16_parallel_train.py --config lora_bf_16_parallel_train.yml > lora_bf_16_parallel_train.log 2>&1
```

Trainer 训练参数在 **`./code/train/lora/bf16/bf_16_parallel_train.yml`**: 
```
# basic setting
model_name: path/of/your/supervised/finetuning/model
dataset: path/to/your/supervised/finetuning/dataset
dataset_eval: path/to/your/evaluate/dataset
deepspeed: ./ds_config.json
seed: 42
max_seq_length: 2048

# train setting 
output_dir: ./lora_bf_16_parallel_train
num_train_epochs: 1
per_device_train_batch_size: 8
per_device_eval_batch_size: 8
gradient_accumulation_steps: 1
learning_rate: 2.0e-4
weight_decay: 0.01
warmup_steps: 500
fp16: 0
bf16: 1
torch_compile: 0
save_strategy: steps
save_steps: 500
evaluation_strategy: steps
eval_steps: 100
logging_steps: 10
remove_unused_columns: 0

# lora setting
rank: 64
lora_alpha: 16
lora_dropout: 0.05
target_modules: ['k_proj', 'o_proj', 'down_proj', 'v_proj', 'q_proj', 'gate_proj', 'up_proj']
bias: 'none'

# restart info
resume_from_checkpoint: null
```

Deepspeed训练参数在 **`./code/train/lora/bf16/ds_config.json`**: 
```
{
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false,

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
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },
    "bf16": {
      "enabled": true
    },
    "zero_optimization": {
        "stage": 0
    }
}
```

# 评测

CFGPT2的评测结果如下所示。

## C-Eval

| Model              | Size | STEM      | Social Science | Humanities | Others  | Average | Average(hard) |
| ------------------ | ---- | -------   | ------         | -----      | ------  | -----   | -------       |
| GPT-4              | -    | **67.1**  | 77.6           | 64.5       | 67.8    | 68.7    | **54.9**      |
| ChatGPT            | 175B | 52.9      | 61.8           | 50.9       | 53.6    | 54.4    | 41.4          |
| InternLM-7B        | 7B   | 48.0      | 67.4           | 55.4       | 45.8    | 52.8    | 37.1          |
| ChatGLM2-6B        | 6B   | 48.6      | 60.5           | 51.3       | 49.8    | 51.7    | 37.1          |
| Qwen-7B            | 7B   | 52.8      | 74.1           | 63.1       | 55.2    | 59.6    | 41.0          |
| Qwen-14B           | 14B  | 65.7      | **85.4**       | **75.3**   | **68.4**| **72.1**| 53.7          |
| Baichuan-7B        | 7B   | 38.2      | 52.0           | 46.2       | 39.3    | 42.8    | 31.5          |
| Baichuan-13B       | 13B  | 47.0      | 66.8           | 57.3       | 49.8    | 53.6    | 36.7          |
| Baichuan2-13B-Chat | 13B  | 48.4      | 70.5           | 60.3       | 55.0    | 56.6    | 37.9          |
| CFGPT-2            | 13B  | 47.6      | 70.0           | 60.7       | 54.5    | 56.2    | 33.7          |

## FinEval

| Model              | Size | Finance | Economy | Accounting | Certificate | Average | 
| ------------------ | ---- | ------- | ------  | -----      | ---------   | ---     |
| GPT-4              | -    | **71.0**| **74.5**| **59.3**   | **70.4**    | **68.6**| 
| ChatGPT            | 175B | 59.3    | 61.6    | 45.2       | 55.1        | 55.0    | 
| InternLM-7B        | 7B   | 49.0    | 49.2    | 40.5       | 49.4        | 47.1    | 
| ChatGLM2-6B        | 6B   | 46.5    | 46.4    | 44.5       | 51.5        | 47.4    | 
| Qwen-Chat-7B       | 7B   | 51.5    | 52.1    | 44.5       | 53.6        | 50.5    | 
| Qwen-7B            | 7B   | 54.5    | 54.4    | 50.3       | 55.8        | 53.8    | 
| Baichuan-7B-Chat   | 7B   | 44.9    | 41.5    | 34.9       | 45.6        | 42.0    | 
| Baichuan-13B-Chat  | 13B  | 51.6    | 51.1    | 41.7       | 52.8        | 49.4    | 
| CFGPT-2            | 13B  | 57.3    | 56.2    | 51.2       | 57.6        | 55.6    | 

## CFBenchmark-Basic
| Model              | Size | Company   | Product   | R.Avg     | Sector  | Event     | Sentiment | C.Avg     | Summary   | Risk      | Suggestion | G.Avg     | Avg       |
| ------------------ | ---- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | ---------- | --------- | --------- |
| HUMAN              | -    | 0.931     | 0.744     | 0.838     | 0.975     | 0.939     | 0.912     | 0.942     | 1.000     | 1.000     | 1.000      | 1.000     | 0.927     |
| ChatGPT            | 20B  | 0.797     | 0.198     | 0.498     | 0.453     | 0.458     | 0.425     | 0.455     | 0.593     | 0.541     | 0.771      | 0.635     | 0.529     |
| ERNIE-Bot          | 260B | 0.807     | 0.300     | 0.533     | 0.408     | 0.350     | 0.186     | 0.315     | 0.715     | 0.590     | 0.716      | 0.673     | 0.507     |
| ERNIE-Bot-4        | -    | 0.819     | 0.417     | 0.618     | 0.418     | 0.358     | 0.375     | 0.384     | 0.721     | 0.629     | 0.718      | 0.689     | 0.564     |
| Falcon-7B          | 7B   | 0.671     | 0.168     | 0.420     | 0.169     | 0.132     | 0.250     | 0.184     | 0.302     | 0.301     | 0.246      | 0.283     | 0.296     |
| Falcon-7B-chat     | 7B   | 0.582     | 0.046     | 0.314     | 0.112     | 0.142     | 0.153     | 0.135     | 0.307     | 0.299     | 0.258      | 0.288     | 0.246     |
| bloomz-7B1         | 7B   | 0.765     | 0.166     | 0.465     | 0.252     | 0.154     | 0.394     | 0.267     | 0.451     | 0.371     | 0.462      | 0.428     | 0.387     |
| bloomz-7Bt1-mt     | 7B   | 0.751     | 0.157     | 0.454     | 0.087     | 0.182     | 0.380     | 0.216     | 0.425     | 0.379     | 0.396      | 0.400     | 0.357     |
| Qwen-7B            | 7B   | 0.780     | 0.357     | 0.569     | 0.480     | 0.335     | 0.379     | 0.398     | 0.750     | 0.505     | 0.713      | 0.656     | 0.541     |
| Qwen-Chat-7B       | 7B   | 0.763     | 0.360     | 0.562     | 0.400     | 0.367     | 0.265     | 0.344     | 0.548     | 0.307     | 0.379      | 0.411     | 0.439     |
| Qwen-14B           | 14B  | 0.805     | 0.421     | 0.613     | 0.481     | 0.350     | 0.385     | 0.405     | 0.754     | 0.608     | 0.717      | 0.693     | 0.570     |
| Qwen-Chat-14B      | 14B  | 0.814     | 0.442     | 0.628     | 0.382     | 0.400     | 0.350     | 0.377     | 0.732     | 0.478     | 0.736      | 0.649     | 0.551     |
| ChatGLM2-6B        | 6B   | 0.747     | 0.313     | 0.530     | 0.285     | 0.300     | 0.357     | 0.314     | 0.657     | 0.454     | 0.671      | 0.594     | 0.479     |
| Baichuan2-7B-Base  | 7B   | 0.672     | 0.340     | 0.506     | 0.342     | 0.490     | 0.480     | 0.437     | 0.739     | 0.619     | 0.751      | 0.703     | 0.549     |
| Baichuan2-7B-Chat  | 7B   | 0.757     | 0.402     | 0.579     | 0.425     | 0.475     | 0.323     | 0.408     | 0.725     | 0.648     | 0.732      | 0.702     | 0.563     |
| Baichuan2-13B-Base | 13B  | 0.781     | 0.330     | 0.555     | 0.436     | 0.496     | 0.477     | 0.470     | 0.725     | 0.503     | 0.747      | 0.658     | 0.561     |
| Baichuan2-13B-Chat | 13B  | 0.797     | 0.314     | 0.556     | 0.472     | 0.507     | 0.387     | 0.455     | 0.739     | 0.634     | 0.746      | 0.706     | 0.572     |
| InternLM-7B        | 7B   | 0.612     | 0.233     | 0.423     | 0.266     | 0.311     | 0.328     | 0.302     | 0.378     | 0.336     | 0.379      | 0.364     | 0.363     |
| InternLM-7B-Chat   | 7B   | 0.632     | 0.261     | 0.447     | 0.272     | 0.364     | 0.399     | 0.345     | 0.363     | 0.270     | 0.353      | 0.329     | 0.374     |
| InternLM-20B       | 20B  | 0.809     | 0.358     | 0.583     | 0.500     | 0.427     | 0.417     | 0.448     | 0.706     | 0.653     | 0.728      | 0.695     | 0.575     |
| InternLM-20B-Chat  | 20B  | 0.488     | 0.362     | 0.425     | 0.323     | 0.327     | 0.370     | 0.340     | 0.706     | 0.578     | 0.762      | 0.662     | 0.476     |
| CFGPT1-stf-LoRA    | 7B   | 0.820     | 0.414     | 0.617     | 0.569     | 0.729     | 0.769     | 0.689     | 0.745     | 0.584     | 0.609      | 0.646     | 0.650     |
| CFGPT1-sft-Full    | 7B   | **0.836** | **0.476** | **0.656** | **0.700** | **0.808** | **0.829** | **0.779** | **0.798** | **0.669** | **0.808**  | **0.758** | **0.731** |
| CFGPT2             | 13B  |**0.861**|**0.490**|**0.676**|**0.722** |**0.835**|**0.831**  |**0.796**|**0.821**|**0.723**|**0.831**   |**0.792**|**0.755**|

## OpenFinData

| Model              | Size | Knowledge | Caluation | Explanation | Identification | Analysis | Compliance | Average | 
| ------------------ | ---- | -------   | ------    | -----       | ---------      | -----    | -------    | -----   |
| ERNIE-Bot-3.5      | -    | 78.0      | 70.4      | 82.1        | 75.3           | 77.7     | 36.7       | 70.0    | 
| ERNIE-Bot-4        | -    | **87.3**  | **73.6**  | **84.3**    | **77.0**       | **79.1** | 37.3       |**73.1** | 
| InternLM-7B        | 7B   | 65.3      | 45.8      | 71.4        | 62.5           | 59.2     | 37.2       | 56.9    | 
| ChatGLM2-6B        | 6B   | 62.4      | 37.2      | 70.8        | 59.2           | 58.3     | 38.7       | 54.4    | 
| Qwen-Chat-7B       | 7B   | 71.3      | 40.5      | 71.4        | 58.6           | 51.3     | 40.0       | 55.5    | 
| Qwen-Chat-14B      | 14B  | 78.0      | 57.6      | 75.6        | 71.6           | 59.3     | 40.6       | 63.8    | 
| Baichuan2-7B-Chat  | 7B   | 46.2      | 37.0      | 76.5        | 60.2           | 55.0     | 28.7       | 50.6    | 
| Baichuan2-13B-Chat | 13B  | 69.3      | 39.5      | 75.3        | 65.7           | 62.0     | 31.3       | 57.2    | 
| CFGPT-2            | 13B  | 86.7      | 64.3      | 77.3        | 73.8           | 65.2     |**70.2**    | 72.9    | 


# 致谢

CFGPT已参考了以下开源项目。我们要向这些项目的研究者表示感谢和尊重。

- InternLM: https://github.com/InternLM/InternLM
- Firefly: https://github.com/yangjianxin1/Firefly
- FinGPT: https://github.com/AI4Finance-Foundation/FinGPT


# 未来工作
- [ ] 使用CFGPT创建下游的应用CFAPP
- [ ] 构建更加全面的训练任务与对应数据
- [ ] 持续性改进CFGPT在更多复杂金融任务上的能力

# 使用许可
CFGPT的代码遵循Apache许可证2.0协议。CFGPT的模型免费开源，商用许可遵循开源基础模型InternLM 7B和百川2-13B的许可协议和OpenAI生成数据的使用条款。如果您发现任何潜在的违规行为，请与我们联系。

### 感谢我们的贡献者 :
<a href="https://github.com/TongjiFinLab/CFGPT/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TongjiFinLab/CFGPT" />
</a>

# 引用
如果您认为**CFGPT**对您的研究有帮助，可以引用以下的论文：

```
@misc{li2023cfgpt,
      title={CFGPT: Chinese Financial Assistant with Large Language Model}, 
      author={Jiangtong Li and Yuxuan Bian and Guoxuan Wang and Yang Lei and Dawei Cheng and Zhijun Ding and Changjun Jiang},
      year={2023},
      eprint={2309.10654},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
```
@article{li2024racfgpt,
         author = {Jiangtong Li, Yang Lei, Yuxuan Bian, Dawei Cheng, Zhijun Ding, Changjun Jiang},
         title = {RA-CFGPT: Chinese Financial Assistant with Retrieval-Augmented Large Language Model},
         publisher = {Front. Comput. Sci.},
         year = {2024},
         journal = {Frontiers of Computer Science},
         volume = {},
         number = {},
         doi = {10.1007/s11704-024-31018-5}
}    
```