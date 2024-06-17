<div style="text-align:center">
<!-- <img src="https://big-cheng.com/k2/k2.png" alt="k2-logo" width="200"/> -->
<h2>📈 CFGPT: Chinese Financial Assistant with Large Language Model</h2>
</div>


<a href='https://arxiv.org/abs/2309.10654'><img src='https://img.shields.io/badge/Paper-ArXiv-C71585'></a> 
<a href='https://huggingface.co/TongjiFinLab/CFGPT1-pt-7B'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging Face-CFGPT(pt)-red'></a> 
<a href='https://huggingface.co/TongjiFinLab/CFGPT1-sft-7B-LoRA'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging Face-CFGPT(sft%20LoRA)-red'></a> 
<a href='https://huggingface.co/TongjiFinLab/CFGPT1-sft-7B-Full'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging Face-CFGPT(sft%20Full)-red'></a> 

English | [简体中文](README-zh.md)

# Introduction

We introduce **CFGPT**, an open-source language model trained by firstly further pretraining general LLMs on collected and cleaned Chinese finance text data (CFData-pt), including financial domain-specific data (announcement, finance articles, finance exams, finance news, finance research papers) and general data (Wikipedia), and secondly fine-tuning with knowledge-intensive instruction tuning data (CFData-sft). 
As for preliminary evaluation, we use CFBenchmark-Basic. 
CFGPT outperforms the baselines on objective and subjective tasks compared to several baseline models with similar parameters. 

- We develop CFGPT2 (7B&20B), which is not only a more powerful Chinese financial large model but also integrates modules for retrieval enhancement, fact verification, compliance checking, and risk monitoring. While enhancing the real-time performance and accuracy of financial large model services, it effectively monitors and controls financial risks.
- We release CFGPT1 (7B) in three parts:
  - [Pretrained Model](https://huggingface.co/TongjiFinLab/CFGPT1-pt-7B): Full model weights after further pretraining with the chinese finance text corpus to comply with the InternLM model license. 
  - [Supervised Finetuned Model (Lora)](https://huggingface.co/TongjiFinLab/CFGPT1-sft-7B-LoRA): Adapter model weights trained by PEFT (LoRA).
  - [Supervised Finetuned Model (Full)](https://huggingface.co/TongjiFinLab/CFGPT1-sft-7B-Full): Full model trained weights based on the pretrained model.

- We release the [CFBenchmark](https://github.com/TongjiFinLab/CFBenchmark), a Chinese financial assistant benhmark for large language model. The basic version of CFBenchmark includes 3917 financial texts spanning three aspects and eight tasks, for the evaluation of the financial text processing capability of LLMs in Chinese financial market.

- We release the code of further pretrain and instruction tuning of CFGPT.

- We further provide several samples about our CFData-sft.

***The following is the overview of training CFGPT:***

<div align="center">
<img align="center" src=./figs/CFGPT-Training.svg width="100%"/>
</div>


# Content

- [Quick Start](#quick-start)
- [User Cases](#user-cases)
- [Data](#data)
- [Code](#code)
- [Evaluation](#evaluation)
- [Acknowledgements](#acknowledgements)
- [To-Do List](#to-do-list)
- [License](#license)
- [Citation](#citation-arxiv)

# Quick Start

**1. Prepare the code and the environment**

Clone our repository, create a Python environment, and activate it via the following command

```bash
git clone https://github.com/TongjiFinLab/CFGPT.git
cd CFGPT
conda create -n env_name python=3.10   
source activate env_name 
pip install -r requirements.txt
```

**2. Prepare the pretrained CFGPT1**

The CFGPT1 consists of three parts: a pretrain model, continued pretraining InternLM-7B on our CFData-pt, an adapter model (trained via PEFT on our CFData-sft), and a Full-finetuned model trained base on the pretrain model.

|                        Pretrain model                        |                        Adapter model                         |                        Full SFT Model                        |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [CFGPT1-pt-7B](https://huggingface.co/TongjiFinLab/CFGPT1-pt-7B) | [CFGPT1-sft-7B-lora](https://huggingface.co/TongjiFinLab/CFGPT1-sft-7B-LoRA) | [CFGPT1-sft-7B-full](https://huggingface.co/TongjiFinLab/CFGPT1-sft-7B-Full) |

**3. Use CFGPT1-sft-7B-LoRA**

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

**4. Use CFGPT1-sft-7B-Full**

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

- **More detail are in `./code/test`**

# User Cases

- [CFGPT-v2-7B User Cases about Banking](cases/case_bank.md)
- [CFGPT-v2-7B User Cases about Algorithm Trading](cases/case_algor_trading.md)
- [CFGPT-v2-7B User Cases about Primary Industry](cases/case_primary_industry.md)
- [CFGPT-v1-7B Dataset Cases](cases/case_CFGPTv1.md)

# Data

In this repo, we share the samples of CFData:

- CFData: `./data`

  The CFData comprising both a pre-training dataset and a supervised fine-tuning dataset, where the pre-training dataset collates Chinese financial data and analytics, alongside a smaller subset of general-purpose text with 584M documents and 141B tokens in total, and the supervised finetuning dataset is tailored for six distinct financial tasks, embodying various facets of financial analysis and decisionmaking with 1.5M instruction pairs and 1.5B tokens in total.


## Further pretrain

The pre-training dataset consists of 591 million documents and 193 billion tokens, including six sub-datasets

* CFData-CP (6.24%): 39 thousand corporate prospectus with 13 billion tokens;
* CFData-CA (12.28%): 6 million corporate announcements with 17 billion tokens; 
* CFData-RR (2.51% ): 392 thousand research reports with 3 billion tokens; 
* CFData-FN (18.70%): 82 million financial news with 26 billion tokens; 
* CFData-SM (60.15%): 495 million social medias and 84 billion tokens; 
* CFData-Wiki (0.09%): 255 thousand Wikipedia content with 137 million tokens.

We sample a financial text sub-corpus from CFData-pt for further pretraining on InternLM-7B consists of 13.7 billion tokens from a large amount of Chinese financial data and analytics and a small amount of general-purpose text, such as announcements, research reports, social media content, financial news articles, and Wikipedia. And they were mainly collected by ourselves.

## Supervised Finetuning

The supervised fine-tuning dataset consist 1.6 million instructions pairs and 1.5 billion tokens, including six financial tasks: 

* CFData-SA (5.69% ): 120 thousand instances with 86 million tokens for sentiment analysis; 
* CFData-RS (50.60%): 369 thousand instances and 765 million tokens for report summary; 
* CFData-ED (22.69% ): 490 thousand instances with 343 million tokens for event detection; 
* CFData-TD (12.37%): 369 thousand instances and 187 million tokens for topic decomposition; 
* CFData-QA (0.39%): 12 thousand instances and 6 million tokens for question-answering; 
* CFData-SP (8.27%): 212 thousand instances and 125 million tokens for stock moving prediction.

We employ high-quality domain specific data to achieve finance domain adaptation during supervised finetuing. The dataset includes six financial datasets to reflect different aspects of financial analysis and decision-making, which include sentiment analysis, event detection, report summarization, topic decomposition, question answering, and stock movement prediction. 
CFData-sft provides much text information in the financial domain, allowing a FinLLM to learn from different of sources.
Considering requirement in reality, we reform these financial supervised finetuning dataset into ten tasks.

The details are as follows:

| Task       | Task Description                                             | Dataset   | Size |
| ---------- | ------------------------------------------------------------ | --------- | ---- |
| Sentiment  | Identify the sentiment associated with financial document    | CFData-SA | 13K  |
| Summary    | Generate a content summary based on the provided financial document | CFData-RS | 18K  |
| Risk       | Generate risk alerts based on the provided financial document | CFData-RS | 20K  |
| Suggestion | Generate investment recommendations based on the provided financial document | CFData-RS | 18K  |
| Event      | Identify the event categories associated with financial document | CFData-ED | 12K  |
| Industry   | Identify the industry categories associated with financial document | CFData-ED | 14K  |
| Company    | Identify the company names associated with financial document | CFData-ED | 12K  |
| Product    | Identify the product names associated with financial document | CFData-ED | 21K  |
| Exam       | Answer true-false questions related to finance question      | CFData-QA | 16K  |
| Stock      | Predict stocks future movement                               | CFData-SP | 15K  |


Due to the data agreement, we cannot publicly release the full version of CFData. The researchers could read the sample case of [CFData](./data)

# Code

## Further Pretrain

The training script is **`./code/train/pretrain`**

```bash
deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 60002 bf_16_parallel_train.py --config bf_16_parallel_train.yml > bf_16_parallel_train.log 2>&1
```

<!-- ![loss curve](https://big-cheng.com/k2/loss_curve.png) -->

<div align="center">
<img align="center" src=./figs/CFGPT-Training-loss.svg width="100%"/>
</div>


The trainer parameters we use are in **`./code/train/pretrain/bf_16_parallel_train.yml`**: 

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

The deepspeed parameters we use are in **`./code/train/pretrain/ds_config.json`**: 

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

## Supervised Finetuning

The training script is in **`./code/train/lora`**. Here we use the lora-bf16 as illustrations.

```bash
deepspeed --include localhost:6,7 --master_port 60005 lora_bf_16_parallel_train.py --config lora_bf_16_parallel_train.yml > lora_bf_16_parallel_train.log 2>&1
```

The trainer parameters we use are in **`./code/train/lora/bf16/bf_16_parallel_train.yml`**: 

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

The deepspeed parameters we use are in **`./code/train/lora/bf16/ds_config.json`**: 

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

# Evaluation

The performance of our CFGPT2 (13B) is shown as follows:

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
| InternLM2-7B       | 7B   | 52.3      | 71.9           | 64.9       | 61.0    | 60.8    | 38.8          |
| InternLM2-20B      | 20B  | 56.1      | 75.7           | 62.6       | 62.4    | 63.0    | 46.3          |
| CFGPT2-7B          | 7B   | 56.7      | 76.4           | 63.9       | 63.0    | 63.5    | 43.2          |
| CFGPT2-20B         | 20B  | 64.6      | 80.8           | 72.1       | 68.9    | 69.2    | 49.9          |

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
| InternLM2-7B       | 7B   | 54.2    | 54.0    | 43.5       | 55.4        | 51.9    |
| InternLM2-20B      | 20B  | 57.3    | 58.9    | 47.4       | 58.6        | 55.5    |
| CFGPT2-7B          | 7B   | 62.6    | 63.9    | 58.9       | 66.0        | 62.9    |
| CFGPT2-20B         | 20B  | 64.0    | 64.9    | 62.1       | 67.9        | 64.8    |

## CFBenchmark-Basic
| Model              | Size | Company   | Product   | R.Avg     | Sector    | Event     | Sentiment | C.Avg     | Summary   | Risk      | Suggestion | G.Avg     | Avg       |
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
| CFGPT2-7B          | 7B   | **0.834** | **0.470** | **0.652** | **0.644** | **0.750** | **0.793** | **0.729** | **0.801** | **0.692** | **0.790**  | **0.761** | **0.714** |
| CFGPT2-20B         | 20B  | **0.891** | **0.501** | **0.696** | **0.722** | **0.825** | **0.865** | **0.806** | **0.825** | **0.727** | **0.823**  | **0.792** | **0.755** |

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
| InternLM2-7B       | 7B   | 70.2      | 39.9      | 73.4        | 62.8           | 61.4     | 39.5       | 57.8    |
| InternLM2-20B      | 20B  | 76.4      | 52.6      | 76.3        | 66.2           | 63.9     | 42.1       | 62.9    |
| CFGPT2-7B          | 7B   | 81.9      | 62.8      | 75.2        | 71.3           | 64.1     | 68.2       | 70.5    |
| CFGPT2-20B         | 20B  | 84.6      | 66.5      | 78.1        | 75.9           | 66.0     | **71.9**   | 73.8    |

# Acknowledgements

CFGPT has referred to the following open-source projects. We want to express our gratitude and respect to the researchers of the projects.

- InternLM: https://github.com/InternLM/InternLM
- Firefly: https://github.com/yangjianxin1/Firefly
- FinGPT: https://github.com/AI4Finance-Foundation/FinGPT

# To-Do List

- [ ] Series of applications with CFGPT.
- [ ] Constructing more comprehensive training tasks and their corresponding databases.
- [ ] Continued improvement of the capabilities of CFGPT in more complex financial tasks.

# License

The use of the source code of CFGPT complies with the Apache 2.0 License.  CFGPT model also supports commercial use under the base model Licenses of InternLM, Baichuan-13B Model and the Terms of Use of the data generated by OpenAI. Please contact us if you find any potential violations. 

### Thanks To Our Contributors :
<a href="https://github.com/TongjiFinLab/CFGPT/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TongjiFinLab/CFGPT" />
</a>

### Citation

If you find **CFGPT** is useful for your research, please consider citing the following papers.

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