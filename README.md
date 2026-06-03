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

**CFGPT** is a family of large language models for Chinese financial scenarios. We perform continual pretraining on collected and cleaned Chinese financial corpora (CFData-pt), and then conduct supervised fine-tuning with knowledge-intensive financial instruction data (CFData-sft), enabling general-purpose LLMs to obtain stronger capabilities in financial text understanding, financial information extraction, financial generation, and risk analysis.

CFGPT currently includes three major versions:

| Version | Size | Base Model | Status |
| - | - | - | - |
| CFGPT1 | 7B | InternLM 7B | Part of the model weights have been open-sourced |
| CFGPT2 | 7B & 20B | InternLM2 7B & 20B | Related models and evaluation results have been released |
| CFGPT3 | 8B & 32B | Qwen3 8B & 32B | Training code has been added; model weights are not open-sourced yet |

- **CFGPT1** is based on InternLM 7B and includes a continual-pretrained model, a LoRA supervised fine-tuned model, and a full-parameter supervised fine-tuned model:
  - [CFGPT1-pt-7B](https://huggingface.co/TongjiFinLab/CFGPT1-pt-7B): model weights after continual pretraining on Chinese financial corpora.
  - [CFGPT1-sft-7B-LoRA](https://huggingface.co/TongjiFinLab/CFGPT1-sft-7B-LoRA): LoRA adapter weights trained based on CFGPT1-pt-7B.
  - [CFGPT1-sft-7B-Full](https://huggingface.co/TongjiFinLab/CFGPT1-sft-7B-Full): full-parameter SFT weights trained based on CFGPT1-pt-7B.
- **CFGPT2** is based on [InternLM2](https://github.com/InternLM/InternLM), covering both 7B and 20B scales. It integrates modules such as retrieval augmentation, fact checking, compliance checking, and risk monitoring, and achieves strong performance on multiple financial tasks.
  - [CFGPT2-7B](https://huggingface.co/TongjiFinLab/CFGPT2-7B)
  - [CFGPT2-20B](https://huggingface.co/TongjiFinLab/CFGPT2-20B)
- **CFGPT3** is based on [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) and [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B). It follows the training route of "financial continual pretraining + financial supervised fine-tuning" and expands the continual-pretraining data. The repository currently includes training and evaluation code for CFGPT3-8B; the model weights of CFGPT3-8B and CFGPT3-32B are not open-sourced yet.

We also released [CFBenchmark](https://github.com/TongjiFinLab/CFBenchmark), which is used to evaluate the text-processing capability of large language models in the Chinese financial market. This repository keeps CFGPT training code, evaluation code, CFData samples, and typical application cases, making it easier for researchers to reproduce the training pipeline and understand the data format.

***The following figure provides an overview of the CFGPT training pipeline:***

<div align="center">
<img align="center" src=./figs/CFGPT-TRAIN.svg width="100%"/>
</div>

# Contents

- [Quick Start](#quick-start)
- [Use Cases](#use-cases)
- [Data](#data)
- [Code](#code)
- [Evaluation](#evaluation)
- [Acknowledgements](#acknowledgements)
- [To-Do List](#to-do-list)
- [License](#license)
- [Citation](#citation)

# Quick Start

**1. Prepare the code and environment**

Clone the repository, create a Python environment, and activate it:

```bash
git clone https://github.com/TongjiFinLab/CFGPT.git
cd CFGPT
conda create -n env_name python=3.10
source activate env_name
pip install -r requirements.txt
```

**2. Select a model version**

The currently open-sourced models are mainly from the CFGPT1 series. CFGPT3-8B and CFGPT3-32B are currently at the stage where training and evaluation code is released, while model weights are not open-sourced yet.

| Model | Type | Link |
| - | - | - |
| CFGPT1-pt-7B | Continual-pretrained model | [Hugging Face](https://huggingface.co/TongjiFinLab/CFGPT1-pt-7B) |
| CFGPT1-sft-7B-LoRA | LoRA supervised fine-tuned model | [Hugging Face](https://huggingface.co/TongjiFinLab/CFGPT1-sft-7B-LoRA) |
| CFGPT1-sft-7B-Full | Full-parameter supervised fine-tuned model | [Hugging Face](https://huggingface.co/TongjiFinLab/CFGPT1-sft-7B-Full) |
| CFGPT2-7B | Supervised fine-tuned model | [Hugging Face](https://huggingface.co/TongjiFinLab/CFGPT2-7B) |
| CFGPT2-20B | Supervised fine-tuned model | [Hugging Face](https://huggingface.co/TongjiFinLab/CFGPT2-20B) |
| CFGPT3-8B / CFGPT3-32B | Qwen3-based financial models | Not open-sourced yet |

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

- **More usage details are available in `./code/test`**

**5. Train CFGPT3-8B**

The CFGPT3-8B code includes continual pretraining and full-parameter supervised fine-tuning. Before running it, please update model paths, data paths, cache paths, and GPU/DeepSpeed configurations according to your local environment.

```bash
# Data processing and training for continual pretraining
cd code/train/Qwen3-8B/pretrain
python process_data.py
bash start.sh

# Supervised fine-tuning
cd ../full-sft
bash start.sh
```

# Use Cases

- [CFGPT-v2-7B Banking Use Case](cases/case_bank.md)
- [CFGPT-v2-7B Algorithmic Trading Use Case](cases/case_algor_trading.md)
- [CFGPT-v2-7B Primary Industry Use Case](cases/case_primary_industry.md)
- [CFGPT-v1-7B Dataset Case](cases/case_CFGPTv1.md)

# Data

In this repository, we share samples of CFData:

- CFData: `./data`

    CFData includes a pretraining dataset (CFData-pt) and a supervised fine-tuning dataset (CFData-sft). The pretraining dataset contains Chinese financial data and analytics, together with a smaller subset of general-purpose text, totaling 584 million documents and 141 billion tokens. The supervised fine-tuning dataset is designed for six different financial tasks, covering multiple aspects of financial analysis and decision-making, with 1.5 million instruction pairs and 1.5 billion tokens in total.

## Continual Pretraining

### CPT for CFGPT1 & CFGPT2

The pretraining dataset contains 591 million documents and 193 billion tokens, including six sub-datasets:

* CFData-CP (6.24%): 39 thousand corporate prospectuses with 13 billion tokens;
* CFData-CA (12.28%): 6 million corporate announcements with 17 billion tokens;
* CFData-RR (2.51%): 392 thousand research reports with 3 billion tokens;
* CFData-FN (18.70%): 82 million financial news articles with 26 billion tokens;
* CFData-SM (60.15%): 495 million social media records with 84 billion tokens;
* CFData-Wiki (0.09%): 255 thousand Wikipedia entries with 137 million tokens.

We sampled a financial text sub-corpus from CFData-pt for further pretraining on InternLM-7B. This sub-corpus contains about 13.7 billion tokens from a large amount of Chinese financial data and analytics, along with a small amount of general-purpose text, including announcements, research reports, social media content, financial news articles, and Wikipedia. These data were mainly collected by ourselves.

### CPT for CFGPT3

For CFGPT3 training, we added a portion of new data. The details are as follows:

* Financial (70.17%)
    * news_rp_origin_text (21.73%)
    * fin_news_2022 (20.95%)
    * fin_announcement (13.61%)
    * fin_articles (6.91%)
    * fin_news_2025 (6.33%)
    * fin_books (0.36%)
    * fin_exam (0.26%)
    * fin_reports (0.01%)
* General (29.83%)
    * CLUECorpusSmall (18.76%)
    * UNv1-0 (3.15%)
    * ParaCrawl-v9 (1.86%)
    * zhwiki_2023 (1.84%)
    * news-crawl (1.64%)
    * csl (1.26%)
    * translation2019zh (0.83%)
    * WikiMatrix (0.43%)
    * news-commentary (0.05%)

The total amount is about 28.4 billion tokens. The continual pretraining of the CFGPT3 system models uses all of these data. These data mainly include financial data such as news, announcements, research reports, financial articles, professional books, financial exams, and financial reports. In addition, a small portion of general corpora is also included. All of the above data were collected by ourselves.

## Supervised Fine-Tuning

The supervised fine-tuning dataset contains 1.6 million instruction pairs and 1.5 billion tokens, including six financial tasks:

* CFData-SA (5.69%): 120 thousand instances with 86 million tokens for sentiment analysis;
* CFData-RS (50.60%): 369 thousand instances with 765 million tokens for report summarization;
* CFData-ED (22.69%): 490 thousand instances with 343 million tokens for event detection;
* CFData-TD (12.37%): 369 thousand instances with 187 million tokens for topic decomposition;
* CFData-QA (0.39%): 12 thousand instances with 6 million tokens for question answering;
* CFData-SP (8.27%): 212 thousand instances with 125 million tokens for stock movement prediction.

We use high-quality domain-specific data to achieve financial domain adaptation through supervised fine-tuning. The dataset includes six financial datasets that reflect different aspects of financial analysis and decision-making, including sentiment analysis, event detection, report summarization, topic decomposition, question answering, and stock movement prediction.

CFData-sft provides a large amount of financial-domain text information, allowing FinLLMs to learn from different sources.

Considering practical requirements, we reorganized these financial supervised fine-tuning datasets into ten tasks.

The details are as follows:

| Task | Task Description | Dataset | Size |
| - | - | - | - |
| Sentiment | Identify sentiment associated with financial documents | CFData-SA | 13K |
| Summary | Generate a content summary based on the provided financial document | CFData-RS | 18K |
| Risk | Generate risk alerts based on the provided financial document | CFData-RS | 20K |
| Suggestion | Generate investment recommendations based on the provided financial document | CFData-RS | 18K |
| Event | Identify event categories associated with financial documents | CFData-ED | 12K |
| Industry | Identify industry categories associated with financial documents | CFData-ED | 14K |
| Company | Identify company names associated with financial documents | CFData-ED | 12K |
| Product | Identify product names associated with financial documents | CFData-ED | 21K |
| Exam | Answer true-false questions related to finance | CFData-QA | 16K |
| Stock | Predict future stock movement | CFData-SP | 15K |

For more information about CFData, researchers can refer to the sample records in [CFData](./data).

# Code

This repository keeps the key training and evaluation code for CFGPT. Overall, the CFGPT training pipeline includes data preprocessing, continual pretraining, supervised fine-tuning, model generation testing, and downstream evaluation. Different versions correspond to different base models and script directories. For detailed parameters, please refer to the `.py`, `.sh`, `.yml`, and `ds_config.json` files in the corresponding directories.

## Code Structure

```text
code/
├── train/
│   ├── InternLM/
│   │   └── pretrain/          # Continual pretraining code for CFGPT1/2 InternLM-series models
│   └── Qwen3-8B/
│       ├── pretrain/          # Continual pretraining and data processing code for CFGPT3-8B
│       └── full-sft/          # Full-parameter supervised fine-tuning code for CFGPT3-8B
├── test/
│   └── eval-generate.py       # Generation test example
└── utils/                     # Common components for data processing, loss, and trainer
```

## Continual Pretraining

Continual pretraining is used to adapt general-purpose base models to Chinese financial corpora. This stage mainly includes tokenization, long-text packing, dataset saving, and distributed training.

InternLM-series training code is located at:

```bash
code/train/InternLM/pretrain
```

CFGPT3-8B training code is located at:

```bash
code/train/Qwen3-8B/pretrain
```

Taking CFGPT3-8B as an example, the running process is as follows:

```bash
cd code/train/Qwen3-8B/pretrain
python process_data.py
bash start.sh
```

Here, `process_data.py` processes raw jsonl text into 2048-token training blocks, `qwen3_8b_pt_train.py` performs continual pretraining, and `start.sh` configures environment variables and launches distributed training. Before running, please update model paths, data paths, cache paths, CUDA paths, and the number of GPUs according to your local environment.

## Supervised Fine-Tuning

Supervised fine-tuning further improves the model's performance on financial question answering, financial information extraction, report summarization, risk alerts, investment suggestions, and other instruction-following tasks.

The original CFGPT1 version includes both LoRA SFT and full-parameter SFT. Here we use lora-bf16 as an example. The training script is located under `./code/train/lora`:

```bash
deepspeed --include localhost:6,7 --master_port 60005 lora_bf_16_parallel_train.py --config lora_bf_16_parallel_train.yml > lora_bf_16_parallel_train.log 2>&1
```

CFGPT3-8B currently provides full-parameter SFT training code:

```bash
code/train/Qwen3-8B/full-sft
```

Running example:

```bash
cd code/train/Qwen3-8B/full-sft
bash start.sh
```

Here, `qwen3_8b_sft.py` loads the continual-pretrained model, constructs ChatML-format training text, and launches SFT training. Training data paths, evaluation data paths, DeepSpeed configuration paths, and output paths should be modified according to the actual machine environment.

## Generation Test

The generation test script is located at:

```bash
code/test/eval-generate.py
```

This script is used to quickly check the model's generation quality on financial tasks. Model loading paths and inference parameters can be modified in the script.

# Evaluation

The evaluation results of CFGPT2 are shown below.

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

CFGPT refers to the following open-source projects during development. We thank the researchers and developers of these projects.

- InternLM: https://github.com/InternLM/InternLM
- Qwen: https://github.com/QwenLM/Qwen
- Firefly: https://github.com/yangjianxin1/Firefly
- FinGPT: https://github.com/AI4Finance-Foundation/FinGPT

# To-Do List

- [ ] Build downstream applications with CFGPT, such as CFAPP.
- [ ] Construct more comprehensive training tasks and corresponding datasets.
- [ ] Continuously improve CFGPT's capabilities on more complex financial tasks.

# License

The code of CFGPT follows the Apache License 2.0. The use of open-sourced models should also comply with the licenses of the corresponding base models and the usage terms of the training data. CFGPT1/CFGPT2 models follow the license requirements of base models such as InternLM / InternLM2. CFGPT3 is trained based on Qwen3, and the model weights of CFGPT3-8B and CFGPT3-32B are not open-sourced yet. If you find any potential risk or violation, please contact us.

### Thanks To Our Contributors:

<a href="https://github.com/TongjiFinLab/CFGPT/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TongjiFinLab/CFGPT" />
</a>

# Citation

If you find **CFGPT** useful for your research, please cite the following papers:

```
@article{li2023cfgpt,
  title={CFGPT: Chinese financial assistant with large language model},
  author={Li, Jiangtong and Bian, Yuxuan and Wang, Guoxuan and Lei, Yang and Cheng, Dawei and Ding, Zhijun and Jiang, Changjun},
  journal={arXiv preprint arXiv:2309.10654},
  year={2023}
}

@article{li2024ra,
  title={RA-CFGPT: Chinese financial assistant with retrieval-augmented large language model},
  author={Li, Jiangtong and Lei, Yang and Bian, Yuxuan and Cheng, Dawei and Ding, Zhijun and Jiang, Changjun},
  journal={Frontiers of Computer Science},
  volume={18},
  number={5},
  pages={185350},
  year={2024},
  publisher={Springer}
}
```
