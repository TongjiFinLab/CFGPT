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

**CFGPT** 是面向中文金融场景的大语言模型系列。我们通过在收集、清洗后的中文金融语料（CFData-pt）上进行继续预训练，并结合知识密集型金融指令数据（CFData-sft）进行有监督微调，使通用大模型具备更强的金融文本理解、金融信息抽取、金融生成与风险分析能力。

CFGPT 目前包含三个主要版本：

| 版本 | 参数规模 | 基座模型 | 状态 |
| - | - | - | - |
| CFGPT1 | 7B | InternLM 7B | 已开源部分模型权重 |
| CFGPT2 | 7B & 20B | InternLM2 7B & 20B | 已发布相关模型与评测结果 |
| CFGPT3 | 8B & 32B | Qwen3 8B & 32B | 训练代码已加入，模型权重暂未开源 |

- **CFGPT1** 基于 InternLM 7B，包含继续预训练模型、LoRA 有监督微调模型和全参数有监督微调模型：
  - [CFGPT1-pt-7B](https://huggingface.co/TongjiFinLab/CFGPT1-pt-7B)：在中文金融语料上继续预训练后的模型权重。
  - [CFGPT1-sft-7B-LoRA](https://huggingface.co/TongjiFinLab/CFGPT1-sft-7B-LoRA)：基于 CFGPT1-pt-7B 训练的 LoRA 适配器权重。
  - [CFGPT1-sft-7B-Full](https://huggingface.co/TongjiFinLab/CFGPT1-sft-7B-Full)：基于 CFGPT1-pt-7B 训练的全参数 SFT 权重。
- **CFGPT2** 基于 [InternLM2](https://github.com/InternLM/InternLM)，覆盖 7B 和 20B 两个规模，并结合检索增强、事实核查、合规检查和风险监测等模块，在多个金融任务上取得较好表现。
  - [CFGPT2-7B](https://huggingface.co/TongjiFinLab/CFGPT2-7B)
  - [CFGPT2-20B](https://huggingface.co/TongjiFinLab/CFGPT2-20B)
- **CFGPT3** 基于 [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)和 [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B)，延续“金融继续预训练 + 金融有监督微调”的训练路线，并在继续预训练数据上进行了扩展。当前仓库已加入 CFGPT3-8B 的训练和评测代码；CFGPT3-8B 与 CFGPT3-32B 模型权重暂未开源。

我们同时发布了 [CFBenchmark](https://github.com/TongjiFinLab/CFBenchmark)，用于评估大语言模型在中文金融市场中的文本处理能力。本仓库保留 CFGPT 的训练代码、评测代码、CFData 样例数据和典型应用案例，方便研究者复现训练流程与理解数据格式。

***以下是训练 CFGPT 的流程概览图：***

<div align="center">
<img align="center" src=./figs/CFGPT-TRAIN_zh.svg width="100%"/>
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

**2. 选择模型版本**

当前已开源的模型主要是 CFGPT1 系列。CFGPT3-8B 和 CFGPT3-32B 当前处于训练与评测代码发布阶段，模型权重暂未开源。

| 模型 | 类型 | 链接 |
| - | - | - |
| CFGPT1-pt-7B | 继续预训练模型 | [Hugging Face](https://huggingface.co/TongjiFinLab/CFGPT1-pt-7B) |
| CFGPT1-sft-7B-LoRA | LoRA 有监督微调模型 | [Hugging Face](https://huggingface.co/TongjiFinLab/CFGPT1-sft-7B-LoRA) |
| CFGPT1-sft-7B-Full | 全参数有监督微调模型 | [Hugging Face](https://huggingface.co/TongjiFinLab/CFGPT1-sft-7B-Full) |
|CFGPT2-7B|有监督微调模型|[Hugging Face](https://huggingface.co/TongjiFinLab/CFGPT2-7B)|
|CFGPT2-20B|有监督微调模型|[Hugging Face](https://huggingface.co/TongjiFinLab/CFGPT2-20B)|
| CFGPT3-8B / CFGPT3-32B | Qwen3 系列金融模型 | 暂未开源 |

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

**5. 训练 CFGPT3-8B**

CFGPT3-8B 的代码包含继续预训练、全参数有监督微调。运行前需要根据本地环境修改模型路径、数据路径、缓存路径和 GPU/DeepSpeed 配置。

```bash
# 继续预训练数据处理与训练
cd code/train/Qwen3-8B/pretrain
python process_data.py
bash start.sh

# 有监督微调
cd ../full-sft
bash start.sh
```

# 典型使用案例

- [CFGPT-v2-7B 银行业场景使用案例](cases/case_bank.md)
- [CFGPT-v2-7B 算法交易场景使用案例](cases/case_algor_trading.md)
- [CFGPT-v2-7B 第一产业场景使用案例](cases/case_primary_industry.md)
- [CFGPT-v1-7B 典型数据案例介绍](cases/case_CFGPTv1.md)

# 数据集

在这个存储库中，我们分享了CFData的样本：
- CFData：`./data`

    CFData包括一个预训练数据集（CFData-pt）和一个监督微调数据集（CFData-sft），其中预训练数据集汇集了中国金融数据和分析，以及一个较小的通用文本子集，总共有5.84亿个文档和1410亿个token，监督微调数据集专为六种不同的金融任务量身定制，涵盖了金融分析和决策制定的各个方面，共有150万个指令对和150亿个token。


## 继续预训练
### CFGPT1&2的CPT
预训练数据集包括 5.91 亿份文档和 1930 亿个token，包括六个子数据集：

* CFData-CP（6.24%）：包括 3,900 份公司招股说明书，共计 130 亿个token；
* CFData-CA（12.28%）：包括 600 万份公司公告，共计 170 亿个token；
* CFData-RR（2.51%）：包括 39.2 万份研究报告，共计 30 亿个token；
* CFData-FN（18.70%）：包括 8,200 万份财经新闻，共计 260 亿个token；
* CFData-SM（60.15%）：包括 4.95 亿份社交媒体内容，共计 840 亿个token；
* CFData-Wiki（0.09%）：包括 25.5 万份维基百科内容，共计 1.37 亿个token。

我们从CFData-pt中抽取了一个财经文本子语料库，以便在InternLM-7B上进行进一步的预训练。该子语料库包含了来自大量中国财经数据和分析以及少量通用文本的共计约137亿个token，这些通用文本包括公告、研究报告、社交媒体内容、财经新闻文章和维基百科等，而这些数据主要由我们自行收集。

### CFGPT3的CPT
对于CFGPT3的训练，我们增加了一部分新的数据。具体是：
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

以上一共大约284亿tokens。本次对CFGPT3系统模型的继续预训练使用了全部的数据。这些数据主要一些金融数据例如包括新闻，公告，研究报告，金融文章，专业书籍，金融测试和财报等等。除此之外也包括小部分通用语料。以上数据皆由我们自行收集。


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

更多关于CFData的信息，研究人员可以参考我们[CFData](./data)的一些示例数据

# 代码

本仓库保留 CFGPT 的关键训练与评测代码。整体上，CFGPT 的训练流程包括：数据预处理、继续预训练、有监督微调、模型生成测试和下游评测。不同版本对应的基座模型和脚本目录略有不同，详细参数请查看对应目录下的 `.py`、`.sh`、`.yml` 和 `ds_config.json` 文件。

## 代码结构

```text
code/
├── train/
│   ├── InternLM/
│   │   └── pretrain/          # CFGPT1/2 相关 InternLM 系列继续预训练代码
│   └── Qwen3-8B/
│       ├── pretrain/          # CFGPT3-8B 继续预训练与数据处理代码
│       └── full-sft/          # CFGPT3-8B 全参数有监督微调代码
├── test/
│   └── eval-generate.py       # 生成测试示例
└── utils/                     # 数据整理、loss 与 trainer 等通用组件
```

## 继续预训练

继续预训练用于将通用基座模型适配到中文金融语料。该阶段主要包含分词、长文本拼接、数据保存和分布式训练。

InternLM 系列训练代码位于：

```bash
code/train/InternLM/pretrain
```

CFGPT3-8B 训练代码位于：

```bash
code/train/Qwen3-8B/pretrain
```

以 CFGPT3-8B 为例，运行流程如下：

```bash
cd code/train/Qwen3-8B/pretrain
python process_data.py
bash start.sh
```

其中，`process_data.py` 负责将原始 jsonl 文本处理为 2048 token 的训练块，`qwen3_8b_pt_train.py` 负责继续预训练，`start.sh` 负责配置环境变量并启动分布式训练。实际运行前请根据本地环境修改模型路径、数据路径、缓存路径、CUDA 路径和 GPU 数量。


## 有监督微调

有监督微调用于进一步提升模型在金融问答、金融信息抽取、报告摘要、风险提示、投资建议等指令任务上的表现。

CFGPT1 原始版本包含 LoRA SFT 和全参数 SFT 两类训练方式。在这里，我们以 lora-bf16 作为示例，训练脚本位于 ./code/train/lora 目录下
```
deepspeed --include localhost:6,7 --master_port 60005 lora_bf_16_parallel_train.py --config lora_bf_16_parallel_train.yml > lora_bf_16_parallel_train.log 2>&1
```
CFGPT3-8B 当前提供全参数 SFT 训练代码：

```bash
code/train/Qwen3-8B/full-sft
```

运行示例：

```bash
cd code/train/Qwen3-8B/full-sft
bash start.sh
```

其中，`qwen3_8b_sft.py` 负责加载继续预训练后的模型、构造 ChatML 格式训练文本并启动 SFT 训练。训练数据路径、评测数据路径、DeepSpeed 配置路径和输出路径需要根据实际机器环境修改。

## 生成测试

生成测试脚本位于：

```bash
code/test/eval-generate.py
```

该脚本用于快速检查模型在金融任务上的生成效果。不同模型的加载路径和推理参数可在脚本中修改。


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


# 致谢

CFGPT的研发过程参考了以下开源项目。我们向这些项目的研究者表示感谢。

- InternLM: https://github.com/InternLM/InternLM
- Qwen: https://github.com/QwenLM/Qwen
- Firefly: https://github.com/yangjianxin1/Firefly
- FinGPT: https://github.com/AI4Finance-Foundation/FinGPT


# 未来工作
- [ ] 使用CFGPT创建下游的应用CFAPP
- [ ] 构建更加全面的训练任务与对应数据
- [ ] 持续性改进CFGPT在更多复杂金融任务上的能力

# 使用许可
CFGPT的代码遵循Apache许可证2.0协议。已开源模型的使用许可需同时遵循对应基座模型许可证和训练数据使用条款。CFGPT1/CFGPT2 相关模型遵循 InternLM / InternLM2 等基础模型的许可要求；CFGPT3 基于 Qwen3 训练，当前 CFGPT3-8B 与 CFGPT3-32B 模型权重暂未开源。如您发现任何潜在的风险行为，请与我们联系。

### 感谢我们的贡献者 :
<a href="https://github.com/TongjiFinLab/CFGPT/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TongjiFinLab/CFGPT" />
</a>

# 引用
如果您认为**CFGPT**对您的研究有帮助，可以引用以下的论文：

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
