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

In this repository, we will share the following models and code.

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
- [Data](#data)
- [Code](#code)
- [Cases](#cases)
- [Benchmark](#benchmark)
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

|Pretrain model|Adapter model|Full SFT Model|
|:-:|:-:|:-:|
 [CFGPT1-pt-7B](https://huggingface.co/TongjiFinLab/CFGPT1-pt-7B)|[CFGPT1-sft-7B-lora](https://huggingface.co/TongjiFinLab/CFGPT1-sft-7B-LoRA)|[CFGPT1-sft-7B-full](https://huggingface.co/TongjiFinLab/CFGPT1-sft-7B-Full)|

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
| Task | Task Description | Dataset | Size |
| - | - | - | - |
| Sentiment | Identify the sentiment associated with financial document | CFData-SA | 13K |
| Summary | Generate a content summary based on the provided financial document | CFData-RS | 18K |
| Risk | Generate risk alerts based on the provided financial document | CFData-RS | 20K |
| Suggestion | Generate investment recommendations based on the provided financial document | CFData-RS | 18K |
| Event | Identify the event categories associated with financial document | CFData-ED | 12K |
| Industry | Identify the industry categories associated with financial document | CFData-ED | 14K |
| Company | Identify the company names associated with financial document | CFData-ED | 12K |
| Product | Identify the product names associated with financial document | CFData-ED | 21K |
| Exam | Answer true-false questions related to finance question | CFData-QA | 16K |
| Stock | Predict stocks future movement |  CFData-SP | 15K |

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


# Cases

- **Sentiment:**
    - **Question:** 
        >你是一名金融从业者，请对这篇新闻进行情感分析。请从（中性、积极、消极）中选取答案。新闻内容：挖贝快讯：特步国际发布2023年第二季度中国内地业务营运状况，披露截至2023年6月30日止3个月零售销售实现高双位数同比增长(包括线上线下渠道)，零售折扣水平约七五折。同时，2022年7月MSCI首次予以特步ESG评级，一年后评级表现即迎来提升。明晟MSCI上调特步ESG评级，由“BB”升至“BBB”。
    - **Answer:** 
        >积极

- **Summary:**
    - **Question:** 
        >作为一名金融投资分析专家，你需要分析并总结来自不同券商的研究报告。请你突出该研报的创新点和市场洞见。研报内容：事件：公司4月26日发布2022年年度报告及2023年一季报，2022年实现营收10.71亿元，同比增长21.89%；归母净利润5.26亿元，同比增长19.95%；扣非净利润5.05亿元，同比增长16.32%。2023年Q1实现营收2.00亿元，同比下降22.90%，环比下降31.59%；归母净利润0.75亿元，同比下降38.88%，环比下降48.40%；扣非净利润0.71亿元，同比下降41.52%，环比下降47.81%。
        2022年，公司积极优化产品结构，持续加强在新能源和工业领域的测试布局，部分抵消了消费类芯片需求下滑的影响，助力全年业绩稳步增长。2023年Q1受行业波动及产品结构变化影响，主营业务毛利有所下降；同时公司增加了市场开拓和研发投入，致使净利润短期承压。公司2022年毛利率为76.88%，同比下降3.34pct；净利率为49.16%，同比下降0.80pct。2023年Q1毛利率为69.68%，同比下降10.35pct，环比下降6.57pct；净利率为37.33%，同比下降9.76pct，环比下降12.17pct。费用方面，Q1销售、管理、研发、财务费用率分别为14.28%/7.15%/16.58%/-3.49%，同比变动分别为5.73/0.88/6.40/-1.23pct。
        公司为国内模拟和混合测试领域的主力测试平台供应商，同时也在分立器件和功率类器件测试领域取得良好进展。公司STS8300机型主要应用于更高引脚数、更高性能、更多工位的电源管理类和混合信号集成电路测试，产品的平台化设计使其具备良好的可扩充性和兼容性，可以更好适应被测试芯片的更新和迭代。2022年，STS8300平台的出货量保持较高增速，装机量稳定增加，客户生态圈构建进展顺利，内部资源板卡加速迭代，应用范围不断拓展。公司正在研发和储备新一代平台型测试设备，该设备将拥有更快的测试频率和更高的测试效率，可覆盖的测试范围更大。随着STS8300市场份额的持续提升，新品研发稳步推进，有望打开业绩成长天花板。
        公司在不断夯实模拟和数模混合领域的优势的同时，对第三代半导体测试领域、功率模块测试以及SoC等新兴领域进行了前瞻布局。当前公司在氮化镓测试领域已取得领先优势，2022年在国家双碳政策推动下，新能源汽车、光伏产业呈现较高的景气度，公司大功率IGBT和碳化硅测试领域开始逐渐放量。客户拓展方面，公司已进入国际封测市场供应商体系，在中国台湾、东南亚、日本、韩国、欧洲、美国、南非和北非等国家或地区均有装机；公司对国内的设计公司和IDM企业保持全面覆盖，确保在长期竞争中保持领先地位，同时持续拓展海外客户，如意法半导体、安森美、安世半导体等均已成为公司客户。
        首次覆盖，给予“增持”评级：公司主营业务为半导体自动化测试系统的研发，生产和销售，主要产品为半导体自动化测试系统及测试系统配件。公司拥有模拟、数模混合、分立器件以及功率模块等测试领域的诸多核心技术，包括V/I源、精密电压电流测量、宽禁带半导体测试和智能功率模块测试等，同时密切跟踪半导体行业的发展方向，不断为客户推出功能更全、精度更高、速度更快的测试设备。我们看好公司在第三代半导体、功率、SoC等新兴领域的先发优势，随着新品的持续研发和推出，以及海内外客户的持续拓展，公司业绩有望实现长期稳定增长。预计公司2023-2025年归母净利润分别为6.13亿元、8.16亿元、10.56亿元，EPS分别为6.73、8.96、11.59元，PE分别为39X、29X、23X，给予“增持”评级。
    - **Answer:** 
        >高基数背景下, Q1盈利能力略降｜新产品有序推进, STS8300销量再创佳绩｜聚焦细分赛道, 拓宽成长空间｜订单释放叠加下游修复, 公司业绩显著回暖｜高阶机型的出货带动收入水平上升, 新产品逐步成熟带来利润弹性｜乘风新能源赛道东风, 多元化的战略布局迎来收获

- **Risk:**
    - **Question:** 
        >你是一位金融投资专家，请分析这份研报中可能存在的投资风险，包括市场变动、政策调整等方面的风险。
        研报内容： 
        　　公司公告称拟以自有资金预计回购公司股份不超 450 万股，占公司目前已发行总股本 2.01%，回购金额在 7000 万元至 13500 万元之间，回购价格不超 30 元/股，回购股份用于股权激励，回购期限自董事会审议通过回购股份方案之日起不超过 6 个月。
    　　    点评：
        　　本次股份回购体现公司对未来发展前景的信心及对公司价值的认可， 有利于稳定投资者对公司股票长期价值的预期， 回购股份用于股权激励可充分调动公司董事、高管、中层管理人员及核心骨干的积极性，将股东利益、公司利益和核心团队利益结合在一起。
        　　产品端： 衣柜、木门贡献新业绩增量， 橱衣木协同效应逐渐显现。1） 不断开拓新产品，夯实橱柜基础。 公司位居橱柜行业第二， 橱柜业务已形成六大产品系列，截止 2018 年底橱柜业专卖店 1487 家； 2）发力全屋定制，衣柜业务高速增长。 2018 年 8 月公司由整体厨房、定制衣柜转向全屋定制，截止 2018 年底定制衣柜专卖店 726 家， IK全屋定专卖店 33 家。 3）孵化木门业务，贡献业绩增量。通过切入木门市场可以实现橱衣木联动，拓展全屋定制家居品类， 截止 2018 年底公司木门经销商 117 家，与其他品类融合的门店 88 家。
        　　渠道端，零售门店持续拓展，大宗客户结构优化， 出口业务快速发展。 公司计划 2019 年新开橱柜门店 150 家、 木门店 100 家、 定制衣柜和全屋定制门店 350 家。大宗业务方面，优化大宗客户结构，大宗客户多为百强头部地产、主板上市公司和地区龙头地产公司， 其中百强地产客户占比已达 30%。 2018 年公司出口收入同比+ 87.26%， ， 创新澳洲 IJF 股权合作模式， 成立美洲市场项目组， 探索全球化发展路径。
        　　我们预计公司 2019-2020 年 EPS 分别为 1.41、 1.61 元，对应 2019-2020 年 PE 分别为 13.56、 11.87 倍， 考虑公司全品类、多品牌布局孕育新营收增长点， 维持“买入”评级。
    - **Answer:** 
        >房地产调控导致需求减弱的风险, 原材料成本波动的风险等。

- **Suggestion:**
    - **Question:** 
        >你是一位资深金融投资者，请根据研报中的投资建议，给出适合该公司的投资评级。研报内容：
        投资要点：
        中国电影是覆盖全产业链的电影龙头公司，实控人是中影集团
        中国电影是国内唯一一家集电影制作、发行、放映和影视服务全产业链覆盖的上市公司， 由中影集团及中国国际电视、央广传媒、长影集团、江苏广电、歌华有线、电广传媒、中国联通 7 家公司共同发起设立。 目前中影集团是公司的控股股东和实际控制人， 拥有公司 67.36%的股份； 截止 2018 年三季报公司前十大股东持股比例为 76.41%， 持股集中度相对较高。
        公司发行和放映业务的实力强， 全产业链竞争力突出
        公司的经营业务全面覆盖电影产业链的上中下游， 分别包括上游的影视制片制作业务、中游的电影发行营销业务以及下游的电影放映、影视服务业务。其中在影视发行营销业务方面，公司是国内两家进口片发行商之一， 具有极高的政策壁垒和竞争力；在电影放映业务方面，公司控股的中影数字、中影星美、中影南方新干线三家院线是国内院线领域的 TOP10， 综合竞争力突出；电影服务业务中，公司拥有中国巨幕的研发、生产和销售能力，此外还进行影院放映设备的销售，受益于近年来国内新建影院的快速增长，公司的中国巨幕系统发展迅速。因此从布局上分析， 中国电影无论是在单项业务还是整体业务实力都在国内处于绝对龙头的地位，竞争力和经营壁垒十分明显。
        电影行业： 2019 年上映大片阵容豪华，驱动市场发展
        2018 年 1-11 月国内电影总票房收入 563.12 亿元， 同比增长 10.96%， 整体表现基本符合预期。我们预计 2019 年全年票房继续保持平稳增长， 进口片的相对竞争优势更强，逻辑在于多部美国科幻电影陆续上映将为国内电影票房市场形成有力的支撑；同时 2019 年春节档的众多国产贺岁电影（如流浪地球等）也具备很强的竞争力，因此整体上我们认为 2019 年国内电影票房市场将继续保持平稳增速水平，增速相较于 2018 年将有所提高， 院线和相关爆款影片的制片/发行方将更加受益。
    - **Answer:** 
        >强烈推荐！
        鉴于公司在发行、放映领域均有较强实力, 且对上下游产业均有一定的话语权, 未来有望持续享受高毛利水平的优质内容产出带来的红利。其强大的出品能力和丰富的资源积累为其提供了良好的创作基础, 并凭借优秀的渠道把控力获得更多优质内容的供应。在此背景下, 考虑到业绩弹性空间大（如《流浪地球》等）等因素, 我坚信该公司未来的前景值得期待。当然, 投资的潜在风险仍需关注监管政策和疫情情况的变化。以上仅是基于当前信息的推理预测, 投资有风险, 决策需谨慎。

- **Event:**
    - **Question:** 
        >现在你会被提供一份中国的新闻：给定新闻的一级分类，请在我给定的二级标题中进行选择。一级分类为：金融市场。可选的二级分类为（金融市场, 衍生品市场, 期货市场, 贵金属, 股票市场, A股, 港股, 房地产, 债券市场, 新三板交易, 海外股票市场, 外汇市场, 人民币汇率, 美元汇率, 国际金融市场, 欧元汇率, 银行贷款, 股权投资, 日元汇率, ETF, 其他汇率, 保险市场, 虚拟货币, 期权市场, 资产证券化市场, 另类市场）。新闻内容：港资8月17日，净买入58.86万股，约335万元，增持比例：0.11%，成交占比：7.591%，持股比例：0.478%，连续8个交易日增持，近5个交易日比例变化：118.91%，近22个交易日比例变化：383.418%，近66个交易日比例变化：292.207%。
    - **Answer:** 
        >金融市场, 股票市场, A股

- **Industry:**
    - **Question:** 
        >你是一名金融行业的专家，给定新闻的一级行业，请在我给定的行业二级标题中进行选择。一级分类为：电气设备。可选的二级分类为（电源设备, 高低压设备, 电气自动化设备, 电机）。新闻内容：    12月31日，资本邦获悉，港股上市公司福莱特玻璃(06865.HK)于12月30日发布公告称，福莱特玻璃及其全资附属公司(下文简称“卖方)已与晶科集团及其全资附属公司(下文简称“买方”)关于销售光伏压延玻璃事项签订《战略合作协议》。根据协议，买方在2021~2023年三年内向卖方采购共计59GW(约3.38亿平方米)组件用光伏压延玻璃，合同履行期限自2021年1月1日起至2023年12月31日止。按照卓创周报2020年12月24日公布的光伏玻璃均价42元/平方(含税)测算，预估合同总金额约141.96亿元人民币(含税)。
        福莱特玻璃认为，此次协议的签订有利于公司光伏压延玻璃产品的市场推广以及进一步提升公司经营业绩。
        据悉，福莱特玻璃是一家集玻璃研发、制造、加工和销售为一体的综合性中大型企业，创建于1998年6月，集团本部位于长三角经济中心浙江嘉兴市。福莱特玻璃的主要产品涉及太阳能光伏玻璃、优质浮法玻璃、工程玻璃、家居玻璃四大领域，以及太阳能光伏电站的建设和石英岩矿开采，形成了比较完整的产业链。
    - **Answer:** 
        >电源设备, 高低压设备


- **Company:**
    - **Question:** 
        >你是一名具有丰富投顾经验的金融研究员，现在需要你与金融市场相关的新闻内容进行分析。需求：你能帮我总结出该新闻中涉及的公司名称吗？新闻内容：    洋河发布核心骨干持股计划，授予对象主要包括公司高管，核心中层及业务骨干，考核目标为21,22年营业收入相比于上一年营业收入增速超过15%。我们认为在次高端行业消费升级以及公司去库存推动渠道利润提升的基础下，公司实现任务目标的概率较高。略调整21-23年EPS为5.57、6.76、8.23.维持22年37x，一年目标价250元，维持“强烈推荐-A”评级。推荐阅读我们的深度报告《洋河深度：改革之路》
        洋河股份发布持股计划，业绩确定性进一步提升。洋河股份发布第一期核心骨干持股计划，授予对象主要包括公司高管以及公司及全资子公司的核心中层及业务骨干。持股计划对应21,22年业绩考核指标为营业收入相比于上一年营业收入增速超过15%。本轮股权激励总金额为10.03亿元，对应股本9,661,310股，占公司总股本的0.64%，对应当前股票市值18.9亿元。授予价格为公司之前回购成本,即103.73元/股。
        低库存和渠道利润支撑下，我们判断公司有望顺利完成目标。今年以来洋河持续去库存，目前渠道库存处于历史低位，上半年终端动销增速超过20%。我们认为在次高端白酒行业消费持续升级以及公司去库存推动渠道价值链不断提升的基础下，公司顺利实现任务目标的概率较高。
        本轮股权激励覆盖面广，提升整体的员工积极性。本轮改革之前，洋河的股权激励对象主要集中在之前的管理层，现任管理层缺乏股权激励或体量相对较小。同时本轮股权激励计划中，核心骨干占比91.72%，覆盖了公司核心中层及业务骨干。长期以来洋河依托于渠道的强控制力以及在品牌，产品上的不断创新，持续提升公司的综合竞争力。我们认为本轮股权激励方案有助于强化公司的渠道控制力以及提升公司业务创新活力，持续提升公司的长期竞争力。
        公司产品改革持续推进，产品动销持续加速。我们前期调研显示，公司省内市场老版天之蓝库存已经基本消化干净，省外部分市场天之蓝仍有少量库存，为产品换代已经做好了准备。梦之蓝水晶版铺货持续推进，省内经销商预计今年中秋国庆期间将迎来放量，目前整体的库存水平相比于年初已经有明显下降，整体处于合理区间内。草根调研显示，M6+批价持续维持在620元以上，渠道利润维持在12%以上，经销商销售积极性高，渠道库存紧张。
        投资建议：洋河高价位段次高端的领先布局与渠道去库存后持续改善将拉动公司的成长，维持“强烈推荐-A”评级。洋河在600-800元价格带布局领先大部分竞争对手，梦6+体量在行业内具备显著优势，未来将率先抢占该价格带的战略发展机会。目前来看公司省内外渠道梳理较为顺畅，渠道利润得到显著恢复，公司2021年将完成水晶梦的换代以及新版天之蓝的推出，有望在2022年进一步助推公司的成长。略调整21-23年EPS为5.57、6.76、8.23.维持22年37x，一年目标价250元，维持“强烈推荐-A”评级。
        风险提示：行业需求回落，省内竞争加剧，省外拓展不及预期。

    - **Answer:** 
        >洋河股份

- **Product:**
    - **Question:** 
        >你是一名关注金融市场的投资者，请分析这篇新闻。麻烦您找出该新闻中涉及到的相关产品。新闻内容：    方正中期研究员 史家亮【行情复盘】虽然有色板块偏弱，然沪锡在现货紧缺因素推动下逆势上涨，表现一枝独秀，继续大涨，09合约尾盘涨2.71%至23.67万元，结算价为23.29万元。当前，锡现货端缺货依旧，当前上期所仓单甚至难以覆盖08合约待交割量，并且云锡已经复产了也没能缓解现货紧缺的现状，所以强供需支撑下锡价强势行情依旧。伦锡方面，截止17：00，LME三个月期锡现报35165美元/吨，创历史新高，整体强势行情仍在持续，对沪锡的有效支撑依然存在。综合考虑汇率、增值税率和港杂费等因素（根据了解，暂时无8%的关税），对锡价内外盘价格进行测算，3.45万的伦锡对应的到岸价格为25.34万元，3.5万的伦锡对应的到岸价格为26.44万元；即使加入运费的考量，伦锡价格依然高于沪锡价格。
        重要资讯】基本面方面依然利好锡价：库存方面，上期所指定交割仓库仓单继续下跌56吨至1514吨，整体去库趋势依旧，并且现货端依然偏紧，消费旺季去库趋势仍将会持续；伦锡库存方面，伦锡库存维持在2245吨左右，当前库存依然处于低位。锡锭产量方面，受云锡冶炼系统升级改造以及华锡受当地限电政策影响，内蒙、云南地区部分企业仍因原料短缺及加工费低迷问题维持低量生产，7月国内精锡总产量下降明显，后市进口和国内原材料的供应依然会偏紧。受锡锭出口下降影响，锡锭表观消费量低位回升；并且6月锡精矿进口出现大幅回升亦使得锡精矿表观消费量出现大幅回升，收复5月跌幅；然而即使如此，国内锡现货市场货源依然紧张，锡锭采购难度较大，说明锡锭的需求亦是强劲，这均支撑了锡价。云锡已经复产，但是依然没有缓解锡现货的紧缺程度。
        【交易策略】基本面和外盘表现继续支撑着内盘锡价，沪锡去库趋势持续，故沪锡供需基本面依然较强，特别是现货端紧缺依旧，锡的整体强势行情仍在持续。当前进入高位震荡行情，但是依然不建议做空，特别是伦锡依然强于沪锡和基本面没有发生变化背景下，依然维持逢低做多思路，上方仍有刷新历史新高的可能（涨至24万上方后，上探25万高位的可能性仍存），短期内下方支撑位为22.6万的和22.3万的前低位置；回调则是做多机会。
    - **Answer:** 
        >锡, 沪锡, 伦锡, 云锡

- **Exam:**
    - **Question:** 
        >命题：按照单位业务数量制定一个固定不变的资费比率，根据用户使用的通信业务数量的多少收取通信费用的一种资费模式是定额资费。 分析解释：本题考查线性资费。线性从量资费指的是按照单位业务数量制定一个固定不变的资费比率，根据用户使用的通信业务数量的多少收取通信费用的一种资费模式。根据以上分析解释，你认为上述命题是否正确？
    - **Answer:** 
        >错误。

- **Stock:**
    - **Question:** 
        >你是一个量化交易研究员。请你根据提供的背景信息，推测贵州茅台未来五天的股价变化趋势。当前贵州茅台的前十天的历史收盘价为[1802.59, 1791.0, 1788.0, 1774.0, 1816.3, 1824.98, 1834.97, 1851.33, 1856.0, 1847.0,]。相关热点新闻信息：近期贵州茅台与瑞幸咖啡跨界即将推出的茅台拿铁迅速出圈，预计销量突破单日500万。相关社交媒体热点评论：瑞幸x茅台「酱香拿铁」爆红；8月31日消息，想必很多80/90后都听过《美酒加咖啡》这首老歌，是的，它真的要来了。近日，网传茅台将与瑞幸将推出联名款咖啡，目前杯套、纸袋的剧透图已流出，曝光的杯套和纸袋上都有非常明显的茅台风格，上面还标注着：“中国人的第一杯酱香拿铁”和“美酒加咖啡，就爱这一杯”；据财联社报道，有接近茅台的人士透露，此次联名咖啡是短期性的，而且有限量。据此可以猜测，这款拿铁怕是又要受到市场热炒了。 
    - **Answer:** 
        >根据历史股价、热点新闻和社媒评论，贵州茅台最近与瑞幸咖啡合作推出茅台拿铁，这将导致销量急剧增长以及企业知名度进一步提升，并有助于挖掘新的客户群体，增强产品粘性。这种积极的市场反应可能对贵州茅台的股价产生正面影响。然而，股价的走势还会受到其他因素的影响，如整体市场情况、公司财务表现、行业竞争等。
        整体来看，未来五天贵州茅台的股价可能会出现1%左右的增幅。

# Benchmark

We release the [CFBenchmark-Basic](https://github.com/TongjiFinLab/CFBenchmark), a chinese finance benchmark, including 3917 financial texts spanning three aspects and eight tasks, for the evaluation of the financial text processing capability of LLMs in Chinese financial market.

CFBenchmark-Basic utilize two types of metrics to evaluate the performance of LLMs in the financial domain on our CFBenchmark-Basic. 
For recognition and classification tasks, we employe the **F1 score** as the evaluation metric, which balance the precision and recall.  
For the generation tasks, we utilize **cosine similarity** between the vectored representation of ground truth and generated answer as measure the generation ability.  Since there are usually different expressions with the similar meaning in our generation tasks, simply employing Rough-Score or BULE-socre is not reasonable. Specifically, the **bge-zh-v1.5** is assigned as the oracle model to generate the sentence embedding. We calculate evaluation scores for each sub-task individually and provide the average score for each category.

The best scores of LLMs (considering zero-shot and few-shot),as well as which of our model,  are demonstrated below:

| Model              | Size | Company | Product | R.Avg   | Industry | Event   | Sentiment | C.Avg   | Summary | Risk    | Suggestion | G.Avg   | Avg     |
| ------------------ | ---- | ------- | ------- | -----   | -------- | -----   | --------- | -----   | ------- | -----   | ---------- | -----   | -----   |
| ChatGPT            | 20B  | 0.797   | 0.198   | 0.498   | 0.453    | 0.458   | 0.425     | 0.455   | 0.593   | 0.541   | 0.771      | 0.635   | 0.529   |
| ERNIE-Bot          | 260B | 0.807   | 0.300   | 0.533   | 0.408    | 0.350   | 0.186     | 0.315   | 0.715   | 0.590   | 0.716      | 0.673   | 0.507   |
| ERNIE-Bot-4        | -    | 0.819   | 0.417   | 0.618   | 0.418    | 0.358   | 0.375     | 0.384   | 0.721   | 0.629   | 0.718      | 0.689   | 0.564   |
| Falcon-7B          | 7B   | 0.671   | 0.168   | 0.420   | 0.169    | 0.132   | 0.250     | 0.184   | 0.302   | 0.301   | 0.246      | 0.283   | 0.296   |
| Falcon-7B-chat     | 7B   | 0.582   | 0.046   | 0.314   | 0.112    | 0.142   | 0.153     | 0.135   | 0.307   | 0.299   | 0.258      | 0.288   | 0.246   |
| bloomz-7B1         | 7B   | 0.765   | 0.166   | 0.465   | 0.252    | 0.154   | 0.394     | 0.267   | 0.451   | 0.371   | 0.462      | 0.428   | 0.387   |
| bloomz-7Bt1-mt     | 7B   | 0.751   | 0.157   | 0.454   | 0.087    | 0.182   | 0.380     | 0.216   | 0.425   | 0.379   | 0.396      | 0.400   | 0.357   |
| Qwen-7B            | 7B   | 0.780   | 0.357   | 0.569   | 0.480    | 0.335   | 0.379     | 0.398   | 0.750   | 0.505   | 0.713      | 0.656   | 0.541   |
| Qwen-Chat-7B       | 7B   | 0.763   | 0.360   | 0.562   | 0.400    | 0.367   | 0.265     | 0.344   | 0.548   | 0.307   | 0.379      | 0.411   | 0.439   |
| Qwen-14B           | 14B  | 0.805   | 0.421   | 0.613   | 0.481    | 0.350   | 0.385     | 0.405   | 0.754   | 0.608   | 0.717      | 0.693   | 0.570   |
| Qwen-Chat-14B      | 14B  | 0.814   | 0.442   | 0.628   | 0.382    | 0.400   | 0.350     | 0.377   | 0.732   | 0.478   | 0.736      | 0.649   | 0.551   |
| ChatGLM2-6B        | 6B   | 0.747   | 0.313   | 0.530   | 0.285    | 0.300   | 0.357     | 0.314   | 0.657   | 0.454   | 0.671      | 0.594   | 0.479   |
| Baichuan2-7B-Base  | 7B   | 0.672   | 0.340   | 0.506   | 0.342    | 0.490   | 0.480     | 0.437   | 0.739   | 0.619   | 0.751      | 0.703   | 0.549   |
| Baichuan2-7B-Chat  | 7B   | 0.757   | 0.402   | 0.579   | 0.425    | 0.475   | 0.323     | 0.408   | 0.725   | 0.648   | 0.732      | 0.702   | 0.563   |
| Baichuan2-13B-Base | 13B  | 0.781   | 0.330   | 0.555   | 0.436    | 0.496   | 0.477     | 0.470   | 0.725   | 0.503   | 0.747      | 0.658   | 0.561   |
| Baichuan2-13B-Chat | 13B  | 0.797   | 0.314   | 0.556   | 0.472    | 0.507   | 0.387     | 0.455   | 0.739   | 0.634   | 0.746      | 0.706   | 0.572   |
| InternLM-7B        | 7B   | 0.612   | 0.233   | 0.423   | 0.266    | 0.311   | 0.328     | 0.302   | 0.378   | 0.336   | 0.379      | 0.364   | 0.363   |
| InternLM-7B-Chat   | 7B   | 0.632   | 0.261   | 0.447   | 0.272    | 0.364   | 0.399     | 0.345   | 0.363   | 0.270   | 0.353      | 0.329   | 0.374   |
| InternLM-20B       | 20B  | 0.809   | 0.358   | 0.583   | 0.500    | 0.427   | 0.417     | 0.448   | 0.706   | 0.653   | 0.728      | 0.695   | 0.575   |
| InternLM-20B-Chat  | 20B  | 0.488   | 0.362   | 0.425   | 0.323    | 0.327   | 0.370     | 0.340   | 0.706   | 0.578   | 0.762      | 0.662   | 0.476   |
| CFGPT1-stf-LoRA    | 7B   | 0.820   | 0.414   | 0.617   | 0.569    | 0.729   | 0.769     | 0.689   | 0.745   | 0.584   | 0.609      | 0.646   | 0.650   |
| CFGPT1-sft-Full    | 7B   |**0.836**|**0.476**|**0.656**|**0.700** |**0.808**|**0.829**  |**0.779**|**0.798**|**0.669**|**0.808**   |**0.758**|**0.731**|

More details can be found in [CFBenchmark-Basic](https://github.com/TongjiFinLab/CFBenchmark)

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
CFGPT is a research preview intended for non-commercial use only, subject to the model License of InternLM and the Terms of Use of the data generated by OpenAI. Please contact us if you find any potential violations. The code is released under the Apache License 2.0. 

# Citation
If you use the code or data of [**CFGPT**](https://arxiv.org/abs/2309.10654), please declare the reference with the following:

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