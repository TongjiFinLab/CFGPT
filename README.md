<div style="text-align:center">
<!-- <img src="https://big-cheng.com/k2/k2.png" alt="k2-logo" width="200"/> -->
<h2>📈 CFBenchmark: Chinese Financial Assistant with Large Language Model</h2>
</div>

<div align="left">
<a href=''><img src='https://img.shields.io/badge/Paper-ArXiv-C71585'></a> 
<a href='https://huggingface.co/datasets/TongjiFinLab/CFBenchmark'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging Face-CFBenchmark-red'></a>  
<a href=''><img src='https://img.shields.io/badge/License-Apache--2.0-blue.svg'></a>  
</div>
 
English | [简体中文](README-CN.md)


# Introduction

Welcome to **CFBenchmark**

In recent years, with the rapid development of Large Language Models~(LLMs), outstanding performance has been achieved in various tasks by existing LLMs. However, we notice that there is currently a limited amount of benchmarks focused on assessing the performance of LLMs in specific domains. 

In this work, we introduce CFBenchmark, a Chinese financial assistant benchmark for large language models. The basic version of CFBenchmark includes 3917 financial texts spanning three aspects and eight tasks. The CFBenchmark is organized into three aspects, financial recognition, financial classification, and financial generation. We conduct experiments in zero-shot and few-shot mode on our CFBenchmark to evaluate renowned LLMs available in the market. The results have revealed that there is still significant room to improve LLMs in basic tasks of financial text processing.

In the future, we will continue to contribute more benchmark evaluations in this series.

<div align="center">
  <img src="imgs/Framework.png" width="100%"/>
  <br />
  <br /></div>


# Contents

- [CFBenchmark-Basic](#cfbenchmark-basic)
    - [QuickStart](#QuickStart)
    - [Performance of Existing LLMs](#performance-of-existing-llms)
- [Acknowledgements](#acknowledgements)
- [To-Do](#to-do)
- [License](#license)
- [Citation](#citation) 

# CFBenchmark-Basic

CFBenchmark-Basic includes 3917 financial texts spanning three aspects and eight tasks, organized from three aspects, financial recognition, financial classification, and financial generation.

- Recognition-Company: Recognize the company names associated with financial documents (273).
- Recognition-Product: Recognize the product names associated with financial documents (297).
- Classification-Sentiment: Classify the sentiment associated with financial documents (591).
- Classification-Event: Classify the event categories associated with financial documents (577).
- Classification-Industry: Classify the industry categories associated with financial documents (402).
- Generation-Suggestion: Generate investment suggestions based on the provided financial document (593).
- Generation-Risk: Generate risk alerts based on the provided financial document (591).
- Generation-Summary: Generate a content summary based on the provided financial document (593).

We provide two examples to reveal how the few-shot setting and zero-shot setting work during evaluation.

Example 1 Fewshot Input:
<div align="center">
  <img src="imgs/fewshot.png" width="100%"/>
  <br />
  <br /></div>

Example 2 Zeroshot Input：
<div align="center">
  <img src="imgs/zeroshot.png" width="100%"/>
  <br />
  <br /></div>

## QuickStart

### Installation

Below are the steps for quick installation.

 ```python
    conda create --name CFBenchmark python=3.10
    conda activate CFBenchmark
 ```

```python
    git clone https://github.com/TongjiFinLab/CFBenchmark
    cd CFBenchmark
    pip install -r requirements.txt
```



### Dataset Preparation

Download the dataset utilizing the Hugging Face dataset. Run the command **Manual download** and unzip it. Run the following command in the CFBenchmark project directory to prepare the data set in the CFBenchmark/CFBenchmark directory.

```text
wget https://huggingface.co/TongjiFinLab/CFBenchmark
unzip CFBenchmark.zip
```


### Evaluation

We have prepared the testing and evaluation codes for you in repo ```/codes```.  

To begin the evaluation, you can run the following code from the command line:
```cmd
cd CFBenchmark/codes
python -m run.py
```
You can enter ```codes/run.py``` to modify the parameters in it to make the code running path meet your requirements.
```py
from CFBenchmark import CFBenchmark
if __name__=='__main__':

    # EXPERIMENT SETUP
    modelname = 'YOUR-MODEL-NAME'
    model_type= 'NORMAL' #NORMAL or LoRA
    model_path= 'YOUR-MODEL-PATH'
    peft_model_path= ''#PASS YOUR OWN PATH OF PEFT MODEL IF NEEDED
    fewshot_text_path= '../fewshot'#DEFAULT PATH
    test_type='few-shot'#LET'S TAKE THE FEW-SHOT TEST AS AN EXAMPLE
    response_path='../cfbenchmark-response'#PATH TO RESERVE THE RESPONSE OF YOUR MODEL
    scores_path='../cfbenchmark-scores'	#PATH TO RESERVE THE SCORE OF YOUR MODEL
    embedding_model_path='../bge-zh-v1.5' #PASS YOUR OWN PATH OF BGE-ZH-V1.5
    benchmark_path='../cfbenchmark' #DEFAULT PATH
    data_source_type='offline'#online or offline

    #generate Class CFBenchmark
    cfb=CFBenchmark(
        model_name=modelname,
        model_type=model_type,
        model_path=model_path,
        peft_model_path=peft_model_path,
        fewshot_text_path=fewshot_text_path,
        test_type=test_type,
        response_path=response_path,
        scores_path=scores_path,
        embedding_model_path=embedding_model_path,
        benchmark_path=benchmark_path,
        data_source_type=data_source_type
    )
    
    cfb.generate_model()# TO GET RESPONSE FROM YOUR MODEL
    cfb.get_test_scores()# TO GET YOUR MODEL SCORES FROM RESPONSE
```

We defined a class ```CFBenchmark``` to do the evaluation. 

```Py
class CFBenchmark:
    def __init__(self,
                 model_name,
                 model_type,
                 model_path,
                 peft_model_path,
                 fewshot_text_path,
                 test_type,
                 response_path,
                 scores_path,
                 embedding_model_path,
                 benchmark_path,
                 data_source_type
                 ) -> None:
```

* You can use the arguments to set paths for models. If you want to use a LoRA fine-tuned model, set model_type`` toLoRAand pass your peft model path throughpeft_model_path```.
* You can set test-type to 'zero-shot' or 'few-shot' to do different evaluations.
* embedding_model_path is set for bzh-zh-v1.5 for calculating cosine-similarity. 
* You can modify the hyperparameters in CFBenchmark.generate_model() for text generations. 
* We provide CFBenchmark saved as a Dataset data type in both Hugging Face and Github. If you want to use an offline version of the benchmark, set the parameter data_source_type to offline````. If you want to use the online version of the benchmark, set the parameterdata_source_typetoonline```.




## Performance of Existing LLMs

We utilize two types of metrics to evaluate the performance of LLMs in the financial domain on our CFBenchmark. 

For recognition and classification tasks, we employ the **F1 score** as the evaluation metric, which balances precision and recall. 

For the generation tasks, we utilize **cosine similarity** between the vector representation of ground truth and the generated answer to measure the generation ability. 

Since there are usually different expressions with similar meanings in our generation tasks, simply employing Rough-Score or BULE-score is not reasonable. 

Specifically, the **bge-zh-v1.5** is assigned as the oracle model to generate the sentence embedding. We calculate evaluation scores for each sub-task individually and provide the average score for each category.

The best scores of LLMs(considering zero-shot and few-shot), as well as which of our model,  are demonstrated below:



| Model              | Size | Company   | Product   | R.Avg     | Industry  | Event     | Sentiment | C.Avg     | Summary   | Risk      | Suggestion | G.Avg     | Avg       |
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



# Acknowledgements

CFBenchmark has referred to the following open-source projects. We want to express our gratitude and respect to the researchers of the projects.

- tiiuae/falcon LLM series(https://huggingface.co/tiiuae/falcon-7b)
- bigscience/bloomz LLM series(https://huggingface.co/bigscience/bloomz-7b1)
- QwenLM/Qwen LLM series(https://github.com/QwenLM/Qwen)
- THUDM/ChatGLM2-6b(https://github.com/THUDM/ChatGLM2-6B)
- baichuan-inc/Baichuan2 LLM series(https://github.com/baichuan-inc/Baichuan2)
- InternLM/InternLM LLM series(https://github.com/InternLM/InternLM)
- ssymmetry/BBT-FinCUGE-Applications(https://github.com/ssymmetry/BBT-FinCUGE-Applications)
- chancefocus/PIXIU(https://github.com/chancefocus/PIXIU)
- SUFE-AIFLM-Lab/FinEval(https://github.com/SUFE-AIFLM-Lab/FinEval)


# To-Do
- [ ] CFBenchmark-Advanced:
    - In various scenarios of Chinese financial usage, propose more evaluation tasks to enrich the CFBenchmark series.

# License
CFBenchmark is a research preview intended for non-commercial use only, subject to the Terms of Use of the data generated by OpenAI. Please contact us if you find any potential violations. The code is released under the Apache License 2.0. 

# Citation

```bibtex
@misc{

}
```
