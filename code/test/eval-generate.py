import os
from transformers import AutoModel, AutoTokenizer
import torch
from peft import PeftModel
from datasets import load_from_disk
import json

base_model = 'TongjiFinLab/CFGPT1-pt-7B'
lora_weights = 'TongjiFinLab/CFGPT1-sft-7B-LoRA'
device_map = 'cuda:0'
new_tokens_limit = 256
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = AutoModel.from_pretrained(
        base_model,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=torch.bfloat16
    )
model = PeftModel.from_pretrained(model, lora_weights, device_map=device_map)
model = model.eval()

data = open('data/summary/data.jsonl', 'r').readlines()
for line in data:
    line = json.loads(line)
    print(line)

for line in data:
    line = json.loads(line)
    input_text = line['instruction'] + '\n' + line['input'] + '\n回答：'
    output_text = line['output']

    with torch.no_grad():
        out = model.generate(
            **tokenizer(input_text, return_tensors='pt').to(device_map),
            max_new_tokens=new_tokens_limit,
            do_sample=True,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.1
        )
        out_text = tokenizer.decode(out.cpu()[0], skip_special_tokens=True)

    print(input_text)
    print(output_text)
    print(out_text.split('回答：')[1], '\n')