import os
from loguru import logger

from transformers import (
    AutoModel,
    AutoTokenizer,
    TrainingArguments,
)
from transformers import (
    Trainer,
    set_seed,
    HfArgumentParser,
    TrainingArguments,
)

import datasets
import torch
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
    prepare_model_for_int8_training,
)
import yaml
import argparse

import sys 
sys.path.append("../../") 
from utils.trainer import LoraTrainer
from utils.collator import DataCollator
from utils.loss import TargetLMLoss


def setup_everything():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="-1")
    parser.add_argument("--local_rank", type=int, default="0")
    args = parser.parse_args()
    args_dict = vars(args)
    with open(args_dict['config']) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        args["device_map"] = device_map
        logger.info('Parsing Args...')
        for k, v in args.items():
            print(f"{k}: {v}")

    set_seed(args['seed'])
    return args

def init_components(args):
    logger.info('Initializing components...')
    tokenizer = AutoTokenizer.from_pretrained(args['model_name'], trust_remote_code=True)

    model = AutoModel.from_pretrained(
            args['model_name'], 
            # load_in_8bit = True,
            trust_remote_code=True, 
            device_map=args['device_map']
        )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    model.config.use_cache = (False)

    logger.info(f'memory footprint of model: {model.get_memory_footprint()/(1024*1024*1024)} GB')

    '''Load Lora'''
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args['rank'],
        lora_alpha=args['lora_alpha'],
        lora_dropout=args['lora_dropout'],
        target_modules=args['target_modules'],
        bias=args['bias'],
    )
    model = get_peft_model(model, lora_config)
    logger.info(f"{model.print_trainable_parameters()}")

    '''Restart Training'''
    resume_from_checkpoint = args['resume_from_checkpoint']
    if resume_from_checkpoint is not None:
        checkpoint_name = os.path.join(resume_from_checkpoint, 'pytorch_model.bin')
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, 'adapter_model.bin'
            )
            resume_from_checkpoint = False
        if os.path.exists(checkpoint_name):
            logger.info(f'Restarting from {checkpoint_name}')
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            logger.info(f'Checkpoint {checkpoint_name} not found')

    '''Load Dataset'''
    dataset = datasets.load_from_disk(args['dataset'])
    dataset_eval = datasets.load_from_disk(args['dataset_eval'])
    logger.info(f"{args['dataset']}has loaded")
    logger.info(f"The Dataset Info:\n{dataset}")
    logger.info(f"{args['dataset_eval']}has loaded")
    logger.info(f"The Dataset-eval Info:\n{dataset}")

    training_args = TrainingArguments(
            output_dir=args['output_dir'],
            logging_steps = args['logging_steps'],
            eval_steps = args['eval_steps'], 
            num_train_epochs = args['num_train_epochs'],
            per_device_train_batch_size=args['per_device_train_batch_size'],
            per_device_eval_batch_size=args['per_device_eval_batch_size'],
            gradient_accumulation_steps=args['gradient_accumulation_steps'],
            learning_rate=args['learning_rate'],
            weight_decay=args['weight_decay'],
            warmup_steps=args['warmup_steps'],
            save_steps=args['save_steps'],
            save_strategy=args['save_strategy'],
            bf16=args['bf16'],
            deepspeed=args['deepspeed'],
            torch_compile = args['torch_compile'],
            evaluation_strategy=args['evaluation_strategy'],
            remove_unused_columns=args['remove_unused_columns'],
            seed=args['seed']
        )
    data_collator = DataCollator(tokenizer, args['max_seq_length'])
    loss_func = TargetLMLoss(ignore_index=-100)
    '''Train'''
    trainer = LoraTrainer(
        model=model, 
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset_eval, 
        data_collator=data_collator, 
        compute_loss=loss_func,
    )
    return trainer

def main():
    '''Parse Args'''
    args = setup_everything()
        
    '''Load Model'''
    trainer = init_components(args=args)
    logger.info("*** starting training ***")
    train_result = trainer.train()
    trainer.save_model(args['output_dir'])

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

if __name__ == "__main__":
    main()


