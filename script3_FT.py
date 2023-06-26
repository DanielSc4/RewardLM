#!/bin/env python

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from rewardlm.data.data_utils import get_DIALOCONAN_prepro, get_dataset_CLM
from rewardlm.utils import load_config
from huggingface_hub import login
import wandb
import os
from argparse import ArgumentParser
import datetime
now = datetime.datetime.now()   # getting current date for log



def print_trainable_parameters(model) -> None:
    """Prints the number of trainable parameters in the model
    """
    train_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            train_params += param.numel()
    
    print(f'trainable params: {train_params} || all params {all_params} || trainable(%): {train_params / all_params * 100:.2f}')


def get_dataset(config, tokenizer, val_set_per: float = .1):
    data = get_DIALOCONAN_prepro(**config['generation']['custom_prompt'])

    if config['data']['subset']:
        print('getting subset')
        # select only the first `subset_size` samples
        data = data[:config['data']['subset_size']]

    dataset = get_dataset_CLM(
        data, 
        context_length = 512, 
        tokenizer = tokenizer,
    )

    if val_set_per > 0:
        assert val_set_per < 1, f'val_set_per should be less than 1 ([0 - 99]%), {val_set_per} given'

        train_eval_d = dataset.train_test_split(
            test_size = val_set_per, shuffle=True, seed=42,
        )
        train_dataset, val_dataset = train_eval_d['train'], train_eval_d['test']
    else:
        train_dataset = dataset
        val_dataset = None

    return train_dataset, val_dataset



class CastOutputToFloat(nn.Sequential):
    def forward(self, x): 
        return super().forward(x).to(torch.float32)
    

def apply_LoRA(model, auto_prepare: bool):
    ## Required a peft model (ensure to have one using get_peft_model fun from peft)
    lora_config = LoraConfig(
        r = 32,
        lora_alpha = 32,
        target_modules = None, # handled automatically by peft
        lora_dropout = .05,
        bias = 'none',
        task_type = 'CAUSAL_LM',
    )

    if auto_prepare:
        print(f'[-] Preparing model for 8bit training [auto-mode] ...')
        model = prepare_model_for_int8_training(model)    # if prepare is False, the preprocessing is done before
    else:
        print(f'[-] Preparing model for 8bit training [manual-mode] ...')
        # manual model preparation
        for param in model.parameters():
            param.requires_grad = False     # freeze all parameters
            if param.ndim == 1:
                param.data = param.data.to(torch.float32)
        model.config.use_cache = False             # silence the warnings. Re-enable for inference!
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()         # Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping the model weights fixed.

        # cast final tensor logits to torch.float32 (or float16)
        setattr(
            model,
            list(model.named_children())[-1][0],    # name of the attribute
            CastOutputToFloat(getattr(model, list(model.named_children())[-1][0]))
        )
    
    print(f'[-] Getting peft model ...')
    model = get_peft_model(model, lora_config)

    return model


def main(config_name: str):
    print(now)
    print(f'[-] Loading {config_name} config')
    config = load_config(name = config_name)

    if torch.cuda.is_available():
        print(f'[-] CUDA detected, downloading {config["generation"]["model_id"]} model in 8-bit mode')
        load_8_bit = True
        repo_id = 'DanielSc4/' + config['generation']['model_id'].split('/')[1] + '-FT-LoRA-8bit-test1'
        model = AutoModelForCausalLM.from_pretrained(
            config['generation']['model_id'], 
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_8bit = True,
        )
    else:
        print(f'[-] No CUDA detected, downloading {config["generation"]["model_id"]} model, fp32')
        load_8_bit = False
        repo_id = 'DanielSc4/' + config['generation']['model_id'].split('/')[1] + '-FT-LoRA-test1'
        model = AutoModelForCausalLM.from_pretrained(
            config['generation']['model_id'], 
        )
    
    print(f'[-] Downloading tokenizer ...')
    tokenizer = AutoTokenizer.from_pretrained(config['generation']['model_id'])
    tokenizer.padding_side = "left"  # Allow batched inference
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0      # unk. we want this to be different from the eos token

    print(f'[-] Getting dataset ...')
    # dataset (default split 10% val, 90% train)
    train_dataset, val_dataset = get_dataset(config, tokenizer)
    train_dataset = train_dataset.shuffle()
    
    model = apply_LoRA(model=model, auto_prepare = False)
    print_trainable_parameters(model)


    train_args = TrainingArguments(
        **config['fine_tune_args'],
        output_dir=os.path.join(
            os.environ['PATH_TO_STORAGE'] if 'PATH_TO_STORAGE' in os.environ else '.', 
            'checkpoints/fine_tune/'
        ),
        fp16 = True if torch.cuda.is_available() else False,
        evaluation_strategy='steps' if val_dataset else 'no',
        eval_steps=200 if val_dataset else None,
        run_name=repo_id + str(now).replace(' ', '_'),
    )
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=train_args,
        data_collator = DataCollatorForSeq2Seq(     # Using data collator for seq2seq because the labels are already in dataset
            tokenizer, 
            pad_to_multiple_of=8,
            return_tensors='pt',
            padding=True,
        ),
    )
    print(f'[-] Training ...')
    trainer.train()

    print(f'[-] Uploading to HF hub ...')
    # assuming debug if subset is active
    if not config['data']['subset']:
        # push to hub
        model.push_to_hub(repo_id)
        print('https://huggingface.co/' + repo_id)
    
    print(f'[-] Done')


if __name__ == '__main__':
    parser = ArgumentParser(description='Process some integers.')
    parser.add_argument('-c', '--config', required = True, help = 'Config name (without the .yaml). Files are stored in PROJ_PATH/configs/*.yaml')

    args = parser.parse_args()
    config_name = args.config

    # login to huggingface_hub and wandb
    credentials = load_config(path = './', name = 'credentials')
    login(token = credentials['huggingface_hub'])
    wandb.login(anonymous='allow', key = credentials['wandb'])

    print(f'Running: {now}')

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['BITSANDBYTES_NOWELCOME'] = '1'
    main(config_name = config_name)
