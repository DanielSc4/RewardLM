#!/bin/env python

import torch
from transformers import GenerationConfig
from rewardlm.data.data_utils import get_DIALOCONAN_prepro, get_dataset_CLM
from rewardlm.core.GenerativeModel import GenerativeModel
from rewardlm.utils import load_config
from huggingface_hub import login

token = load_config(path = './', name = 'credentials')['huggingface_hub']
login(token = token)

config = load_config(name = 'debug_FT')
repo_id = 'DanielSc4/' + config['generation']['model_id'].split('/')[1] + '-FT-LoRA-test1'

generator_manager = GenerativeModel(
    config['generation']['model_id'],
    load_dtype = '8-bit' if torch.cuda.is_available() else 'fp32',
    # force the use of CPU on Apple Silicon devices (mps not supported):
    generation_config=GenerationConfig(**config['generation']['generation_config']),
    accelerator_kwargs = {
        'cpu': False if torch.cuda.is_available() else True,
    },
)

# download dataset
data = get_DIALOCONAN_prepro(**config['generation']['custom_prompt'])
if config['data']['subset']:
    print('getting subset')
    # select only the first `subset_size` samples
    data = data[:config['data']['subset_size']]
dataset = get_dataset_CLM(
    data, 
    context_length = 512, 
    tokenizer = generator_manager.tokenizer
)

generator_manager.fine_tune(
    dataset = dataset, 
    val_set_per=.1,     # 10% of the dataset
    optimized = True,   # if torch.cuda.is_available() else False,
    lr = config['generation']['lr'],
    epochs = config['generation']['epochs'],
    initial_bs = config['generation']['initial_bs'],
    run_name = repo_id.split('/')[1],
)

# assuming debug if subset is active
if not config['data']['subset']:
    # save model to the hub
    generator_manager.push_to_hub(repo_id = repo_id)