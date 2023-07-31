import torch
from argparse import ArgumentParser
import yaml

import os
# disable welcome message
os.environ['BITSANDBYTES_NOWELCOME'] = '1'

from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import inseq

import pandas as pd


DATASETS_PATHS = {
    'EleutherAI/gpt-neo-125m': 'results/new_prompts/measured_tox_PT_gpt-neo-125m.csv',      # Debug
    'tiiuae/falcon-7b-instruct': 'results/new_prompts/measured_tox_PT_falcon-7b-instruct.csv',      # PT
    'DanielSc4/falcon-7b-instruct-FT-LoRA-8bit-test1': 'results/new_prompts/measured_tox_FT_falcon-7b-instruct-FT-LoRA-8bit-test1.csv',     # FT
    'DanielSc4/falcon-7b-instruct-RL-LoRA-8bit-test1': 'results/new_prompts/measured_tox_RL_falcon-7b-instruct-RL-LoRA-8bit-test1.csv',     # RL
    'togethercomputer/RedPajama-INCITE-Chat-3B-v1': 'results/new_prompts/measured_tox_PT_RedPajama-INCITE-Chat-3B-v1.csv',      # PT
    'DanielSc4/RedPajama-INCITE-Chat-3B-v1-FT-LoRA-8bit-test1': 'results/new_prompts/measured_tox_FT_RedPajama-INCITE-Chat-3B-v1-FT-LoRA-8bit-test1.csv',       # FT
    'DanielSc4/RedPajama-INCITE-Chat-3B-v1-RL-LoRA-8bit-test1': 'results/new_prompts/measured_tox_RL_RedPajama-INCITE-Chat-3B-v1-RL-LoRA-8bit-test1.csv',       # RL
}


def read_config(config_path):
    with open(config_path) as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def load_model(model_id: str, load_from_peft: bool):

    def download_model(pretrained):
        return AutoModelForCausalLM.from_pretrained(
            pretrained,
            # load in bf16, shouldn't be a problem w/ inseq and merging LoRA weights(?)
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map = 'auto' if torch.cuda.is_available() else 'cpu',
            trust_remote_code=True,
        )
    
    if load_from_peft:
        lora_config = LoraConfig.from_pretrained(model_id)
        original_pretrained = lora_config.base_model_name_or_path

        base_model = download_model(original_pretrained)
        model = PeftModel.from_pretrained(base_model, model_id)
        # merging LoRA weights. Note that 8-bit is not (yet) supported
        model = model.merge_and_unload()
    else:
        original_pretrained = model_id
        model = download_model(original_pretrained)
    
    tokenizer = AutoTokenizer.from_pretrained(original_pretrained)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.pad_token
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id

    return model, tokenizer



def select_prompts(config):
    df = pd.read_csv(DATASETS_PATHS[config['model_id']], index_col = 0)
    
    # if dubug on subset is active
    if config['data']['subset']:
        df = df.head(config['data']['subset_size'])

    output = {
        'input_texts': df['prompts'].to_list(),
        'generated_texts': [],
    }
    respones = df['responses'].to_list()
    
    output['generated_texts'] = list(map(
        lambda prompt, respo: str(prompt) + str(respo), 
        output['input_texts'], 
        respones,
        )
    )

    return output



def main(config, args):
    model, tokenizer = load_model(config['model_id'], config['load_from_peft'])

    # load in inseq
    seq_model = inseq.load_model(
        model, # model,
        attribution_method=args.attribution_method,
        tokenizer=tokenizer,
    )

    inputs = select_prompts(config)

    # debug checks
    print('Input_texts len: {l}'.format(l=len(inputs['input_texts'])))
    print('Generated_text len: {l}'.format(l=len(inputs['generated_texts'])))

    out = seq_model.attribute(
        **inputs,
        step_scores=["probability"],
        generation_args = config['generation']['generation_config'],
        batch_size = config['inference_batch_size'],
    )
    
    print(out.show())
    print('[x] Saving attributes')
    out.save(
        args.output_path + 'attributes_{model_name}.json'.format(model_name = config['model_id'].split('/')[-1]),
        overwrite=True,    
    )





if __name__ == '__main__':

    parser = ArgumentParser(description='lala')
    parser.add_argument(
        '-c', '--config', 
        required=True, 
        help='Config file (.yaml) of the model to test',
        default='configs/debug_GPT-neo.yaml',
    )
    parser.add_argument(
        '-a', '--attribution_method', 
        required=False, 
        help='Attribuition method used for inseq',
        default='input_x_gradient',
    )
    parser.add_argument(
        '-o', '--output_path', 
        required=False, 
        help='Attribuition method used for inseq',
        default='./results/interp_res/',
    )

    args = parser.parse_args()

    config = read_config(args.config)

    main(config, args)