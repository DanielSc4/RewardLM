import torch
from argparse import ArgumentParser
import yaml

import psutil
import os
# disable welcome message
os.environ['BITSANDBYTES_NOWELCOME'] = '1'

from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import inseq
from tqdm import tqdm
import time
import pandas as pd

from interp_utils import _assign_label


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
            # load in fp16, shouldn't be a problem w/ inseq and merging LoRA weights(?)
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
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



def stratify_df(df: pd.DataFrame, pre_set_size):
    assert 'pro_API_response_score' in df.columns
    
    df['label'] = df['pro_API_response_score'].apply(_assign_label)

    # selecting stratified
    c = df['label'].value_counts().apply(lambda x: x/len(df))
    stratified_df = pd.concat([
            group.sample(
                int(c[lbl] * pre_set_size), 
                replace=False, 
                random_state=42
            ) for lbl, group in df.groupby('label')
    ])
    return stratified_df

    # TODO: probably better selecting non stratified (?)



def select_prompts(model_config, data_config):
    df = pd.read_csv(DATASETS_PATHS[model_config['model_id']], index_col = 0)

    # used to jump ahead in the dataset in case of attribution already done in a previous backup (if 0 the entire df is left untouched)
    df = df[data_config['start_from']:]
    
    # if dubug on subset is active
    if model_config['data']['subset']:
        df = df.head(model_config['data']['subset_size'])
    
    if data_config['pre_set'] > 0:
        df = stratify_df(df, data_config['pre_set'])


    output = {
        'input_texts': df['prompts'].to_list(),
        'generated_texts': [],
    }
    respones = df['responses'].to_list()
    
    output['generated_texts'] = list(map(
        lambda prompt, respo: str(prompt) + str(respo) if str(respo) else '-', 
        output['input_texts'], 
        respones,
        )
    )

    return output


## TODO:
# [x] backup every 500 it, for out of memory reason
# [x] implement your own progress bar
# [x] near 133 of red_FT some generation are empty; ensure that response is not ''

def _get_pbar_desc():
    return f'RAM usage: {psutil.virtual_memory()[3] / 1e9:.2f} / {psutil.virtual_memory()[0] / 1e9:.0f} GB ({psutil.virtual_memory()[2]}%) | Progress'


def main(model_config, interp_config):

    # memory usage
    model, tokenizer = load_model(model_config['model_id'], model_config['load_from_peft'])

    # load in inseq
    seq_model = inseq.load_model(
        model, # model,
        attribution_method=interp_config['inseq_settings']['attribution_method'],
        tokenizer=tokenizer,
    )

    inputs = select_prompts(model_config, interp_config['data'])

    # initial checks
    print('Input_texts len: {l}'.format(l=len(inputs['input_texts'])))
    print('Generated_text len: {l}'.format(l=len(inputs['generated_texts'])))


    pbar = tqdm(
        enumerate(zip(*inputs.values())),
        desc=_get_pbar_desc(),
        total=len(inputs['generated_texts']),
    )

    # one by one since I want to control the progressbar and batchsize is not supported anyway
    for i, (input_text, generated_text) in pbar:
        pbar.set_description(_get_pbar_desc())

        out_tmp = seq_model.attribute(
            input_texts=input_text,
            generated_texts=generated_text,
            step_scores=["probability"],
            # generation_args = config['generation']['generation_config'],        # not used when contrained generation is on
            # batch_size = config['inference_batch_size'],
            show_progress = False,      # decluttering logs
            pretty_progress = False,
        )
        # output aggregation to store only a G x T matrix
        out_tmp = out_tmp.aggregate("subwords", special_symbol=("Ġ", "Ċ")).aggregate()

        # first it
        if i == 0:
            out = out_tmp
        else:
            out = inseq.FeatureAttributionOutput.merge_attributions([out, out_tmp])
        
        # backup every 500 attributions
        if i % 1 == 0:
            start = time.time()
            out.save(
                interp_config['data']['output_path'] + 'attributes_{model_name}_{it}it.json'.format(model_name = model_config['model_id'].split('/')[-1], it = i),
                overwrite=True,
            )
            end = time.time()
            print(end - start)
    
    
    print('[x] Saving all attributions')
    out.save(
        interp_config['data']['output_path'] + 'attributes_{model_name}.json'.format(model_name = model_config['model_id'].split('/')[-1]),
        overwrite=True,    
    )
    print('[x] Done')


if __name__ == '__main__':

    parser = ArgumentParser(description='lala')
    parser.add_argument(
        '-m', '--model_config', 
        required=True, 
        help='Config file (.yaml) of the model to test',
        default='configs/debug_GPT-neo.yaml',
    )
    parser.add_argument(
        '-i', '--interp_config', 
        required=True, 
        help='Config file (.yaml) for the interpretability script',
        default='interpretability/interp_configs/i_debug_GPT-neo.yaml',
    )
    

    args = parser.parse_args()

    model_config = read_config(args.model_config)
    interp_config = read_config(args.interp_config)

    main(model_config, interp_config)