import torch
from argparse import ArgumentParser
import yaml
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import inseq

import os
# disable welcome message
os.environ['BITSANDBYTES_NOWELCOME'] = '1'




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
    
    return model



def select_prompts():
    # how to do?

    # tmp:
    return ['hello world', 'this is a test']



def main(config, args):
    model = load_model(config['model_id'], config['load_from_peft'])
    
    # load in inseq
    seq_model = inseq.load_model(
        model, # model,
        attribution_method=args.attribution_method
    )

    prompts = select_prompts()

    out = seq_model.attribute(
        prompts,
        step_scores=["probability"],
        generation_args = config['generation']['generation_config'],
        batch_size = config['inference_batch_size'],
    )
    
    print('[x] Saving attributes')
    out.save(args.output_path + 'attributes_{model_name}.json'.format(model_name = config['model_id'].split('/')[-1]))
    






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