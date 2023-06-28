import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['BITSANDBYTES_NOWELCOME'] = '1'

from torch.utils.data import DataLoader
from rewardlm.core.RewardModel import RewardModel
from rewardlm.data.data_utils import get_DIALOCONAN_prepro
from rewardlm.data.CustomDatasets import ToxicityGeneratedSet
from rewardlm.utils import load_config

from trl import (
    AutoModelForCausalLMWithValueHead,     # for the generative model
    PPOConfig,
    PPOTrainer,
    set_seed,
)
set_seed(42)

from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from datasets import Dataset
from peft import LoraConfig
from tqdm import tqdm

from huggingface_hub import login
import wandb

from argparse import ArgumentParser
import datetime
import time
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

def get_dataset(config, tokenizer):
    data = get_DIALOCONAN_prepro(
        **config['generation']['custom_prompt'],
        delete_last_assistant_response = True
    )
    if config['data']['subset']:
        print('[-] getting subset')
        # select only the first `subset_size` samples
        data = data[:config['data']['subset_size']]
    
    ds = Dataset.from_dict({'text': data})

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["text"])
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample
    
    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds
    
def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

def gpu_usage():
    if torch.cuda.is_available():
        return [e/1024/1024/1024 for e in torch.cuda.mem_get_info()]
    else:
        'ERR: no cuda GPU detected'

def main(config_name: str):
    print(now)
    print(f'[-] Loading {config_name} config')

    config = load_config(name = config_name)
    debug = config['debug']
    if debug:
        print(f'[init] global free and total GPU memory occupied: {gpu_usage()} GB.')

    ppo_config = PPOConfig(
        model_name=config['model_id'],
        **config['RL_args']['PPO_config'],
    )

    # We then define the arguments to pass to the sentiment analysis pipeline.
    # We set `return_all_scores` to True to get the sentiment score for each token.
    # sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": config['RL_args']['PPO_config']['mini_batch_size']}

    lora_config = LoraConfig(
        **config['LoRA_config'],
    )

    # download model
    if torch.cuda.is_available():
        print(f'[-] CUDA detected, downloading {config["model_id"]} model in 8-bit mode')
        repo_id = 'DanielSc4/' + config['model_id'].split('/')[1] + '-RL-LoRA-8bit-test1'
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            config['model_id'],
            load_in_8bit=True,
            torch_dtype=torch.float16,
            peft_config=lora_config,
            trust_remote_code=True,
        )
    else:
        print(f'[-] No CUDA detected, downloading {config["model_id"]} model, fp32')
        repo_id = 'DanielSc4/' + config['model_id'].split('/')[1] + '-RL-LoRA-test1'
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            config['model_id'],
            peft_config=lora_config,
            trust_remote_code=True,
        )

    print(f'[-] Downloading tokenizer ...')
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'])
    tokenizer.padding_side = "left"  # Allow batched inference
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token      # unk. we want this to be different from the eos token

    print_trainable_parameters(model=model)

    # dataset
    print(f'[-] Getting dataset ...')
    dataset = get_dataset(config=config, tokenizer=tokenizer)

    ppo_trainer = PPOTrainer(
        ppo_config, 
        model, 
        ref_model=None, 
        tokenizer=tokenizer, 
        dataset=dataset, 
        data_collator=collator
    )

    print(f'[-] Getting reward_model ...')
    reward_manager = RewardModel(
        config['RL_args']['reward_model_id'], 
        device = ppo_trainer.accelerator.device,
    )

    print(f'[-] Training ...')
    for n_batch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]      # dim: (should) [batch_siz, n_tokens]

        model.gradient_checkpointing_disable()
        model.config.use_cache = True

        if debug:
            print(f'  [t] Generating ...')
            start = time.time()
        # Get response from Causal LM
        response_tensors = ppo_trainer.generate(
            query_tensors, 
            return_prompt=False, 
            **config['generation']['generation_config'],
            pad_token_id=tokenizer.eos_token_id,        # `to avoid Setting `pad_token_id` to `eos_token_id`:11 for open-end generation.` warning
        )

        if debug:
            end = time.time()
            print(f'  [t] \'- elapsed: {end - start}')
            start = time.time()
            print(f'  [t] Decoding responses ...')
        # decoded response
        batch["response"] = tokenizer.batch_decode(
            response_tensors,
            skip_special_tokens=True,     # TODO: check if sholud be False to keep the same tensor batch dimension (s)!
        )

        # concatenate query and response given by the model (useless; calculating scores only based on responses)
        # tot_texts = [q + r for q, r in zip(batch["query"], batch["response"])]

        if debug:
            end = time.time()
            print(f'  [t] \'- elapsed: {end - start}')
            start = time.time()
            print(f'  [t] Generating new dataset for rewards ...')
        model_tox_set = ToxicityGeneratedSet(
            prompts = batch['query'],
            responses = batch['response'],
            tokenizer = reward_manager.tokenizer,
            max_len = 512,
        )

        if debug:
            end = time.time()
            print(f'  [t] \'- elapsed: {end - start}')
            start = time.time()
            print(f'  [t] Getting rewards score ... bs = {len(batch["query"])}')
        result_tox = reward_manager.get_batch_score_pair(
            DataLoader(model_tox_set, batch_size = len(batch['query']), shuffle = False)
        ) 
        rewards = [torch.tensor(s) for s in result_tox['response_score']]

        # debug output w/ decoded query, response and calculated score
        if debug:
            for q, r, s in zip(batch['query'], batch['response'], rewards):
                q = q.replace('\n', ' ').rstrip()
                r = r.replace('\n', ' ').rstrip()
                print('\t query:')
                if len(q) > 200:
                    print(f'\t {q[:100]} [...] {q[20:]}')
                else:
                    print(f'\t {q}')
                print('\t response:')
                if len(r) > 200:
                    print(f'\t {r[:100]} [...] {r[20:]}')
                else:
                    print(f'\t {r}')
                print(f'\t score: {s}')
                print()

        # Run PPO step
        model.gradient_checkpointing_enable()
        model.pretrained_model.config.use_cache = False
        if debug:
            end = time.time()
            print(f'  [t] \'- elapsed: {end - start}')
            start = time.time()
            print(f'  [t] [pre-step] global free and total GPU memory occupied: {gpu_usage()} GB.')
            print(f'  [t] Updating model ...')
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        if debug:
            end = time.time()
            print(f'  [t] \'- elapsed: {end - start}')
            start = time.time()
            print(f'  [t] [post-step] global free and total GPU memory occupied: {gpu_usage()} GB.')
            print(f'  [t] Model updated ...')
        ppo_trainer.log_stats(stats, batch, rewards)


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

    main(config_name=config_name)
