import torch
from rewardlm.core.RL.RLModel import RLModel
from rewardlm.data.data_utils import get_DIALOCONAN_prepro
from transformers import GenerationConfig
from rewardlm.utils import load_config

from trl import (
    AutoModelForCausalLMWithValueHead,     # for the generative model
    PPOConfig,
    PPOTrainer,
    set_seed,
)
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from peft import LoraConfig
from tqdm import tqdm

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




def main(config_name: str):
    print(now)
    print(f'[-] Loading {config_name} config')
    config = load_config(name = config_name)

    ppo_config = PPOConfig(
        model_name=config['model_id'],
        **config['RL_args']['PPO_config'],
    )

    # We then define the arguments to pass to the sentiment analysis pipeline.
    # We set `return_all_scores` to True to get the sentiment score for each token.
    sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": config.mini_batch_size}

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
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.padding_side = "left"  # Allow batched inference
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0      # unk. we want this to be different from the eos token

    print_trainable_parameters(model=model)

    # TODO: dataset
    # ...
    print(f'[-] Getting dataset ...')
    dataset = []


    ppo_trainer = PPOTrainer(
        ppo_config, 
        model, 
        ref_model=None, 
        tokenizer=tokenizer, 
        dataset=dataset, 
        # TODO: define correct datacollator
        data_collator=DataCollatorForSeq2Seq(     # Using data collator for seq2seq because the labels are already in dataset
            tokenizer, 
            pad_to_multiple_of=8,
            return_tensors='pt',
            padding=True,
        ),
        # TODO: check if missing optimizer is good
    )

    print(f'[-] Training ...')
    for n_batch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]      # dim: (should) [batch_siz, n_tokens]

        model.gradient_checkpointing_disable()
        model.config.use_cache = True

        # Get response from Causal LM
        response_tensors = ppo_trainer.generate(
            query_tensors, 
            return_prompt=False, 
            **config['generation']['generation_config']
        )

        # decoded response
        batch["response"] = tokenizer.batch_decode(
            response_tensors,
            # skip_special_tokens=True,     # TODO: check if sholud be True
        )

        # concatenate query and response given by the model
        tot_texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        # TODO: get reward from RewardModel (token level?)
        pipe_outputs = [
            0.1,
            0.99,
            # ... len(query_tensors)
        ]
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

        # Run PPO step
        model.gradient_checkpointing_enable()
        model.pretrained_model.config.use_cache = False
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)


    print(f'[-] Uploading to HF hub ...')
    # assuming debug if subset is active
    if not config['data']['subset']:
        # push to hub
        model.push_to_hub(repo_id)
        print('https://huggingface.co/' + repo_id)
    
    print(f'[-] Done')

    # rlmanager = RLModel(
    #     model_id = config['generation']['model_id'],
    #     reward_model_id = config['reward']['model_id'],
    #     optimized = True,   # use LoRA
    #     bs = config['PPO']['bs'],
    #     mini_bs = config['PPO']['mini_bs'],
    #     # force the use of CPU on Apple Silicon devices (mps not supported):
    #     accelerator_kwargs = {
    #         'cpu': False if torch.cuda.is_available() else True,
    #     },
    #     generation_config=GenerationConfig(
    #         max_new_tokens = 512,
    #         min_new_tokens = 5,
    #         pad_token_id = 0,       # crashes while using batchsize > 1 only on mps device if not set
    #         temperature = 1,
    #         top_p = .7,
    #         top_k = 0,
    #         do_sample = True
    #         # diversity_penalty = .1, # should use num_beam_groups > 1
    #     )
    # )

    # data = get_DIALOCONAN_prepro(delete_last_assistant_response = True)
    # if config['data']['subset']:
    #     # select only the first `subset_size` samples
    #     data = data[:config['data']['subset_size']]
    # dataset = rlmanager.generate_dataset(text = data)

    # stats = rlmanager.train_PPO(dataset = dataset)
    # print('Done')

    # # assuming debug if subset is active
    # if not config['data']['subset']:
    #     # save trainer (model, tokenizer & config) to the hub
    #     repo_id = 'DanielSc4/' + config['generation']['model_id'].split('/')[1] + '-RL-LoRA-test0'

    #     rlmanager.push_generator_to_hub(repo_id = repo_id)



if __name__ == '__main__':
    parser = ArgumentParser(description='Process some integers.')
    parser.add_argument('-c', '--config', required = True, help = 'Config name (without the .yaml). Files are stored in PROJ_PATH/configs/*.yaml')
    
    args = parser.parse_args()
    config_name = args.config

    # login to huggingface_hub and wandb
    credentials = load_config(path = './', name = 'credentials')
    login(token = credentials['huggingface_hub'])
    wandb.login(anonymous='allow', key = credentials['wandb'])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['BITSANDBYTES_NOWELCOME'] = '1'
    main(config_name=config_name)