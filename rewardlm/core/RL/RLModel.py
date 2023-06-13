# Performance-Efficient Fine-Tuning using Reinforcement Learning
# Adaptation of https://github.com/lvwerra/trl/blob/main/examples/sentiment/scripts/gpt-neox-20b_peft/gpt-neo-20b_sentiment_peft.py

from pyparsing import Any
import torch
from torch.utils.data import DataLoader
from trl import PPOConfig, PPOTrainer, set_seed
from accelerate import Accelerator
from transformers import GenerationConfig
import datasets
from datasets import Dataset


from ..GenerativeModel import GenerativeModel
from ..RewardModel import RewardModel
from ...data.CustomDatasets import PromptsDataset, ToxicityGeneratedSet

from tqdm import tqdm

class RLModel:
    def __init__(
            self, 
            model_id: str, 
            reward_model_id: str, 
            optimized: bool = False,
            lr: float = 1.41e-5, 
            mini_bs: int = 16, 
            bs: int = 256, 
            gradient_acc_steps: int = 1, 
            dtype: str = 'fp32',
            seed: int = 42,
            log_method: str = None,
            accelerator_kwargs: dict = {},
            generation_config: GenerationConfig = None,
        ) -> None:
        """Manager of the trl

        Args:
            `model_id` (str): The model name
            `reward_model_id` (str): The reward model name.
            `optimized` (bool, optional) True if you want to use Performance Efficient Fine-Tuning and LoRA. Defaults to False.
            `lr` (float, optional): learning rate. Defaults to 1.41e-5.
            `mini_bs` (int, optional): PPO minibatch size. Defaults to 16.
            `bs` (int, optional): batch size. Defaults to 256.
            `gradient_acc_steps` (int, optional): the number of gradient accumulation steps. Defaults to 1.
            `device` (str, optional): device on which the model should be trained. Defaults to 'cpu'.
            `seed` (int, optionl): The seed. Defaults to 42.
            `log_method` (str, optional): Log stats with ['wandb', 'tensorboard']. Defaults to None.
            `dataset` (rewardlm.data.CustomDatasets.PromptsDataset): Dataset to be used. Defaults to None.
        """
        assert bs >= mini_bs, f'bs should be >= than mini_bs. Got bs: {bs}; mini_bs: {mini_bs}'

        self.accelerator = Accelerator(**accelerator_kwargs)

        set_seed(seed)
        self.generator_manager = GenerativeModel(
            model_id, 
            device=self.accelerator.device, 
            load_dtype=dtype, 
            generation_config=generation_config,
        )
        self.reward_manager = RewardModel(reward_model_id, device = self.accelerator.device)

        if optimized:
            self.generator_manager.apply_LoRA()
            self.generator_manager.print_trainable_parameters()

        self.config = PPOConfig(
            model_name = model_id,
            learning_rate = lr,
            log_with = log_method,
            mini_batch_size = mini_bs,
            batch_size = bs,
            gradient_accumulation_steps = gradient_acc_steps,
            accelerator_kwargs = accelerator_kwargs,
        )
        self.ppo_trainer = None

        
    def generate_dataset(
            self, 
            text: list[str], 
            max_len: int = 256, 
            custom_prompt: str = '{prompt}'
        ) -> torch.utils.data.Dataset | datasets.Dataset:
        """Build dataset from training

        Args:
            text (list[str]): List of prompts (str)
            max_len (int, optional): max length for the tokenizer. Defaults to 256.
            custom_prompt (str, optional): format string containing '{prompt}' to modify the original prompt. Defaults to '{prompt}'.

        Returns:
            torch.utils.data.Dataset: The torch dataset used for training
        """
        # legacy:
        # return PromptsDataset(
        #     tokenizer = self.generator_manager.tokenizer,
        #     text = text,
        #     max_len = max_len,
        #     custom_prompt = custom_prompt,
        # )

        adj_prompt = list(map(
            lambda s: custom_prompt.format(prompt = s), 
            text)
        )
        
        self.generator_manager.tokenizer.pad_token = self.generator_manager.tokenizer.eos_token
        ds = Dataset.from_dict({'text': adj_prompt})

        def tokenize(sample):
            prompt = sample['text']
            # continuation = sample['continuation']

            sample['input_ids'] = self.generator_manager.tokenizer.encode(prompt)
            sample['query'] = self.generator_manager.tokenizer.decode(sample['input_ids'])
            return sample
        
        ds = ds.map(tokenize, batched=False)
        ds.set_format(type='torch')
        ds = ds.train_test_split(test_size = .15, shuffle = False)['train']
        return ds


    def collator(self, data):
        return dict(
            (key, [d[key] for d in data]) for key in data[0]
        )
    
    
    def train_PPO(
            self, 
            dataset: torch.utils.data.Dataset | datasets.Dataset, 
            model_save_path: str = './checkpoints/',
        ) -> tuple[PPOTrainer, dict[str, Any]]:
        """Custom PPO trainer, train loop with automated reward system based on the `rewardlm.core.rewardModel` class

        Args:
            dataset (torch.utils.data.Dataset | datasets.Dataset): dataset, output from the `self.generate_dataset` fun
            model_save_path (str, optional): Path to save the model. Defaults to './checkpoints/'.

        Returns:
            tuple[PPOTrainer, dict[str, Any]]: Trainer and stats about the training process
        """

        tot_stats = []

        # autoregressive model with a value head in addition to the language model head.
        self.generator_manager.wrap_valueHead()

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.generator_manager.model.parameters()), lr = self.config.learning_rate
        )

        self.ppo_trainer = PPOTrainer(
            self.config, 
            self.generator_manager.model, 
            ref_model = None,   # let the PPO trainer deal with the reference model
            tokenizer = self.generator_manager.tokenizer, 
            dataset = dataset, 
            data_collator = self.collator,      # now providing w/ datasets.Dataset fun
            optimizer = optimizer,
        )

        # creating dataloader manually
        model_loader = DataLoader(
            dataset = dataset, 
            batch_size = self.config.batch_size,
            drop_last = True,       # There is a ValueError if the last batch does not match the batchsize dimension! (trl/trainer/ppo_trainer.py:408)
        )

        # legacy:
        # model_loader = self.accelerator.prepare(model_loader)
        model_loader = self.ppo_trainer.dataloader

        print(f'loader len: {len(model_loader)}')
        for n_batch, batch in tqdm(enumerate(model_loader)):
            self.generator_manager.model.gradient_checkpointing_disable()
            self.generator_manager.model.config.use_cache = True

            # get generation
            responses = []
            for prompt in batch['input_ids']:
                response = self.ppo_trainer.generate(
                    prompt, 
                    generation_config = self.generator_manager.generation_config,
                )
                responses.append(response.squeeze()[len(prompt):])

            batch['prompt'] = [
                self.generator_manager.tokenizer.decode(p.squeeze(), skip_special_tokens = True) for p in batch["input_ids"]
            ]
            
            # ### DEBUG pt.1
            # for i, gg in enumerate(responses):
            #     print(f'{i}- {gg.shape}: {gg}')
            # ### DEBUG pt.1

            batch['response'] = [
                self.generator_manager.tokenizer.decode(
                    r.squeeze(), skip_special_tokens = True
                ) for r in responses
            ]
            
            ### DEBUG pt.2
            # ### print statement to check the prompt, response pair of the current batch
            # for i, (pro, res) in enumerate(zip(batch['prompt'], batch['response'])):
            #     print(f'Input n. {i}, \n\t --Prompt-len: {len(pro)}-> "{pro.rstrip()}"\n\t --Generation-len: {len(res)}-> "{res.rstrip()}"\n{"-"*20}\n')
            # ### END DEBUG pt.2

            model_tox_set = ToxicityGeneratedSet(
                prompts = batch['prompt'],
                responses = batch['response'],
                tokenizer = self.reward_manager.tokenizer,
                max_len = len(batch["input_ids"][0]),       # keep the same length
            )

            result_tox = self.reward_manager.get_batch_score_pair(
                DataLoader(model_tox_set, batch_size = 32, shuffle = False)
            ) # TODO: or should be get_batch_score_pair (?)
            # what reward contains? Check on original script
            # Hp: score_resposne should be a list of torch.Tensor

            self.generator_manager.model.gradient_checkpointing_enable()
            self.generator_manager.model.pretrained_model.config.use_cache = False
            
            try:
                stats = self.ppo_trainer.step(
                    queries = list(batch['input_ids']),     # get list of tensor, shape [sample, max_len]
                    responses = responses, 
                    scores = [torch.tensor(s) for s in result_tox['response_score']],
                )
                self.ppo_trainer.log_stats(stats, batch, rewards = result_tox['response_score'])
            
            except ValueError as ve:
                print(ve)
                print(f'Skipping current batch [n: {n_batch}]')
            
            tot_stats.append(stats)

            # Save model every 2 batch
            if n_batch % 2 == 0:
                if self.ppo_trainer.accelerator.is_main_process:
                    self.ppo_trainer.save_pretrained(model_save_path)
                    print('PPO trainer checkpoint saved')
            
        return tot_stats

    def push_generator_to_hub(self, repo_id: str):
        """After training w/ PPO | FT this function pushes the generator model to HuggingFace Hub to share/backup it.
        Note that, if you are using LoRA adapters, only the model the adapters will be pushed (original models remain the pretrained)

        Args:
            repo_id (str): Repository id in the form of username/name-repository
        """
        assert self.ppo_trainer != None, 'You should train the model first using train_PPO function'
        
        self.generator_manager.model.push_to_hub(repo_id)
        
        print('https://huggingface.co/' + repo_id)