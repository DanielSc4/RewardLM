# Performance-Efficient Fine-Tuning using Reinforcement Learning
# Adaptation of https://github.com/lvwerra/trl/blob/main/examples/sentiment/scripts/gpt-neox-20b_peft/gpt-neo-20b_sentiment_peft.py

import torch
from trl import PPOConfig, PPOTrainer, set_seed

from ..GenerativeModel import GenerativeModel
from ..RewardModel import RewardModel
from ...data.CustomDatasets import PromptsDataset

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
            device: str = 'cpu',
            dtype: str = 'fp32',
            seed: int = 42,
        ) -> None:
        """Manager of the trl

        Args:
            model_id (str): The model name
            reward_model_id (str): The reward model name.
            optimized (bool, optional) True if you want to use Performance Efficient Fine-Tuning and LoRA. Defaults to False.
            lr (float, optional): learning rate. Defaults to 1.41e-5.
            mini_bs (int, optional): PPO minibatch size. Defaults to 16.
            bs (int, optional): batch size. Defaults to 256.
            gradient_acc_steps (int, optional): the number of gradient accumulation steps. Defaults to 1.
            device (str, optional): device on which the model should be trained. Defaults to 'cpu'.
            seed (int, optionl): The seed. Defaults to 42.

            dataset (rewardlm.data.CustomDatasets.PromptsDataset): Dataset to be used. Defaults to None.
        """
        
        set_seed(seed)
        self.device = device
        self.generator_manager = GenerativeModel(model_id, device = self.device, load_dtype = dtype)
        self.reward_manager = RewardModel(reward_model_id, device = self.device)

        if optimized:
            self.generator_manager.apply_LoRA()

        self.config = PPOConfig(
            model_name = model_id,
            learning_rate = lr,
            log_with='wandb',
            mini_batch_size = mini_bs,
            batch_size = bs,
            gradient_accumulation_steps = gradient_acc_steps
        )

        
    def generate_dataset(
            self, 
            text: list[str], 
            max_len: int = 256, 
            custom_prompt: str = '{prompt}'
        ) -> torch.utils.data.Dataset:
        """Build dataset from training

        Args:
            text (list[str]): List of prompts (str)
            max_len (int, optional): max length for the tokenizer. Defaults to 256.
            custom_prompt (str, optional): format string containing '{prompt}' to modify the original prompt. Defaults to '{prompt}'.

        Returns:
            torch.utils.data.Dataset: The torch dataset used for training
        """
        return PromptsDataset(
            tokenizer = self.generator_manager.tokenizer,
            text = text,
            max_len = max_len,
            custom_prompt = custom_prompt,
        )
    

    def collator(self, data):
        return dict((key, [d[key] for d in data]) for key in data[0])
    
    
    def train_PPO(self, dataset: torch.utils.data.Dataset):
        
        self.generator_manager.wrap_valueHead()

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.generator_manager.model.parameters()), lr = self.config.learning_rate
        )

        ppo_trainer = PPOTrainer(
            self.config, 
            self.generator_manager.model, 
            ref_model = None, 
            tokenizer = self.generator_manager.tokenizer, 
            dataset = dataset, 
            data_collator = self.collator, 
            optimizer = optimizer
        )

        device = ppo_trainer.accelerator.device
        if ppo_trainer.accelerator.num_processes == 1:
            device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

        for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):

            self.generator_manager.model.gradient_checkpointing_disable()
            self.generator_manager.model.config.use_cache = True

            # get generation
            responses = []
            for prompt in batch['input_ids']:
                response = ppo_trainer.generate(prompt, self.generator_manager.generation_config)
                responses.append(response.squeeze()[len(prompt):])
            batch['response'] = [
                self.generator_manager.tokenizer.decode(r.squeeze()) for r in responses
            ]
            batch['prompt'] = [
                self.generator_manager.tokenizer.decode(p.squeeze()) for p in batch["input_ids"]
            ]

            # compute reward
            # texts = [
            #     q + r for q, r in zip(batch['prompt'], batch['response'])
            # ]
            
            _, score_resposne = self.reward_manager.get_score_pair(
                prompt = batch['prompt'],
                response = batch['response'],
            ) # TODO: or should be get_batch_score_pair (?)
            # what reward contains? Check on original script
            # Hp: score_resposne should be a list of torch.Tensor

            self.generator_manager.model.gradient_checkpointing_enable()
            self.generator_manager.model.pretrained_model.config.use_cache = False

            stats = ppo_trainer.step(batch['input_ids'], responses, score_resposne)
            ppo_trainer.log_stats(stats, batch, rewards = score_resposne)


    





        