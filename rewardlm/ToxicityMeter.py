import torch
from transformers import GenerationConfig
from tqdm import tqdm
import pandas as pd

from .utils.general_utils import device_selector
from .data.data_utils import gen_loader
from .data.CustomDatasets import ToxicityGeneratedSet
from .core.RewardModel import RewardModel
from .core.GenerativeModel import GenerativeModel



class ToxicityMeter:
    def __init__(self, 
        generator_manager: GenerativeModel = None,
        reward_manager: RewardModel = None,
    ) -> None:
        """Measures the toxicity of a given generative model based on the output of the toxicity_model

        Args:
            model_id (str): model name from HuggingFace
            load_dtype (str): dtype of the model, can be 8-bit, bf16 otherwise fp32 (standard mode)
            device (str, optional): device where tensors will be placed. Can be 'cuda', 'mps' or 'cpu'. Important: dtype = 8bit can be used only on 'cuda'. Defaults to None.
        """
        
        assert generator_manager is not None, 'please use a GenerativeModel, None provided'
        self.generator_manager = generator_manager

        if reward_manager is None:
            self.reward_manager = RewardModel('facebook/roberta-hate-speech-dynabench-r4-target')
        

    def __get_prompts_responses(
            self, 
            prompts, 
            responses, 
            print_r = True
        ):
        # prompts: inputs['input_ids']
        # responses: `output` of model.generate(**inputs, ...)

        colors = {
            'yellow': '\033[93m',
            'green': '\033[92m',
            'red': '\033[91m',
            'blue': '\033[94m',
            'purple': '\033[95m',
        }

        prompts = self.generator_manager.tokenizer.batch_decode(prompts, skip_special_tokens = True)
        responses = self.generator_manager.tokenizer.batch_decode(responses, skip_special_tokens = True)

        clean_responses = []
        for prompt, response in zip(prompts, responses):
            # cleaning output of the model
            clean_responses.append(response[len(prompt):].replace('\n', ' '))
            if print_r:
                print(colors['blue'] + prompt, end = '')
                print(colors['purple'] + clean_responses[-1])
            
        return (prompts, clean_responses)
    

    def measure_toxicity(
            self,
            text_prompt: list[str],
            custom_prompt: str,
            print_response: bool = False,
            batch_size: int = 8,
        ):
        """Main function for measuring the toxicity of the model using RealToxicityPrompt as benchmark. 
        Responses are generated and consequently the toxicity of both the prompt and the response is measured

        Args:
            custom_prompt (str): format string where '{prompt}' is the original prompt. Defaults to '{prompt}'.
            generation_config (GenerationConfig, optional): GenerationConfig from transformer generation. Defaults to None.
            print_response (bool, optional): print each response during generation. Defaults to False.

        Returns:
            pandas.DataFrame: Pandas DataFrame containing the prompts, the responses, and measured toxicity both from the prompt and the response
        """

        generation = {
            'prompts': [],
            'responses': [],
        }

        # preparing loader
        loader = gen_loader(
            text = text_prompt,
            tokenizer = self.generator_manager.tokenizer, 
            max_len = 128 + len(custom_prompt),
            custom_prompt = custom_prompt,
            batch_size = batch_size,
        )

        # generating text
        for inputs in tqdm(loader):
            for ele in inputs:
                inputs[ele] = inputs[ele].to(self.generator_manager.accelerator.device)
            
            output = self.generator_manager.inference_fine_tuned(
                tokenized_batch = inputs,
                return_decoded = False,
            )
            prmpt, rspns = self.__get_prompts_responses(
                prompts = inputs['input_ids'],
                responses = output,
                print_r = print_response,
            )
            generation['prompts'].extend(prmpt)
            generation['responses'].extend(rspns)
        
        # generating toxicity scores from reward model
        gen_tox_df = pd.DataFrame.from_dict(generation)
        model_tox_set = ToxicityGeneratedSet(
            prompts = gen_tox_df['prompts'].to_list(),
            responses = gen_tox_df['prompts'].to_list(),
            tokenizer = self.reward_manager.tokenizer, 
            max_len = 128,
        )
        model_tox_loader = torch.utils.data.DataLoader(
            model_tox_set, 
            batch_size = 32, 
            shuffle = False,
        )

        result_tox = self.reward_manager.get_batch_score_pair(model_tox_loader)
        # adding toxicity scores
        toxicity_df = gen_tox_df.join(pd.DataFrame.from_dict(result_tox))

        return toxicity_df
