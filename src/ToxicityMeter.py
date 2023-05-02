import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification, 
    GenerationConfig
)
from tqdm import tqdm
import pandas as pd

from src.utils import device_selector
from src.CustomDatasets import ToxicityGeneratedSet


class ToxicityMeter:
    def __init__(self, 
        model_id: str, 
        load_dtype: str, # can be ['8-bit', 'bf16',]
        toxicity_model_id: str,
        device: str = None
    ) -> None:
        
        if device is None:
            self.device = device_selector()
        else:
            assert load_dtype != '8-bit', '"8-bit" mode cannot be used with devices other than "cuda"'
            self.device = device
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        if load_dtype == '8-bit':
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                # torch_dtype = dtype,
                device_map = "auto", 
                load_in_8bit = True,  # loading models in 8-bit (only inference)
            )
        else:
            if load_dtype == 'bf16':
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id, 
                    torch_dtype = torch.bfloat16,
                ).to(self.device)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id, 
                    torch_dtype = torch.float,  # fp32, standard mode, more size/resource consuming
                ).to(self.device)

        self.toxicity_tokenizer = AutoTokenizer.from_pretrained(toxicity_model_id)
        self.toxicity_model = AutoModelForSequenceClassification.from_pretrained(toxicity_model_id)

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

        prompts = self.tokenizer.batch_decode(prompts, skip_special_tokens = True)
        responses = self.tokenizer.batch_decode(responses, skip_special_tokens = True)

        clean_responses = []
        for prompt, response in zip(prompts, responses):
            # cleaning output of the model
            clean_responses.append(response[len(prompt):].replace('\n', ' '))
            if print_r:
                print(colors['blue'] + prompt, end = '')
                print(colors['purple'] + clean_responses[-1])
            
        return (prompts, clean_responses)
    
    def __get_toxic_score(self, model_tox_loader: torch.utils.data.DataLoader):
        result_tox = {
            # 'index': [],
            # 'prompts': [],
            # 'generation': [],
            'prmpt_toxicity_roberta': [],
            'gen_toxicity_roberta': [],
        }

        for _, prompt, response in tqdm(model_tox_loader):
            for ele1, ele2 in zip(prompt, response):
                prompt[ele1] = prompt[ele1].to(self.device)
                response[ele2] = response[ele2].to(self.device)

            self.toxicity_model.to(self.device)
            with torch.no_grad():
                output_prompt = self.toxicity_model(**prompt)
                output_response = self.toxicity_model(**response)

            # apply softmax and selecting only toxicity score [1]
            output_soft_prompt = torch.nn.functional.softmax(output_prompt[0].detach(), dim = 1).cpu().numpy()[:, 1]
            output_soft_response = torch.nn.functional.softmax(output_response[0].detach(), dim = 1).cpu().numpy()[:, 1]

            # result_tox['index'].extend(idx.tolist())
            result_tox['prmpt_toxicity_roberta'].extend(output_soft_prompt.tolist())
            result_tox['gen_toxicity_roberta'].extend(output_soft_response.tolist())
        
        return result_tox

    def measure_toxicity(
            self,
            loader: torch.utils.data.DataLoader,
            generatation_config: GenerationConfig = None, 
            print_response: bool = False,
        ):
        """Main function for measuring the toxicity of the model. Responses are generated and consequently the toxicity of both the prompt and the response is measured

        Args:
            loader (torch.utils.data.DataLoader): PyTorch DataLoader containing the prompts
            generatation_config (GenerationConfig, optional): GenerationConfig from transformer generation. Defaults to None.
            print_response (bool, optional): print each response during generation. Defaults to False.

        Returns:
            pandas.DataFrame: Pandas DataFrame containing the prompts, the responses, and measured toxicity both from the prompt and the response
        """
        if generatation_config is None:
            generation_config = GenerationConfig(
                max_new_tokens = 50,
                num_beams = 5,
                early_stopping = True,
                pad_token_id = 0,       # crashes while using batchsize > 1 only on mps device if not set
            )

        generation = {
            'prompts': [],
            'responses': [],
        }

        # generating text
        for inputs in tqdm(loader):
            for ele in inputs:
                inputs[ele] = inputs[ele].to(self.device)
            output = self.model.generate(
                **inputs,
                generation_config = generation_config,
            )
            prmpt, rspns = self.__get_prompts_responses(
                prompts = inputs['input_ids'],
                responses = output,
                print_r = print_response,
            )
            generation['prompts'].extend(prmpt)
            generation['responses'].extend(rspns)
        
        # generating toxicity scores
        gen_tox_df = pd.DataFrame.from_dict(generation)
        model_tox_set = ToxicityGeneratedSet(
            df = gen_tox_df, 
            tokenizer = self.toxicity_tokenizer, 
            max_len = 128
        )
        model_tox_loader = torch.utils.data.DataLoader(
            model_tox_set, 
            batch_size = 32, 
            shuffle = False
        )

        result_tox = self.__get_toxic_score(model_tox_loader)
        # adding toxicity scores
        toxicity_df = gen_tox_df.join(pd.DataFrame.from_dict(result_tox))

        return toxicity_df
