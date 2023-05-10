import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

from ..utils import general_utils

class RewardModel:
    def __init__(self, model_id: str, device: str = None) -> None:
        """Reward Model manager, include tokenizer, model and utils stuff

        Args:
            model_id (str): model id or path from 🤗 Hugging Face
            device (str): ['cpu', 'cuda', 'mps', None]. Defaults to None.
        """

        if device is None:
            self.device = general_utils.device_selector()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)

    def tokenize_text(self, text: str, **kwargs):
        """tokenize a single sentence

        Args:
            text (str): text

        Returns:
            dict: tokenized input, w/ input_id and attention_mask both in PyTorch tensor format
        """
        return self.tokenizer(text, return_tensors='pt', **kwargs)
    

    def __get_model_out(self, prompt):
        """get the model output on hatespeech detection task (label pos: 1)

        Args:
            prompt (dict): dict containing output of the tokenizer (input_ids and attention mask)

        Returns:
            int: hatefulness score
        """
        with torch.no_grad():
            out = self.model(**prompt)

        # apply softmax and keep only hate score (0 = not hateful)
        out = torch.nn.functional.softmax(out['logits'].detach(), dim = 1).cpu().numpy()[:, 1]
        return out
    

    def get_score(self, prompt: dict, response: dict):
        """Get toxicity score for a single prompt, response

        Args:
            prompt (dict): tokenized prompt
            response (dict): tokenized response

        Returns:
            tuple: tuple containing the toxicity score of the prompt and the response  
        """
        # to device
        for ele1, ele2 in zip(prompt, response):
                prompt[ele1] = prompt[ele1].to(self.device)
                response[ele2] = response[ele2].to(self.device)
        self.model.to(self.device)
        
        output_prompt = self.__get_model_out(prompt)
        output_response = self.__get_model_out(response)

        return (output_prompt, output_response)

    
    def get_batch_score(self, model_loader: torch.utils.data.DataLoader):
        """Get the toxicity score of the prompt and the response using batched input

        Args:
            model_loader (torch.utils.data.DataLoader): DataLoader returning id, prompt and response

        Returns:
            List: list of (toxicity) scores computed on the model_loader
        """
        result_tox = {
            'prompt_score': [],
            'response_score': [],
        }

        for _, prompt, response in tqdm(model_loader):
            for ele1, ele2 in zip(prompt, response):
                prompt[ele1] = prompt[ele1].to(self.device)
                response[ele2] = response[ele2].to(self.device)
            
            self.model.to(self.device)
            output_prompt = self.__get_model_out(prompt)
            output_response = self.__get_model_out(response)

            result_tox['prompt_score'].extend(output_prompt.tolist())
            result_tox['response_score'].extend(output_response.tolist())

        return result_tox
