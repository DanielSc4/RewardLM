import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from utils.general_utils import device_selector

class RewardModel:
    def __init__(self, model_id, device: str) -> None:

        if device is None:
            self.device = device_selector()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
    
    def get_score(self, model_loader: torch.utils.data.DataLoader):
        result_tox = {
            'prompt_score': [],
            'response_score': [],
        }

        for _, prompt, response in tqdm(model_loader):
            for ele1, ele2 in zip(prompt, response):
                prompt[ele1] = prompt[ele1].to(self.device)
                response[ele2] = response[ele2].to(self.device)
            
            self.model.to(self.device)
            with torch.no_grad():
                output_prompt = self.model(**prompt)
                output_response = self.model(**response)

            # apply softmax and selecting only scores [1]
            out_soft_prompt = torch.nn.functional.softmax(output_prompt[0].detach(), dim = 1).cpu().numpy()[:, 1]
            out_soft_response = torch.nn.functional.softmax(output_response[0].detach(), dim = 1).cpu().numpy()[:, 1]

            result_tox['prompt_score'].extend(out_soft_prompt.tolist())
            result_tox['response_score'].extend(out_soft_response.tolist())

        return result_tox
