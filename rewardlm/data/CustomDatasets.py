import torch
from torch.utils.data import Dataset

class PromptsDataset(Dataset):
    def __init__(self, tokenizer, text: list[str], max_len = 256, custom_prompt = '{prompt}'):
        """Dataset returning a tokenized prompt w/ left padding

        Args:
            - `tokenizer` (`*Tokenizer`): tokenizer of the generative model
            - `text` (`list[str]`): List of strings (prompts)
            - `max_len` (`int`, optional): max lenght for the tokenizer. Defaults to 256.
            - `custom_prompt` (`str`, optional): format string containing '{prompt}' to modify the original prompt. Defaults to '{prompt}'.
        """
        assert '{prompt}' in custom_prompt

        tokenizer.padding_side = "left" 
        tokenizer.pad_token = tokenizer.eos_token

        # adjusting prompt based on custom_prompt parameter
        adj_prompt = list(map(
            lambda s: custom_prompt.format(prompt = s), 
            text)
        )

        self.text = tokenizer(
            adj_prompt, 
            padding = 'max_length', 
            max_length = max_len, 
            truncation = True,
            return_tensors = "pt",
        )

    def __len__(self,):
        return len(self.text['input_ids'])
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.text['input_ids'][idx],
            # 'token_type_ids': self.text['token_type_ids'][idx],
            'attention_mask': self.text['attention_mask'][idx],
        }

class PromptDataset_CLM(Dataset):
    def __init__(self, text, tokenizer, custom_prompt = '{prompt}') -> None:
        assert '{prompt}' in custom_prompt

        self.tokenizer = tokenizer

        # adjusting prompt based on custom_prompt parameter
        adj_prompt = list(map(
            lambda s: custom_prompt.format(prompt = s), 
            text)
        )

        self.tokenized = self.tokenizer(
            adj_prompt, 
            return_tensors = 'pt', 
            truncation = True, 
            padding = True,
        )

        # self.tokenized = list(map(
        #     lambda samples: self.tokenizer(
        #         samples, 
        #         return_tensors = 'pt', 
        #         truncation=True, 
        #         padding=True,
        #     ), 
        #     adj_prompt,
        # ))
    
    def __len__(self,):
        return len(self.tokenized)
    
    def __getitem__(self, index):
        item = {key: torch.as_tensor(val[index]) for key, val in self.tokenized.items()}
        return item



# dataset containing the pair (prompt, response) to measure the toxicity
class ToxicityGeneratedSet(Dataset):
    """PyTorch dataset used to measure toxicity on both the prompt and the response of a model. 
    Tokenizes both input and responses (using the reward model tokenizer) and returns the index with
    the tokenized prompt and response

    Args:
        - `prompts` (`list[str]`): List of prompts
        - `responses` (`list[str]`): List of model response to each prompt
        - `tokenizer` (`*Tokenizer`): Reward model tokenizer
        - `max_len` (`int`, optional): Tokenizer max length. Defaults to 512.

    """
    def __init__(self, prompts, responses, tokenizer, max_len = 512):
        
        self.prompts = tokenizer(
            prompts, 
            padding = 'max_length', 
            max_length = max_len, 
            truncation = True,
            return_tensors = "pt",
        )
        self.responses = tokenizer(
            responses, 
            padding = 'max_length', 
            max_length = max_len, 
            truncation = True,
            return_tensors = "pt",
        )

    def __len__(self,):
        """_summary_

        Returns:
            int: length of the dataset
        """
        return len(self.prompts['input_ids'])
    
    def __getitem__(self, idx):
        """returns a pair of tokenized prompt and response

        Args:
            idx (Int): index of the selected (prompt, response)

        Returns:
            tuple: idx and pair of prompt and response, each one being a dict containing input_ids and attention_mask
        """
        return (
            idx,
            {'input_ids': self.prompts['input_ids'][idx], 'attention_mask': self.prompts['attention_mask'][idx]},
            {'input_ids': self.responses['input_ids'][idx], 'attention_mask': self.responses['attention_mask'][idx]},
        )