from torch.utils.data import Dataset

class PromptsDataset(Dataset):
    def __init__(self, df, tokenizer, text_col = 'text', max_len = 256, custom_prompt = '{prompt}'):
        
        assert '{prompt}' in custom_prompt

        tokenizer.padding_side = "left" 
        tokenizer.pad_token = tokenizer.eos_token

        # adjusting prompt based on custom_prompt parameter
        adj_prompt = list(map(
            lambda s: custom_prompt.format(prompt = s), 
            df[text_col].to_list())
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
    


# dataset containing the pair (prompt, response) to measure the toxicity
class ToxicityGeneratedSet(Dataset):
    def __init__(self, df, tokenizer, max_len = 512):
        
        self.prompts = tokenizer(
            df['prompts'].to_list(), 
            padding = 'max_length', 
            max_length = max_len, 
            truncation = True,
            return_tensors = "pt",
        )
        self.responses = tokenizer(
            df['responses'].to_list(), 
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