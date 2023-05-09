import torch
from datasets import load_dataset
import pandas as pd
import transformers
from torch.utils.data import DataLoader

from utils.CustomDatasets import PromptsDataset

def device_selector():
    """_summary_

    Returns:
        str: device type available [cuda, mps, cpu]
    """
    # selecting device based on whats available
    device = ''
    if torch.cuda.is_available():
        print('Using Nvidia GPU')
        print(torch.cuda.get_device_name())
        device = 'cuda'
    # Only on torch night for Apple M1 GPU
    elif torch.backends.mps.is_available():
        print('Using MPS (Apple Silicon GPU)')
        device = 'mps'
    else:
        print('Using CPU, :(')
        device = 'cpu'
    return device


def __get_real_toxicity_prompts():
    """downloads 'real-toxicity-prompts' dataset from hugging face and selects only the challenging prompts

    Returns:
        pd.DataFrame: subset of real-toxicity-prompts containing only challenging prompts
    """
    df = pd.DataFrame(
        load_dataset("allenai/real-toxicity-prompts", split = 'train')
    )
    # selecting only the challenging prompts
    query = df['challenging'] == True
    c_prompts = pd.DataFrame(df[query]['prompt'].to_list())
    
    return c_prompts


def gen_data(
        tokenizer: transformers.AutoTokenizer, 
        max_len: int = 128,
        custom_prompt: str = '{prompt}',
        batch_size: int = 8,
    ):
    """generate PyTorch DataLoader based on the given parameter. Prompts can also be customized using custom_prompt parameter

    Args:
        tokenizer (transformers.AutoTokenizer): tokenizer for the generative model
        max_len (int, optional): max length of a single sentence when tokenizing. Defaults to 128.
        custom_prompt (str, optional): format string where '{prompt}' is the original prompt. Defaults to '{prompt}'.
        batch_size (int, optional): batch size dimension. Defaults to 8.

    Returns:
        torch.utils.data.DataLoader: PyTorch DataLoader containing all the prompts
    """
    prompts = __get_real_toxicity_prompts()

    model_set = PromptsDataset(
        df = prompts,
        tokenizer = tokenizer,
        text_col = 'text', 
        max_len = max_len,
        custom_prompt = custom_prompt
    )
    model_loader = DataLoader(model_set, batch_size = batch_size)
    return model_loader

