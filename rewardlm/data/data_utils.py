import pandas as pd
from datasets import load_dataset
from torch.utils.data import DataLoader

from ..data.CustomDatasets import PromptsDataset


def download_DIALOCONAN():
    CSV_URL = 'https://raw.githubusercontent.com/marcoguerini/CONAN/master/DIALOCONAN/DIALOCONAN.csv'
    return pd.read_csv(CSV_URL)


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


def gen_benchmark_data(
        tokenizer, 
        max_len: int = 128,
        custom_prompt: str = '{prompt}',
        batch_size: int = 8,
    ):
    """Generate PyTorch DataLoader based on the given parameter using RealToxicityPrompt as benchmark dataset. 
    Prompts can also be customized using custom_prompt parameter

    Args:
        tokenizer (transformers.AutoTokenizer): tokenizer of the generative model
        max_len (int, optional): max length of a single sentence when tokenizing. Defaults to 128.
        custom_prompt (str, optional): format string where '{prompt}' is the original prompt. Defaults to '{prompt}'.
        batch_size (int, optional): batch size dimension. Defaults to 8.

    Returns:
        torch.utils.data.DataLoader: PyTorch DataLoader containing all the prompts
    """
    prompts = __get_real_toxicity_prompts()


    model_set = PromptsDataset(
        text = prompts['text'].to_list(),
        tokenizer = tokenizer,
        max_len = max_len,
        custom_prompt = custom_prompt,
    )
    model_loader = DataLoader(model_set, batch_size = batch_size)
    
    return model_loader

