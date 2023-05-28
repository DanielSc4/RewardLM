import pandas as pd
import numpy as np
from datasets import load_dataset

from torch.utils.data import DataLoader

from ..data.CustomDatasets import PromptsDataset


def get_real_toxicity_prompts():
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


def download_DIALOCONAN():
    CSV_URL = 'https://raw.githubusercontent.com/marcoguerini/CONAN/master/DIALOCONAN/DIALOCONAN.csv'
    return pd.read_csv(CSV_URL)


def get_DIALOCONAN_for_finetune(return_text_only = True):
    """Download DIALOCONAN dataset and adapt it to fine-tuning process

    Args:
        return_text_only (bool, optional): if False return (dict) having dialog_id as id. True returns a list of text. Defaults to True.

    Returns:
        dict | list: check return_text_only arg
    """

    dataset = download_DIALOCONAN()
    
    def _pairwise(iterable):
        a = iter(iterable)
        return zip(a, a)
    
    new_df = {}
    for idx in np.unique(dataset['dialogue_id']):
        new_df[idx] = {}
        for i, (u_text, a_text) in enumerate(_pairwise(dataset[dataset['dialogue_id'] == idx]['text'])):
            if i == 0:
                new_df[idx][i] = "User: {u_text}\nAssistant: {a_text}".format(
                    u_text = u_text.replace('\n', ' '), 
                    a_text = a_text.replace('\n', ' ')
                )
            else:
                new_df[idx][i] = new_df[idx][i - 1] + '\n' + "User: {u_text}\nAssistant: {a_text}".format(
                    u_text = u_text.replace('\n', ' '), 
                    a_text = a_text.replace('\n', ' ')
                )
    if return_text_only:
        all_text = []
        for dialog_id in new_df:
            for num_ in new_df[dialog_id]:
                all_text.append(new_df[dialog_id][num_])
        return all_text
    else:
        return new_df



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
    prompts = get_real_toxicity_prompts()


    model_set = PromptsDataset(
        text = prompts['text'].to_list(),
        tokenizer = tokenizer,
        max_len = max_len,
        custom_prompt = custom_prompt,
    )
    model_loader = DataLoader(model_set, batch_size = batch_size)
    
    return model_loader


def gen_loader(
        tokenizer,
        text: list[str],
        max_len: int = 256,
        custom_prompt: str = '{prompt}',
        batch_size: int = 8,
):    

    model_set = PromptsDataset(
        text = text,
        tokenizer = tokenizer,
        max_len = max_len,
        custom_prompt = custom_prompt,
    )

    model_loader = DataLoader(model_set, batch_size = batch_size)

    return model_loader
