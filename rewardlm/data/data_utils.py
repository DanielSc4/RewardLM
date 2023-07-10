import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset

from torch.utils.data import DataLoader

from ..data.CustomDatasets import PromptsDataset


def get_real_toxicity_prompts(toxicity_threshold: float = .5):
    """downloads 'real-toxicity-prompts' dataset from hugging face and selects only the challenging prompts

    Args:
        - `toxicity_threshold`: Returns only prompts with a toxicity value > `toxicity_threshold` (prompt AND continuation). Defaults to 0.5
    Returns:
        pd.DataFrame: subset of real-toxicity-prompts containing only challenging prompts
    """
    #### OLD:
    # df = pd.DataFrame(
    #     load_dataset("allenai/real-toxicity-prompts", split = 'train')
    # )
    # # selecting only the challenging prompts
    # query = df['challenging'] == True
    # c_prompts = pd.DataFrame(df[query]['prompt'].to_list())
    
    # return c_prompts
    dataset = load_dataset("allenai/real-toxicity-prompts", split = 'train')

    # selecting only toxic prompts
    def get_toxic(args):
        idx, data_point = args
        if data_point['prompt']['toxicity'] and data_point['continuation']['toxicity']:     # avoid None type
            if data_point['prompt']['toxicity'] > toxicity_threshold and data_point['continuation']['toxicity'] > toxicity_threshold:
                return idx

    res = list(map(get_toxic, enumerate(dataset)))
    res = set(res)      # drop all None
    res.discard(None)

    new_data = []
    for prompts, continuations in zip(dataset[res]['prompt'], dataset[res]['continuation']):
        new_data.append(prompts['text'] + ' ' + continuations['text'])
    
    return new_data


def download_DIALOCONAN():
    CSV_URL = 'https://raw.githubusercontent.com/marcoguerini/CONAN/master/DIALOCONAN/DIALOCONAN.csv'
    return pd.read_csv(CSV_URL)


def download_CONAN():
    CSV_URL = 'https://raw.githubusercontent.com/marcoguerini/CONAN/master/CONAN/CONAN.csv'
    return pd.read_csv(CSV_URL)


def get_CONAN_prepro(
        delete_last_assistant_response = False,
        user_name: str = 'User:',
        bot_name: str = 'Assistant:',
):
    dataset = download_CONAN()
    print('Not yet implemented')



def get_DIALOCONAN_prepro(
        return_text_only = True, 
        delete_last_assistant_response = False,
        user_name: str = 'User:',
        bot_name: str = 'Assistant:',
    ):
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
        clean_data = dataset[dataset['dialogue_id'] == idx]['text'].apply(lambda x: x.replace('\n', ' '))

        old_pair = ''
        for i, (ele1, ele2) in enumerate(_pairwise(clean_data)):
            new_df[idx][i] = old_pair + f'{user_name} {ele1}.\n{bot_name} {ele2 if not delete_last_assistant_response else ""}'
            old_pair += f'{user_name} {ele1}.\n{bot_name} {ele2}\n'

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


def get_dataset_CLM(data, tokenizer, context_length = 512, custom_prompt = '{prompt}', train_on_inputs: bool = False):
    """Generate a HuggingFace Dataset tokenizing the given prompts

    Args:
        data (list[str]): list of data points (str)
        tokenizer (_type_): Hugging Face tokenizer
        context_length (int, optional): Max len of the context. Defaults to 512.
        train_on_inputs (bool, optional): if True input promt contribute in loss . Defaults to False.

    Returns:
        datasets.Dataset: Dataset containing input_ids, attention_mask and labels for CLM task
    """
    assert '{prompt}' in custom_prompt, 'custom_prompt must contain \'{prompt}\' ' + f'{custom_prompt} given.'

    raw_dataset = Dataset.from_list([{'text': text} for text in data])

    def _tokenize(element, add_eos_token=True):
        """tokenize a single element of the dataset
        """
        tokenized = tokenizer(
            element["text"],
            truncation=True,
            max_length=context_length,
            padding = False,
            return_tensors = None,
        )
        if (
            tokenized['input_ids'][-1] != tokenizer.eos_token_id        # encoded not ends with eos token
            and len(tokenized["input_ids"]) < context_length            # input is shorter than context
            and add_eos_token                                           # allowed to append eos token at the end
        ):
            tokenized['input_ids'].append(tokenizer.eos_token_id)
            tokenized['attention_mask'].append(1)
        
        # CLM task requires labels == input_ids (next token pred.)
        tokenized['labels'] = tokenized['input_ids'].copy()
        
        return tokenized
    
    tokenized_datasets = raw_dataset.map(_tokenize, remove_columns=raw_dataset.column_names)

    return tokenized_datasets