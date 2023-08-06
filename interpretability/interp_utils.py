import pandas as pd
import numpy as np

from inseq import FeatureAttributionOutput



def _assign_label(value):
    if value < 0.333:
        return 'low'
    if value >= 0.333 and value <= 0.666:
        return 'mid'
    else:
        return 'high'
    

def stratify_df(df: pd.DataFrame, subset_size):
    # selecting stratified
    c = df['label'].value_counts().apply(lambda x: x/len(df))
    stratified_df = pd.concat([
            group.sample(
                int(c[lbl] * subset_size), 
                replace=False, 
                random_state=42,
            ) for lbl, group in df.groupby('label')
    ])
    return stratified_df


def get_prompt_dependancy(attributions: FeatureAttributionOutput, max_n_tok: int = 50):
    r"""Compute prompt token level dependancy for each generation in attributions
    Notice: generation dependancy can be calculated as `1 - prompt_dependancy`.

    Args:
        `attributions` (FeatureAttributionOutput): attributions used to compute the prompt token level dependancy
        `max_n_tok` (int, optional): number of maximum generated tokens where dependancy is computed. Defaults to 50.

    Returns:
        `np.array`: numpy array [len(attributions), max_n_tok]
    """

    final = []
    for attr in attributions:
        if len(attr.target_attributions.shape) != 2:
            # attribution has not been previously aggregated
            attr = attr.aggregate()
        
        attr_matrix = np.nan_to_num(attr.target_attributions)
        final.append((
                np.array([
                    attr_matrix[:len(attr.source), n_tok].sum().item()      # sum the column for each generated token (entrire column (prompt + gen) sums to one)
                    if n_tok < attr_matrix.shape[1] else np.nan             # nan is left where generations length is less than max_n_tok
                    for n_tok in range(max_n_tok)       # each generated token
                ])
            ))
    
    # final = np.nan_to_num(np.vstack(final))

    return np.array(final)
