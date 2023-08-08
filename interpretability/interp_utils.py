import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from inseq import FeatureAttributionOutput

# plot color palette
palette = ['#2D3047', '#576490', '#7796CB', '#A3BCF9', '#D1D2F9', '#C9CAD9', '#B7B8C5', '#FCB97D']
diversity_palette = ['#2D3047', '#7796CB', '#FCB97D', '#E54F6D', '#79C99E', '#A9F0D1',]


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


def _get_offsets_ci(dependencies: np.array, z = 1.96):
    r"""Return confidence interval offsets for dependencies

    Args:
        dependencies (`np.array`): dependencies of shape `(n, n_tokens)` where `n` is the number of records.
        z (`float`, optional): Z parameter, (`1.96` for 95%, `2.58` for 99%). Defaults to `1.96`.

    Returns:
        `np.array`: array of offsets of shape `(1, n_tokens)`
    """
    offsets = np.nanstd(dependencies, axis = 0) / np.sqrt(np.invert(np.isnan(dependencies)).sum(axis = 0))
    offsets *= z
    return offsets


def get_prompt_dependancy(attributions: FeatureAttributionOutput, max_n_tok: int = 50,):
    r"""Compute prompt token level dependancy for each generation in attributions
    Notice: generation dependancy can be calculated as `1 - prompt_dependancy`.

    Args:
        `attributions` (FeatureAttributionOutput): attributions used to compute the prompt token level dependancy
        `max_n_tok` (int, optional): number of maximum generated tokens where dependancy is computed. Defaults to 50.
        
    Returns:
        `np.array`: numpy array with `[len(attributions), max_n_tok]` shape.
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


def get_plot_prompt_dep_toxicity(dependancies: np.array, attr_labels: np.array, model_name: str, fig_kwargs: dict = None):
    r"""Plot average prompt dependancy, distinguishing between the bucket in `attr_labels`.

    Args:
        dependancies (`np.array`): prompt dependancies obtained from `get_prompt_dependancy` fun.
        attr_labels (`np.array`): array with shape `(n, 1)` where `n` is the number of attributions.
        model_name (`str`): name of the model (title).
        fig_kwargs (`dict`): figure kwargs. Defaults to None.

    Returns:
        `matplotlib.pyplot`: same as description.
    """ 

    local_palette = diversity_palette[3], diversity_palette[4], diversity_palette[1]
    
    if attr_labels[0]:
        assert dependancies.shape[0] == len(attr_labels), f"Number of dependancies (0 dim of dependancies: {dependancies.shape[0]}) must be equal to the number of given labels ({len(attr_labels)})"

    if not fig_kwargs:
        fig_kwargs = {
            'figsize': (9, 6),
            'dpi': 250,
        }

    fig, ax = plt.subplots(**fig_kwargs)
    ax.set_title(f'[{model_name}], Generation prompt dependancy, avg per toxicity level')

    for i, (lbl, color) in enumerate(zip(np.unique(attr_labels), local_palette)):
        group = dependancies[(attr_labels == np.unique(attr_labels))[:, i]]

        avgs = np.nanmean(group, axis = 0)
        
        offsets = _get_offsets_ci(group)
        ax.plot(
            np.arange(0, len(avgs)),
            avgs,
            label = lbl,
            color = color,
        )
        ax.fill_between(
            np.arange(0, len(avgs)),
            (avgs - offsets),
            (avgs + offsets),
            color = color, alpha = .08,
        )

    ax.set_ylim(0.19, 1.1)
    ax.legend()
    ax.set_xlabel(r'$n$ generated tokens')
    ax.set_ylabel('prompt dependancy (sum)')
    ax.grid(alpha = .3, linestyle = ':')
    return plt




def get_plot_training_compare(dependencies: dict, model_name:str, fig_kwargs: dict = None):
    """Plot prompt dependancy comparing different type of training, Pre-Trained, Fine-Tuned and Reinforcement Learning.

    Args:
        dependencies (`np.array`): prompt dependancies obtained from `get_prompt_dependancy` fun.
        model_name (`str`): name of the model (title).
        fig_kwargs (`dict`): figure kwargs. Defaults to None.

    Returns:
        `matplotlib.pyplot`: same as description.
    """
    local_palette = diversity_palette[1], diversity_palette[4], diversity_palette[2], diversity_palette[3]
    
    # check
    exp_keys = ['PT', 'FT', 'RL']
    # assert set(exp_keys) == set(attribution.keys()), f'dependencies must be a dict with {exp_keys} keys. Got {list(attribution.keys())}'

    if not fig_kwargs:
        fig_kwargs = {
            'figsize': (9, 6),
            'dpi': 250,
        }
    
    fig, ax = plt.subplots(**fig_kwargs)
    ax.set_title(f'[{model_name}], Prompt dependancy, PT vs FT vs RL')

    for k, color in zip(dependencies, local_palette):
        avgs = np.nanmean(dependencies[k], axis = 0)
        offsets = _get_offsets_ci(dependencies[k])

        ax.plot(
            np.arange(0, len(avgs)),
            avgs,
            label = k,
            color = color
        )
        ax.fill_between(
            np.arange(0, len(avgs)),
            (avgs - offsets),
            (avgs + offsets),
            color = color, alpha = .15,
        )
    
    ax.set_ylim(0.19, 1.1)
    ax.set_xlabel(r'$n$ generated tokens')
    ax.set_ylabel('prompt dependancy (sum)')
    ax.legend()
    ax.grid(alpha = .3, linestyle = ':')
    
    return plt


def get_plot_toxlev2toxlev(deps, lbls, start: str, end: str, model_name: str, fig_kwargs: dict = None):
    # assert start in lbls_1, f'start ({start}) must be in lbls_1'
    # assert end in lbls_2, f'end ({end}) must be in lbls_2'
    
    assert deps.keys() == lbls.keys(), f'deps ({deps.keys()}) and lbls ({lbls.keys()}) not having the same keys.'
    assert len(deps.keys()) == 2, f'deps and lbls must have {2} keys ({len(deps.keys())} were given).'

    local_palette = diversity_palette[1], diversity_palette[4], diversity_palette[2], diversity_palette[3]

    if not fig_kwargs:
        fig_kwargs = {
            'figsize': (9, 6),
            'dpi': 250,
        }
    
    fig, ax = plt.subplots(**fig_kwargs)

    first_key, second_key = lbls.keys()
    if lbls[first_key].shape != lbls[second_key].shape:
        warnings.warn(f'Labels not matching in shape ({lbls[first_key].shape} and {lbls[second_key].shape}). Using the lowest dimension between the two, cutting away the end of the biggest label set.')
        lowest_dim = min(
            lbls[first_key].shape[0],
            lbls[second_key].shape[0],
        )
        lbls[first_key], lbls[second_key] = lbls[first_key][:lowest_dim], lbls[second_key][:lowest_dim]
        deps[first_key], deps[second_key] = deps[first_key][:lowest_dim], deps[second_key][:lowest_dim]
    indexes = ((lbls[first_key] == start) & (lbls[second_key] == end)).flatten()
    
    if not indexes.sum() > 0:
        print(f'No instance of {start}2{end} found.')

    ax.set_title(f'[{model_name}], Prompt dependancy, toxicity: {first_key}.{start} -> {second_key}.{end}')

    ## first line (start)
    d = deps[first_key][indexes]
    avgs = np.nanmean(d, axis = 0)
    ax.plot(
        np.arange(0, len(avgs)),
        avgs,
        label = f'{first_key} {start} toxicity',
        color = local_palette[0],
    )

    offsets = _get_offsets_ci(d)
    ax.fill_between(
        np.arange(0, len(avgs)),
        (avgs - offsets),
        (avgs + offsets),
        color = local_palette[0], alpha = .15,
    )

    ## second line (end)
    d = deps[second_key][indexes]
    avgs = np.nanmean(d, axis = 0)
    ax.plot(
        np.arange(0, len(avgs)),
        avgs,
        label = f'{second_key} {end} toxicity',
        color = local_palette[1],
    )

    offsets = _get_offsets_ci(d)
    ax.fill_between(
        np.arange(0, len(avgs)),
        (avgs - offsets),
        (avgs + offsets),
        color = local_palette[0], alpha = .15,
    )

    ax.set_ylim(0.19, 1.1)
    ax.set_xlabel(r'$n$ generated tokens')
    ax.set_ylabel('prompt dependancy (sum)')
    ax.legend()
    ax.grid(alpha = .3, linestyle = ':')

    return plt
