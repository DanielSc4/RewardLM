import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import colorgetter as cg
import numpy as np
from scipy.stats import entropy

from inseq import FeatureAttributionOutput
from tqdm import tqdm

from typing import Callable

# plot color palette
palette = ['#2D3047', '#576490', '#7796CB', '#A3BCF9', '#D1D2F9', '#C9CAD9', '#B7B8C5', '#FCB97D']
diversity_palette = ['#2D3047', '#7796CB', '#FCB97D', '#E54F6D', '#79C99E', '#A9F0D1',]


def _assign_label(value):
    if value < 0.333:
        return 'low'
    if value >= 0.333 and value <= 0.666:
        return 'mid'
    if value > 0.66 and value <= 1:
        return 'high'
    else:
        return np.nan
    

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
        - `dependencies` (`np.array`): dependencies of shape `(n, n_tokens)` where `n` is the number of records.
        - `z` (`float`, optional): Z parameter, (`1.96` for 95%, `2.58` for 99%). Defaults to `1.96`.

    Returns:
        `np.array`: array of offsets of shape `(1, n_tokens)`
    """
    offsets = np.nanstd(dependencies, axis = 0) / np.sqrt(np.invert(np.isnan(dependencies)).sum(axis = 0))
    offsets *= z
    return offsets


def get_prompt_dependancy(attributions: FeatureAttributionOutput, max_n_tok: int = 50):
    r"""Compute prompt token level dependancy for each generation in attributions
    Notice: generation dependancy can be calculated as `1 - prompt_dependancy`.

    Args:
        - `attributions` (`FeatureAttributionOutput`): attributions used to compute the prompt token level dependancy
        - `max_n_tok` (`int`, optional): number of maximum generated tokens where dependancy is computed. Defaults to 50.
        
    Returns:
        - `np.array`: numpy array with `[len(attributions), max_n_tok]` shape.
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
                    for n_tok in range(max_n_tok)                           # each generated token
                ])
            ))
    
    # final = np.nan_to_num(np.vstack(final))

    return np.array(final)


def get_plot_prompt_measure_toxicity(
        measurements: np.array, 
        attr_labels: np.array, 
        model_name: str, 
        measure_name: str, 
        fig_kwargs: dict = None, 
        aggregation_fun: Callable = np.nanmean,
    ):
    r"""Plot average prompt dependancy | entropy | any metric with shape `(n_attributions, num_of_tokens)` 
    distinguishing between the bucket in `attr_labels`.

    Args:
        measurements (`numpy.ndarray`): prompt dependancies obtained from `get_prompt_dependancy` fun.
        attr_labels (`numpy.ndarray`): array with shape `(n, 1)` where `n` is the number of attributions.
        model_name (`str`): name of the model (title).
        measure_name(`str`): name of the measure (title).
        fig_kwargs (`dict`): figure kwargs. Defaults to None.
        aggregation_fun (`Callable`): custom function to aggregate results. Defaults to `numpy.nanmean`.

    Returns:
        `matplotlib.pyplot`: same as description.
    """ 

    local_palette = diversity_palette[3], diversity_palette[4], diversity_palette[1]
    
    if attr_labels[0]:
        assert measurements.shape[0] == len(attr_labels), f"Number of measurements (0 dim of measurements: {measurements.shape[0]}) must be equal to the number of given labels ({len(attr_labels)})"

    assert isinstance(aggregation_fun, Callable), f'aggregation_fun must be Callable, {type(aggregation_fun)}, given.'

    if not fig_kwargs:
        fig_kwargs = {
            'figsize': (9, 6),
            'dpi': 250,
        }

    fig, ax = plt.subplots(**fig_kwargs)
    ax.set_title(f'[{model_name}], {measure_name}, avg per toxicity level')

    for i, (lbl, color) in enumerate(zip(np.unique(attr_labels), local_palette)):
        group = measurements[(attr_labels == np.unique(attr_labels))[:, i]]

        avgs = aggregation_fun(group, axis = 0)
        
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

    if max(avgs) <= 1.1 and aggregation_fun == np.nanmean:
        ax.set_ylim(0.19, 1.1)
    # else:
    #     ax.set_ylim(
    #         np.floor(min(avgs)),
    #         np.ceil(min(avgs)),
    #     )
    ax.legend()
    ax.set_xlabel(r'$n$ generated tokens')
    ax.set_ylabel('prompt dependancy (sum)')
    ax.grid(alpha = .3, linestyle = ':')
    return fig




def get_plot_training_compare(
        measurements: dict, 
        model_name:str, 
        measure_name: str, 
        fig_kwargs: dict = None, 
        aggregation_fun: Callable = np.nanmean,
    ):
    """Plot prompt dependancy | entropy | any metric with shape `(n_attributions, num_of_tokens)` 
    comparing different type of training, Pre-Trained, Fine-Tuned and Reinforcement Learning.

    Args:
        measurements (`numpy.ndarray`): measurements obtained from `get_prompt_dependancy` | `get_prompt_shannon_entropy` | ... functions.
        model_name (`str`): name of the model (title).
        measure_name(`str`): name of the measure (title).
        fig_kwargs (`dict`): figure kwargs. Defaults to None.
        aggregation_fun (`Callable`): custom function to aggregate results. Defaults to `numpy.nanmean`.

    Returns:
        matplotlib.pyplot: same as description.
    """
    local_palette = diversity_palette[1], diversity_palette[4], diversity_palette[2], diversity_palette[3]
    assert isinstance(aggregation_fun, Callable), f'aggregation_fun must be Callable, {type(aggregation_fun)}, given.'
    
    # check
    exp_keys = ['PT', 'FT', 'RL']
    # assert set(exp_keys) == set(attribution.keys()), f'dependencies must be a dict with {exp_keys} keys. Got {list(attribution.keys())}'

    if not fig_kwargs:
        fig_kwargs = {
            'figsize': (9, 6),
            'dpi': 250,
        }
    
    fig, ax = plt.subplots(**fig_kwargs)
    ax.set_title(f'[{model_name}], {measure_name}, IT vs FT vs RL')

    for k, color in zip(measurements, local_palette):
        avgs = aggregation_fun(measurements[k], axis = 0)
        offsets = _get_offsets_ci(measurements[k])

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
    
    if max(avgs) <= 1.1 and aggregation_fun == np.nanmean:
        ax.set_ylim(0.19, 1.1)
    # else:
    #     ax.set_ylim(
    #         np.floor(min(avgs)),
    #         np.ceil(min(avgs)),
    #     )
    ax.set_xlabel(r'$n$ generated tokens')
    ax.set_ylabel('prompt dependancy (sum)')
    ax.legend()
    ax.grid(alpha = .3, linestyle = ':')
    
    return fig


def get_plot_toxlev2toxlev(
        deps: dict, 
        lbls: dict, 
        from_to: list[tuple[str]], 
        model_name: str, 
        fig_kwargs: dict = None,
        aggregation_fun: Callable = np.nanmean,
    ):
    """Plot the prompt dependancy level comparing two different training methods. 
    The function selects all generation classified as `from_to[_][0]` by the first model
    and generations classified as `from_to[_][1]` by the second model.

    Args:
        deps (`dict`): Dict with 2 keys naming two different models and their prompt dependancy.
        lbls (`dict`): Dict with 2 keys naming two different models and their nominal labels
        from_to (`list[tuple[str]]`): list of tuples. Each tuple contains two string labels.
        model_name (`str`): Name of the model for title.
        fig_kwargs (`dict`, optional): figure kwargs. Defaults to None.
        aggregation_fun (`Callable`): custom function to aggregate results. Defaults to `numpy.nanmean`.

    Returns:
        - `matplotlib.pyplot`: same as description.
    """
    
    
    assert deps.keys() == lbls.keys(), f'deps ({deps.keys()}) and lbls ({lbls.keys()}) not having the same keys.'
    assert len(deps.keys()) == 2, f'deps must have {2} keys ({len(deps.keys())} were given).'
    assert len(lbls.keys()) == 2, f'lbls must have {2} keys ({len(lbls.keys())} were given).'
    
    assert hasattr(from_to, '__iter__'), f'from_to parameter must be a list or iterable of tuples, {type(from_to)} were given.'
    assert all([isinstance(x, tuple) for x in from_to]), f'all of the object in from_to must be tuple, {type(from_to)[0]} were given.'
    assert all([len(x) == 2 for x in from_to]), f'all tuples in from_to must be of len {2}, {len(from_to[0])} were given.'

    first_key, second_key = lbls.keys()
    assert [item in lbls[first_key] for item in np.unique([ele[0] for ele in from_to])], f'found key in from_to not in lbls [0 dim].'
    assert [item in lbls[second_key] for item in np.unique([ele[1] for ele in from_to])], f'found key in from_to not in lbls [1 dim].'

    assert isinstance(aggregation_fun, Callable), f'aggregation_fun must be Callable, {type(aggregation_fun)}, given.'

    local_palette = diversity_palette[1], diversity_palette[4], diversity_palette[2], diversity_palette[3], diversity_palette[5], diversity_palette[0]
    

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
    
    

    ax.set_title(f'[{model_name}] {first_key} -> {second_key}, Prompt dependancy')

    
    for color_s, (start, end) in zip(local_palette, from_to):
        indexes = ((lbls[first_key] == start) & (lbls[second_key] == end)).flatten()
    
        if not indexes.sum() > 0:
            warnings.warn(f'No instance of {start}2{end} found, jumping line plot')
        else: 
            ## first line (start)
            d = deps[first_key][indexes]
            avgs = aggregation_fun(d, axis = 0)
            ax.plot(
                np.arange(0, len(avgs)),
                avgs,
                label = f'{first_key} {start} -> {second_key} {end} ({first_key})',
                color = color_s,
            )

            offsets = _get_offsets_ci(d)
            ax.fill_between(
                np.arange(0, len(avgs)),
                (avgs - offsets),
                (avgs + offsets),
                color = color_s, alpha = .12,
            )

            ## second line (end)
            color_e = cg.rgb_to_hex(     # getting (two times) darker color
                *cg.darker(*cg.darker(
                    *cg.hex_to_rgb(color_s)
                ))
            )
            d = deps[second_key][indexes]
            avgs = aggregation_fun(d, axis = 0)
            ax.plot(
                np.arange(0, len(avgs)),
                avgs,
                label = f'{first_key} {start} -> {second_key} {end} ({second_key})',
                color = color_e,
                ls = '--',
            )

            offsets = _get_offsets_ci(d)
            ax.fill_between(
                np.arange(0, len(avgs)),
                (avgs - offsets),
                (avgs + offsets),
                color = color_e, alpha = .12,
            )

    if max(avgs) <= 1.1 and aggregation_fun == np.nanmean:
        ax.set_ylim(0.19, 1.1)
    # else:
    #     ax.set_ylim(
    #         np.floor(min(avgs)),
    #         np.ceil(min(avgs)),
    #     )
    ax.set_xlabel(r'$n$ generated tokens')
    ax.set_ylabel('prompt dependancy (sum)')
    ax.legend()
    ax.grid(alpha = .3, linestyle = ':')

    return fig




def prompt_kl_divergence(attributions_p, attributions_q, max_n_tok = 50):
    def kl_divergence(p, q):
        assert (np.nansum(p) > 0.9) and np.nansum(p) < 1.1, f'p distribution should sum to 1, now {np.nansum(p)}'
        assert (np.nansum(q) > 0.9) and np.nansum(q) < 1.1, f'q distribution should sum to 1, now {np.nansum(q)}'
        tot_sum = np.nansum([p[i] * np.log2(p[i] / q[i]) for i in range(len(p))])
        return tot_sum

    kls = []
    for attr_p, attr_q in tqdm(zip(attributions_p, attributions_q), total = len(attributions_p)):
        assert len(attr_p.source) == len(attr_q.source), f'Prompts are not the same lenght. Got {len(attr_p.source)} and {len(attr_q.source)}'

        p_prompt_matrix = attr_p.target_attributions[:len(attr_p.source), :]
        q_prompt_matrix = attr_q.target_attributions[:len(attr_q.source), :]

        kls.append(
            np.array([
                kl_divergence(
                    p = p_prompt_matrix[:, n_tok] / np.nansum(p_prompt_matrix[:, n_tok]),
                    q = q_prompt_matrix[:, n_tok] / np.nansum(q_prompt_matrix[:, n_tok]),
                ) if n_tok < min(p_prompt_matrix.shape[1], q_prompt_matrix.shape[1]) else np.nan 
                for n_tok in range(max_n_tok)
            ])
        )

    return np.array(kls)




def get_plot_kl(kls: dict, model_name: str, fig_kwargs: dict = None):

    local_palette = diversity_palette[1], diversity_palette[4], diversity_palette[2], diversity_palette[3]

    if not fig_kwargs:
        fig_kwargs = {
            'figsize': (9, 6),
            'dpi': 250,
        }

    fig, ax = plt.subplots(**fig_kwargs)
    ax.set_title(f'[{model_name}] Measured KL div. between prompt scores')
    
    for k, color in zip(kls, local_palette):
        avgs = np.nanmean(kls[k], axis = 0)
        offsets = _get_offsets_ci(kls[k])

        name, training_type = k.split("_")
        ax.plot(
            np.arange(0, len(avgs)),
            avgs,
            label = f'KL {name}: {training_type}',
            color = color
        )

        ax.fill_between(
            np.arange(0, len(avgs)),
            (avgs - offsets),
            (avgs + offsets),
            color = color, alpha = .15,
        )

    ax.set_ylim(0.0, .6)
    ax.set_xlabel(r'$n$ generated tokens')
    ax.set_ylabel('KL divergence on prompt scores')
    ax.legend()
    ax.grid(alpha = .3, linestyle = ':')
    
    return fig



def get_plot_KL_toxlev2toxlev(kls: dict, lbls: dict, from_to: dict[str, list[tuple[str]]], model_name: str, fig_kwargs: dict[str] = None):
    """Plot prompt KL divergence stored in kls comparing different models with their training methods.

    Args:
        kls (`dict[str, np.ndarray]`): KL divergences. Keys must respect the following format `modelname_traintype1-traintype2` where
            the corresponding value represent the `modelname` KL calculated between its `traintype1` and `traintype2` version.
        lbls (`dict[str, dict[str, np.ndarray]]`): labels where the key follows the patter `lbls[modelname][traintype*]`.
        from_to (`dict[str, list[tuple[str]]]`): dict having the same keys of kls. Values are list of tuples, each tuple
            indicating the indexes to select and the following line to plot based on toxicity level (high, mid, low).
        model_name (`str`): model name for title.
        fig_kwargs (`dict[str]`): figure kwargs. Default to `'figsize': (9, 6)` and `'dpi': 200`.
    """

    # keys format: 'modelname_traintype1-traintype2'
    
    # assert kls.keys() == lbls.keys(), f'KLs ({kls.keys()}) and lbls ({lbls.keys()}) not having the same keys.'
    # assert kls.keys() == from_to.keys(), f'KLs ({kls.keys()}) and from_to ({from_to.keys()}) not having the same keys.'
    
    assert all([k in kls.keys() for k in from_to.keys()]), f'all keys in from_to must be in kls keys. {from_to.keys()} were given.'
    for x in from_to.values():
        for t in x:
            assert isinstance(t, tuple), f'all values in from_to must be a list of tuples, {from_to.values()} were given.'
            assert len(t) == 2, f'all tuples in from_to must be of len {2}, {from_to.values()} were given.'
    
    local_palette = diversity_palette[1], diversity_palette[4], diversity_palette[2], diversity_palette[3], diversity_palette[5], diversity_palette[0]
    
    if not fig_kwargs:
        fig_kwargs = {
            'figsize': (9, 6),
            'dpi': 200,
        }
    fig, ax = plt.subplots(**fig_kwargs)
    ax.set_title(f'[{model_name}], KL divergence, toxicity levels comparasion')

    # for key (KL model PT FT)
        # for group label (low -> high, high -> high)
            # line
    for k in from_to:
        name, training_type = k.split("_")
        # select corresponding keys
        first_key, second_key = training_type.split('-')
        
        # in case PT, FT or RL do not match in shape
        if lbls[name][first_key].shape != lbls[name][second_key].shape:
            lowest_dim = min(
                lbls[name][first_key].shape[0],
                lbls[name][second_key].shape[0],
            )
            warnings.warn(
                f'Labels (and KLs) not matching in shape ({lbls[name][first_key].shape} and {lbls[name][second_key].shape}). Using the lowest dimension between the two ({lowest_dim}), cutting away the end of the biggest label set.'
            )
            lbls[name][first_key], lbls[name][second_key] = lbls[name][first_key][:lowest_dim], lbls[name][second_key][:lowest_dim]
            kls[k] = kls[k][:lowest_dim]

        for color, (start, end) in zip(local_palette, from_to[k]):
            indexes = ((lbls[name][first_key] == start) & (lbls[name][second_key] == end)).flatten()
            if not indexes.sum() > 0:
                warnings.warn(f'No instance of {start}2{end} found, jumping line plot')
            else:
                kl_values = kls[k][indexes]
                avgs = np.nanmean(kl_values, axis=0)
                ax.plot(
                    np.arange(0, len(avgs)),
                    avgs,
                    label = f'KL div. {name}: {start}{first_key} -> {end}{second_key}',
                    color = color,
                )
                offsets = _get_offsets_ci(kl_values)
                ax.fill_between(
                    np.arange(0, len(avgs)),
                    (avgs - offsets),
                    (avgs + offsets),
                    color = color,
                    alpha = .08,
                )
                
    ax.set_xlabel(r'$n$ generated tokens')
    ax.set_ylabel('KL divergence, avg')
    ax.legend()
    ax.grid(alpha = .3, linestyle = ':')

    return fig




def get_prompt_shannon_entropy(attributions: FeatureAttributionOutput, max_n_token: int = 50):
    """Return shannon entropy for each attribution in `attributions`.

    Args:
        attributions (`inseq.FeatureAttributionOutput`): attributions used to compute the prompt entropy accross each generated token.
        max_n_tok (`int`, optional): number of maximum generated tokens where dependancy is computed. Defaults to 50.

    Returns:
        `numpy.ndarray`: entropies numpy array with `[len(attributions), max_n_tok]` shape.
    """
    entropies = []
    for attr in attributions:
        p_attr_matrix = attr.target_attributions[:len(attr.source)]             # .sum(axis = 0)
        entr_per_token = entropy(p_attr_matrix / p_attr_matrix.sum(axis = 0), base=2, axis=0)[:max_n_token]      # limit at max n generated token
        entr_per_token = np.pad(entr_per_token, (0, max_n_token - len(entr_per_token)), 'constant', constant_values=np.nan)
        entropies.append(entr_per_token)
    
    return np.vstack(entropies)

