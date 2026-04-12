from scipy.stats import kendalltau, spearmanr
from itertools import product
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
import pandas as pd
from scaling.utils import (
    get_pareto_frontier,
    get_final_points_from_curve_set,
    fit_linear_model,
    functional_form_chin3,
    fit_parametric_form,
    fit_parametric_form_stable,
    functional_form_chin3_stable,
)
from scaling.visualize import visualize_train_curves, plot_line_fit, plot_isoflops
import argparse

def load_and_process_hpo_results(df, x_axis = "flops", correlation_type='spearmanr') -> pd.DataFrame:
    assert x_axis in ['flops', 'percent']
    df = df.copy()
    if correlation_type == 'spearmanr':
        correlation_function = spearmanr
    elif correlation_type == "kendalltau":
        correlation_function = kendalltau
    else:
        raise ValueError(f"Unknown correlation: {correlation_type}")


    df['max_flops'] = df['curve_flops'].map(lambda x: np.nanmax(x))
    df['min_flops'] = df['curve_flops'].map(lambda x: np.nanmin(x))
    df['curve_percent'] = df['curve_flops'].map(lambda x: x / x.max() * 100)

    if x_axis == 'flops':
        intervals = np.linspace(df['min_flops'].max(), df['max_flops'].min(), 200)
    elif x_axis == 'percent':
        intervals = np.linspace(0, 100, 200)[1:]
    else:
        raise ValueError(f"Unknown x_axis: {x_axis}")

    df['final_val_loss'] = df['curve_val'].map(lambda x: [x[-1],]*len(intervals))

    values = []
    target_values = []
    for row in df.itertuples(index=False):
        _df = pd.DataFrame({
            'val': row.curve_val,
        }, index=row.curve_percent if x_axis == "percent" else row.curve_flops).dropna()
        _df = _df.reindex(_df.index.union(intervals)).sort_index()
        _df['val'] = _df['val'].interpolate(method='linear')
        _df = _df.loc[intervals]
        values.append(_df['val'].values)
        target_values.append(row.final_val_loss)
    values = np.array(values)
    target_values = np.array(target_values)

    # apply correlation function column-wise
    correlations = []
    for i in range(values.shape[1]):
        correlations.append(correlation_function(values[:, i], target_values[:, i]).correlation)
    return pd.DataFrame({
        "intervals": intervals,
        "curve_correlation": correlations,
    })

def preprocess(df):
    width_scale_to_params = {
        0: 0.1e6,
        1: 0.1e6,
        2: 0.2e6,
        3: 0.4e6,
        4: 0.9e6,
        5: 1.8e6,
        6: 3.5e6,
        7: 7.0e6,
        8: 14.2e6,
        9: 25.2e6,
        10: 100.7e6,
    }
    width_scale_to_hidden_dim = {
        0: 48,
        1: 68,
        2: 96,
        3: 136,
        4: 192,
        5: 272,
        6: 384,
        7: 540,
        8: 768,
        9: 1024,
        10: 2048,
    }

    # create base_N and target_N columns
    df['base_N'] = df['base_scale'].map(lambda x: width_scale_to_hidden_dim.get(x, np.nan))
    df['target_N'] = df['target_scale'].map(lambda x: width_scale_to_hidden_dim.get(x, np.nan))
    df['max_flops'] = df['curve_flops'].map(lambda x: x[-1])
    df['val_loss'] = df['curve_val'].map(lambda x: x[-1])
    return df

def plot_method(axes, _method_df, target_scales, method_name):
    for j, target_scale in enumerate(target_scales):
        method_df = _method_df[_method_df['target_scale']==target_scale]
        if len(method_df) == 0:
            continue
        print(f"Method: {method_name}, Target: {target_scale}, Num Configs: {len(method_df)}")
        hpo_df = load_and_process_hpo_results(method_df)

        axes[j].plot(hpo_df['intervals'], hpo_df['curve_correlation'], label=method_name)
        axes[j].axhline(y=1, color='black', linestyle=':', alpha=.4)

        axes[j].set_title(f"Target: {target_scale}")
        axes[j].legend()

def base_target_correlation(base_df, target_df, hyperparameter_columns):
    merged_df = pd.merge(
        base_df,
        target_df,
        on=hyperparameter_columns,
        suffixes=('_base', '_target')
    )
    assert len(merged_df) == len(target_df), "Merged dataframe length does not match base dataframe length"


    corr = spearmanr(
        merged_df[f"val_loss_base"],
        merged_df[f"val_loss_target"]
    ).correlation
    return corr

if __name__ == "__main__":
    # setup argsparse with file name as single argument

    mlp_df = pd.read_parquet(
        "../data/runs_more.parquet",
    )
    ws_df = pd.read_parquet(
        "../data/runs_ws_grid.parquet",
    )
    base_df = pd.read_parquet(
        "../data/runs_all.parquet",
    )
    mlp_df = preprocess(mlp_df)
    ws_df = preprocess(ws_df)
    base_df = preprocess(base_df)

    hyperparameter_columns = ['cfg_lr', 'cfg_batch_size', 'cfg_weight_decay', 'cfg_warmup_fraction', 'cfg_cooldown_fraction']
    assert all(col in mlp_df.columns for col in hyperparameter_columns), "Not all hyperparameter columns are in mlp_df"
    assert all(col in ws_df.columns for col in hyperparameter_columns), "Not all hyperparameter columns are in ws_df"
    assert all(col in base_df.columns for col in hyperparameter_columns), "Not all hyperparameter columns are in base_df"
    # lr and weight decay
    default_cooldown = 0.2
    default_warmup = 0.01
    default_batch_size = 256

    mlp_df = mlp_df[mlp_df['cfg_cooldown_fraction'] == default_cooldown]
    mlp_df = mlp_df[mlp_df['cfg_warmup_fraction'] == default_warmup]
    mlp_df = mlp_df[mlp_df['cfg_batch_size'] == default_batch_size]

    ws_df = ws_df[ws_df['cfg_cooldown_fraction'] == default_cooldown]
    ws_df = ws_df[ws_df['cfg_warmup_fraction'] == default_warmup]
    ws_df = ws_df[ws_df['cfg_batch_size'] == default_batch_size]

    base_df = base_df[base_df['cfg_cooldown_fraction'] == default_cooldown]
    base_df = base_df[base_df['cfg_warmup_fraction'] == default_warmup]
    base_df = base_df[base_df['cfg_batch_size'] == default_batch_size]

    # hyperparameter_columns = ['cfg_warmup_fraction', 'cfg_cooldown_fraction']
    base_df = base_df[base_df['scale_id'] == 's0']
    base_df = base_df[base_df['scaling'] == 'width']
    base_df = base_df[base_df['tkpm_group'] == 20.]
    ws_df = ws_df[ws_df['base_scale'] == 0]


    # check if hyperparameter columns are unique in base_df
    assert len(base_df) == len(base_df.drop_duplicates(subset=hyperparameter_columns)), "Hyperparameter columns are not unique in base_df"

    # base_scales = sorted(mlp_df['base_scale'].unique())

    target_scales = sorted(mlp_df['target_scale'].unique())
    fig, axes = plt.subplots(1, len(target_scales), figsize=(5 * len(target_scales), 5), layout='constrained')

    rows = []
    for method, df in [('scratch', mlp_df), ('net2net', ws_df), ('snp', ws_df)]:
        _method_df = df[df['method']==method]
        plot_method(axes, _method_df, target_scales, method)

        for target_scale in target_scales:
            target_df = _method_df[_method_df['target_scale']==target_scale]
            correlation = base_target_correlation(base_df, target_df, hyperparameter_columns)
            rows.append((method, target_scale, correlation))


    # pivot table
    corr_df = pd.DataFrame(rows, columns=['method', 'target_scale', 'correlation'])
    corr_pivot = corr_df.pivot(index='target_scale', columns='method', values='correlation')
    print(corr_pivot)
    plt.show()




