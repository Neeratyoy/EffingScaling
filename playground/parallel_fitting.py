import argparse
import time
from itertools import product
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scaling.visualize import visualize_train_curves, plot_line_fit, plot_isoflops
from pathlib import Path
from scaling.utils import (
    get_final_points_from_curve_set,
    fit_parametric_form,
    functional_form_chin3,
    fit_parametric_form_parallel
)
from sklearn.metrics import r2_score
from IPython.display import display, Math
from matplotlib.colors import TwoSlopeNorm


def preprocess_warmstarting(df, y_col_to_smooth=None, smoothing_window=100):
    __df = pd.DataFrame()
    for i, x in enumerate(df.groupby(unique_col_list)):
        _df = x[1].sort_values(by="flops")
        # smooth it
        if y_col_to_smooth is not None:
            # +"_smoothed"
            _df[y_col_to_smooth] = _df[y_col_to_smooth].rolling(smoothing_window, win_type='gaussian', min_periods=1).mean(std=smoothing_window / 10)

        # scaling tokens and flops to the max
        max_intended_tokens = (_df.iloc[-1]["target_N"] * _df.iloc[-1]["tkpm"])
        if abs((max_intended_tokens -  _df["tokens"].max()) / _df["tokens"].max()) > 0.01:
            print("Wrong tkpm: ", x[0])
            continue
        _df["tokens"] = np.round(max_intended_tokens / _df["tokens"].max() * _df["tokens"])

        max_intended_flops = 6. * max_intended_tokens * _df["target_N"]
        _df["flops"] = np.round(max_intended_flops / _df["flops"].max() * _df["flops"])

        __df = pd.concat([__df, _df])

    print('Dropping tkpm <= 5')
    __df = __df[__df['tkpm'] > 5.]

    return __df

def preprocess_approach3_data(df, log_transform_y=True):
    N, D, G = df["target_N"].values, df["tokens"].values, df["g"].values
    y = df["Validation Loss"].values

    _df = pd.DataFrame.from_dict({
        "N": N,
        "D": D,
        "G": G,
        "Loss": y
    }).groupby(by=["N", "D", "G"]).min().reset_index()
    _df.sort_values(by=["N", "D", "G"], inplace=True)

    data_X = _df[["N", "D", "G"]].values
    data_y = _df["Loss"].values
    if log_transform_y:
        data_y = np.log(data_y)

    return data_X, data_y

def filter_pairs(df, jump_size=1):
    distinct_values = sorted(
        pd.unique(df[['base_N', 'target_N']].values.ravel())
    )
    pairs = list(zip(distinct_values[:-jump_size], distinct_values[jump_size:]))
    df_filtered = df[df[['base_N', 'target_N']].apply(tuple, axis=1).isin(pairs)]
    return df_filtered

def fit_chin3(data_X, data_y, n_workers = 1, worker_idx=0, dense_grid=False):
    if dense_grid:
        initialization = list(product(
            np.arange(0, 25, 1),
            np.arange(0, 2, 0.1),
            np.arange(0, 25, 1),
            np.arange(0, 2, 0.1),
            np.arange(-1, 1, 0.1)
        ))
    else:
        initialization = list(product(
            np.arange(0, 25, 5),
            np.arange(0, 2, 0.5),
            np.arange(0, 25, 5),
            np.arange(0, 2, 0.5),
            np.arange(-1, 1, 0.5)
        ))
    df = fit_parametric_form_parallel(
        functional_form_chin3,
        data_X[:,:2],
        data_y,
        initialization,
        n_workers=n_workers,
        worker_idx=worker_idx
    )

    return df

if __name__ == "__main__":
    # argparse
    argparser = argparse.ArgumentParser(description='Fit scaling laws in parallel.')
    argparser.add_argument('--n_workers', type=int, default=1, help='Number of parallel workers to use for fitting.')
    argparser.add_argument('--worker_idx', type=int, default=0, help='Index of the current worker (0-based).')
    argparser.add_argument('--dense_grid', action='store_true', help='Whether to use a dense grid for initialization.')
    argparser.add_argument('--output_dir', type=str, help='Directory to save the fitting results.')
    argparser.add_argument('--data_path', type=str, default="../data/warmstarting_results.parquet", help='Path to the warmstarting results parquet file.')
    args = argparser.parse_args()

    unique_col_list = ["base_N", "target_N", "tkpm", "shrink", "method"]
    y_col = "Validation Loss"
    x_col = "flops"

    warmstarting_df = pd.read_parquet(args.data_path)
    warmstarting_df = warmstarting_df.dropna(subset=[y_col])
    warmstarting_df = preprocess_warmstarting(warmstarting_df)

    final_points_df = get_final_points_from_curve_set(
        warmstarting_df,
        unique_col_list,
        x_col="flops",
        y_col="Validation Loss",
        get_pareto=False,
    )

    final_points_df['g'] = final_points_df["target_N"] / final_points_df["base_N"]

    mup_df = final_points_df[final_points_df['method']=='mup']
    mup_jump_df = filter_pairs(mup_df, jump_size=1)
    mup_data_X, mup_data_y = preprocess_approach3_data(mup_jump_df)

    # Save the dataframe for mup and warmstarting
    df_mup = fit_chin3(mup_data_X, mup_data_y, n_workers=1, worker_idx=0, dense_grid=True)

    warm_df = final_points_df[(final_points_df['method']=='paws') & (final_points_df['shrinking']==0.4)]
    warm_jump_df = filter_pairs(warm_df, jump_size=1)
    warm_data_X, warm_data_y = preprocess_approach3_data(warm_jump_df)
    df_warm = fit_chin3(warm_data_X, warm_data_y)






