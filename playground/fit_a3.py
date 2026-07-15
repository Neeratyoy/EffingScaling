from itertools import product

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scaling.utils import (
    fit_parametric_form_stable,
    functional_form_chin3_stable,
    get_final_points_from_curve_set,
)


def preprocess_warmstarting(df, y_col_to_smooth=None, smoothing_window=100):
    __df = pd.DataFrame()
    for i, x in enumerate(df.groupby(unique_col_list)):
        _df = x[1].sort_values(by="flops")
        # smooth it
        if y_col_to_smooth is not None:
            # +"_smoothed"
            _df[y_col_to_smooth] = _df[y_col_to_smooth].rolling(smoothing_window, win_type='gaussian',
                                                                min_periods=1).mean(std=smoothing_window / 10)

        # scaling tokens and flops to the max
        max_intended_tokens = (_df.iloc[-1]["target_N"] * _df.iloc[-1]["tkpm"])
        if abs((max_intended_tokens - _df["tokens"].max()) / _df["tokens"].max()) > 0.01:
            print("Wrong tkpm: ", x[0])
            continue
        _df["tokens"] = np.round(max_intended_tokens / _df["tokens"].max() * _df["tokens"])

        max_intended_flops = 6. * max_intended_tokens * _df["target_N"]
        _df["flops"] = np.round(max_intended_flops / _df["flops"].max() * _df["flops"])

        __df = pd.concat([__df, _df])

    print('Droping tkpm <= 5')
    __df = __df[__df['tkpm'] > 5.]

    return __df

def filter_pairs(df, jump_size=1):
    distinct_values = sorted(
        pd.unique(df[['base_N', 'target_N']].values.ravel())
    )
    pairs = list(zip(distinct_values[:-jump_size], distinct_values[jump_size:]))
    df_filtered = df[df[['base_N', 'target_N']].apply(tuple, axis=1).isin(pairs)]
    return df_filtered


if __name__ == "__main__":
    unique_col_list = ["base_N", "target_N", "tkpm", "shrink"]
    warmstarting_df = pd.read_parquet(
        "../data/warmstarting_results.parquet",
    )
    warmstarting_df = preprocess_warmstarting(warmstarting_df)
    y_col = "Validation Loss"
    x_col = "flops"
    jump_sizes = 1



    # axes[i].set_title(f"{(2)**(i+1)}x growth factor")
    jump_df = filter_pairs(warmstarting_df, jump_size=jump_size)

    shrink_factors = sorted(jump_df['shrink'].unique())
    colors = plt.cm.Blues(np.linspace(0, 1, len(shrink_factors)))

    jump_df = get_final_points_from_curve_set(
        jump_df,
        unique_col_list,
        x_col=x_col,
        y_col=y_col,
        get_pareto=False
    )

    shrink_df = jump_df[jump_df['shrink']==0.4]
    N = shrink_df["target_N"].values
    D = shrink_df["tokens"].values
    y = shrink_df["Validation Loss"].values

    _df = pd.DataFrame.from_dict({
        "N": N,
        "D": D,
        "Loss": y
    }).groupby(by=["N", "D"]).min().reset_index()
    _df.sort_values(by=["N", "D"], inplace=True)

    data_X = _df[["N", "D"]].values
    data_y = _df["Loss"].values

    initialization = list(product(
        np.linspace(0, 25, 5),  # a
        np.linspace(0., 1., 5),  # alpha
        np.linspace(0, 25, 5),  # b
        np.linspace(0., 1., 5),  # beta
        np.linspace(-1., 1., 5),  # e
    ))
    best_params, best_loss = fit_parametric_form_stable(
        functional_form_chin3_stable,
        data_X,
        data_y,
        initialization
    )

    _a, alpha, _b, beta, _e = best_params

    A = np.exp(_a)
    B = np.exp(_b)
    E = np.exp(_e)

    a = beta / (alpha + beta)
    b = alpha / (alpha + beta)

    G = ((alpha*A) / (beta*B)) ** (1 / (alpha + beta))




