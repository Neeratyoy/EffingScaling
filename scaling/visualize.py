from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .utils import get_pareto_frontier


def visualize_train_curves(
    ax: plt.axes,
    df: pd.DataFrame, 
    unique_col_list: List[str],
    x_col: str = "flops",
    y_col: str = "Validation Loss",
    plot_all_curves: bool = True,
    plot_final: bool = True,
    plot_pareto_final: bool = True,
    xlims: Optional[Tuple[float, float]] = None,
    ylims: Optional[Tuple[float, float]] = None,
    xlog: bool = False,
    ylog: bool = False,
    style: Optional[dict] = None,
) -> None:
    if style is None:
        style = {}
    __df = pd.DataFrame()
    for i, x in enumerate(df.groupby(unique_col_list)):
        _df = x[1].dropna(subset=[x_col, y_col])
        _df = _df.sort_values(by=x_col)
        if plot_all_curves:
            curve_style = {
                "alpha": 0.02,
                "color": "black"
            }
            curve_style.update(style)
            ax.plot(_df[x_col], _df[y_col], **curve_style)
        __df = pd.concat([__df, _df.iloc[-1:]])

    final_point_df = __df
    final_pareto_df = get_pareto_frontier(final_point_df, x_col, y_col)

    if plot_final:
        final_style = {
            "alpha": 0.5,
            "color": "black",
            "s": 1,
        }
        final_style.update(style)
        ax.scatter(
            final_point_df[x_col],
            final_point_df[y_col],
            **final_style,
        )

    if plot_pareto_final:
        pareto_style = {
            "marker": "x",
            "s": 5,
            "zorder": 10,
            "color": "red",
        }
        pareto_style.update(style)
        ax.scatter(
            final_pareto_df[x_col],
            final_pareto_df[y_col],
            **pareto_style,
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    if ylims is not None:
        ax.set_ylim(*ylims)
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    if xlims is not None:
        ax.set_xlim(*xlims)

    return final_pareto_df


def plot_line_fit(
    ax: plt.axes,
    X: np.ndarray,
    Y: np.ndarray,
    slope: float = None,
    intercept: float = None,
    x_extrapolate: List[float]=None,
    y_extrapolate: List[float]=None,
    style: Optional[dict]=None,
) -> None:  
    """ Function to plot line fit on log-log scale.

    Args:
        ax (plt.axes): Matplotlib axes to plot on.
        X (np.ndarray): X data points.
        Y (np.ndarray): Y data points.
        slope (float): Slope of the fitted line in log-log scale.
        intercept (float): Intercept of the fitted line in log-log scale.
        x_extrapolate (List[float], optional): X values to extrapolate. Defaults to None.
        y_extrapolate (List[float], optional): Y values corresponding to x_extrapolate. Defaults to None.

    Returns:
        None.
    """
    if style is None:
        style = {}
    scatter_style = {
        "label": "raw data",
    }
    scatter_style.update(style)
    ax.scatter(X, Y, **scatter_style)
    if slope is not None and intercept is not None:
        _x_plot = np.linspace(X.min(), X.max(), 100)
        line_style = {
            "color": "red",
            "label": "fitted line"
        }
        line_style.update(style)
        ax.plot(
            _x_plot,
            np.exp(intercept + slope * np.log(_x_plot)),
            **line_style
        )

        if x_extrapolate is not None:
            x_extrapolate_style = {
                "color": "green",
                "marker": "x",
                "label": "extrapolate"
            }
            x_extrapolate_style.update(style)
            ax.scatter(
                x_extrapolate,
                np.exp(intercept + slope * np.log(x_extrapolate)),
                **x_extrapolate_style
            )

        if x_extrapolate is not None and y_extrapolate is not None and len(y_extrapolate) <= len(x_extrapolate):
            y_extrapolate_style = {
                "color": "black",
                "label": "our data point"
            }
            y_extrapolate_style.update(style)
            ax.scatter(
                x_extrapolate[:len(y_extrapolate)],
                y_extrapolate,
                zorder=-10,
                **y_extrapolate_style
            )
    ax.legend(loc="lower right")

    ax.set_xscale("log")
    ax.set_yscale("log")
