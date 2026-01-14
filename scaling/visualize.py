from matplotlib import pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Optional
import numpy as np

from .utils import get_pareto_frontier, fit_linear_model


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
) -> None:
    final_points = {}
    for i, x in enumerate(df.groupby(unique_col_list)):
        _df = x[1].dropna(subset=[x_col, y_col])
        _df = _df.sort_values(by=x_col)
        if plot_all_curves:
            ax.plot(_df[x_col], _df[y_col], alpha=0.01, color="black")
        final_points.update({i: {
            x_col: _df[x_col].values[-1],
            y_col: _df[y_col].values[-1],
        }})

    final_point_df = pd.DataFrame.from_dict(final_points, orient="index")
    final_pareto_df = get_pareto_frontier(final_point_df, x_col, y_col)

    if plot_final:
        ax.scatter(
            final_point_df[x_col],
            final_point_df[y_col],
            color="black",
            s=1,
        )

    if plot_pareto_final:
        ax.scatter(
            final_pareto_df[x_col],
            final_pareto_df[y_col],
            color="red",
            s=5,
            marker="x",
            label="Pareto Frontier",
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


def plot_line_fit(
    ax: plt.axes,
    X: np.ndarray,
    Y: np.ndarray,
    slope: float,
    intercept: float,
    x_extrapolate: List[float]=None,
    y_extrapolate: List[float]=None,
    xlog: bool=True,
    ylog: bool=True,
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
    ax.scatter(X, Y, label="raw data")
    _x_plot = np.linspace(X.min(), X.max(), 100)
    ax.plot(
        _x_plot, 
        np.exp(intercept + slope * np.log(_x_plot)), 
        color="red", 
        label="fitted line"
    )

    if x_extrapolate is not None:
        ax.scatter(
            x_extrapolate, 
            np.exp(intercept + slope * np.log(x_extrapolate)), 
            color="green",
            marker="x", 
            label="extrapolate"
        )

    if x_extrapolate is not None and y_extrapolate is not None and len(y_extrapolate) <= len(x_extrapolate):
        ax.scatter(
            x_extrapolate[:len(y_extrapolate)],
            y_extrapolate,
            color="black",
            zorder=-10,
            label="our data point",
        )
    ax.legend(loc="lower right")

    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
