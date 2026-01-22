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
    xlog: bool=True,
    ylog: bool=True,
    x_max_plot: float = None,
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
    if x_max_plot is None:
        x_max_plot = - np.inf
    if style is None:
        style = {}
    scatter_style = {
        "label": "raw data",
    }
    scatter_style.update(style)
    ax.scatter(X, Y, **scatter_style)
    if slope is not None and intercept is not None:
        _x_plot = np.linspace(X.min(), max(X.max(), x_max_plot), 100)
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

    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")


import matplotlib.ticker as mticker


def plot_isoflops(
        ax: plt.Axes,
        isodata: pd.DataFrame,
        disable_y_label: bool = False,
) -> None:
    cmap = plt.cm.Greens  # (np.linspace(0.3, 0.9, len(intervals)))
    norm = plt.Normalize(min(isodata.index), max(isodata.index))

    flip_warmstarting = dict()
    min_loss = dict()
    for flops in isodata.index:
        x = isodata.columns.astype(float).values
        y = isodata.loc[flops].values

        # adding the points selected as per interval flop-loss
        ax.scatter(x, y, color=cmap(norm(flops)))
        z = np.polyfit(np.log(x), y, 2)
        p = np.poly1d(z)
        _x = np.linspace(min(x), max(x), 200)
        _y = p(np.log(_x))
        # find minimum loss point
        _min_coord = (_x[_y.argmin()], _y.min())
        min_loss[flops] = _min_coord
        # find flip point
        flip_index = np.sum(_y <= y[0])
        if flip_index < len(_x):
            _flip_coord = (_x[flip_index], _y[flip_index])
            flip_warmstarting[flops] = _flip_coord

        ax.plot(
            _x, _y,
            # marker='o',
            linewidth=2,
            # color=line_colors[i-1],
            color=cmap(norm(flops))
            # linestyle='--'
        )

        # Plot trace of minimum loss
    _min_loss = np.array([[*v] for k, v in min_loss.items()])
    ax.plot(
        _min_loss[:, 0], _min_loss[:, 1],
        marker='^',  # triangle marker pointing up
        markerfacecolor='none',  # transparent fill
        markeredgecolor='red',  # red edge
        markersize=4,
        color='red',
        linestyle='--'
    )
    _flip_warmstarting = np.array([[*v] for k, v in flip_warmstarting.items()])
    ax.plot(
        _flip_warmstarting[:, 0], _flip_warmstarting[:, 1],
        marker='v',  # triangle marker pointing up
        markerfacecolor='none',  # transparent fill
        markeredgecolor='blue',  # red edge
        markersize=4,
        color='blue',
        linestyle=':'
    )

    # Create colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.ax.invert_yaxis()
    _cbar = f"FLOPs"
    if disable_y_label:
        cbar.set_label(_cbar, fontsize=15)
    cbar.ax.tick_params(labelsize=15)
    # ax.xlabel(f"Growth factor", fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    # x_axis log scale
    ax.set_xscale('log')
    # get x axis limits

    # ax.get_xlimits()
    max_x = isodata.columns.astype(float).max()
    # power of 2 just above max_x
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlim(0.9, max_x * 1.1)
    ax.xaxis.set_minor_locator(mticker.NullLocator())