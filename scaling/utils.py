from functools import partial
import numpy as np
import pandas as pd
import scipy
from typing import Callable, Dict, List, Tuple
from scipy.special import logsumexp


_MISSING = object()


def _check_log_range(x: np.ndarray, tol: float=1e-200) -> bool:
    """ Check if the input array has values that are positive and not too close to zero for log transformation.
    """
    return np.all(x > tol)


def huber_loss(
    params: List | Dict,
    data: List | Dict,
    y_true: List,
    func_form: Callable=_MISSING,
    delta: float=1e-3,
    ylog: bool=False,
    tol: float=1e-200,
) -> float:
    """ Compute the Huber loss between true and predicted values.

    NOTE: assuming this is used by scipy.optimize.minimize, the first argument is params.

    Args:
        func_form (Callable): The functional form to generate predictions.
            Function that takes in data and parameters and returns predicted values.
        params (List | Dict): Parameters for the functional form.
        data (List | Dict): Input data for predictions.
        y_true (List): True target values.
        delta (float, optional): The threshold parameter for Huber loss. Defaults to 1e-3.
        ylog (bool, optional): Whether to apply log transformation to y_true and predictions. 
            Defaults to False.
    
    Returns:
        float: The computed Huber loss.
    """
    if func_form is _MISSING:
        raise TypeError("huber_loss() missing required keyword argument: 'func_form'")
    y_pred = func_form(data, params)
    if ylog:
        if not _check_log_range(y_true, tol=tol) or not _check_log_range(y_pred, tol=tol):
            raise ValueError(
                f"ylog=True requires all values > 0. "
                f"Got y_true min={np.array(y_true).min()}, y_pred min={np.array(y_pred).min()}"
            )
    residuals = (
        np.log(np.array(y_true)) - np.log(np.array(y_pred))
        if ylog else np.array(y_true) - np.array(y_pred)
    )
    abs_residuals = np.abs(residuals)
    
    return np.sum(np.where(
        abs_residuals <= delta,
        0.5 * residuals ** 2,  # Quadratic part got <= delta
        delta * (abs_residuals - 0.5 * delta)  # Linear part, otherwise
    ))


def huber_loss_scipy(
    params: List | Dict,
    data: List | Dict,
    y_true: List,
    func_form: Callable=_MISSING,
    delta: float=1e-3,
    ylog: bool=False,
    tol: float=1e-200,
) -> float:
    """ Compute the Huber loss using SciPy.

    NOTE: assuming this is used by scipy.optimize.minimize, the first argument is params.

    Args:
        func_form (Callable): The functional form to generate predictions.
            Function that takes in data and parameters and returns predicted values.
        params (List | Dict): Parameters for the functional form.
        data (List | Dict): Input data for predictions.
        y_true (List): True target values.
        delta (float, optional): The threshold parameter for Huber loss. Defaults to 1e-3.
        ylog (bool, optional): Whether to apply log transformation to y_true and predictions. 
            Defaults to False.
    
    Returns:
        float: The computed Huber loss.
    """
    if func_form is _MISSING:
        raise TypeError("huber_loss_scipy() missing required keyword argument: 'func_form'")
    y_pred = func_form(data, params)
    if ylog:
        if not _check_log_range(y_true, tol=tol) or not _check_log_range(y_pred, tol=tol):
            raise ValueError(
                f"ylog=True requires all values > 0. "
                f"Got y_true min={np.array(y_true).min()}, y_pred min={np.array(y_pred).min()}"
            )
    residuals = (
        np.log(np.array(y_true)) - np.log(np.array(y_pred))
        if ylog else np.array(y_true) - np.array(y_pred) 
    )
    # scipy.special.huber returns the Huber function value
    # huber(delta, r) computes the Huber loss for residual r
    losses = scipy.special.huber(delta, residuals)
    
    return np.sum(losses)


def fit_linear_model(
    X_data: np.ndarray | List | pd.Series,
    y_data: np.ndarray | List | pd.Series,
    xlog: bool=True,
    ylog: bool=True,
) -> List[float | float | float | float | float]:
    """ Fit a linear model to log-transformed data.

    Args:
        X_data (List | np.ndarray): Input data for predictions.
        y_data (List | np.ndarray): True target values.

    Returns:
        Tuple[List[float], float]: The best-fitting parameters and the corresponding loss value.
    """
    if isinstance(X_data, pd.Series):
        X_data = X_data.values
    if isinstance(y_data, pd.Series):
        y_data = y_data.values

    if xlog:
        X_data = np.log(X_data)
    if ylog:
        y_data = np.log(y_data)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X_data, y_data)
    
    return slope, intercept, r_value, p_value, std_err


def fit_linear_model_bootstrapped(
    X_data: np.ndarray | List | pd.Series,
    y_data: np.ndarray | List | pd.Series,
    bootstrap_iters: int=100,
    bootstrap_fraction: float=0.8
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """ Fit a linear model to log-transformed data with bootstrapping.
    """
    _alphas = []
    _intercepts = []
    _residuals = []
    for i in range(bootstrap_iters):
        _idx = np.random.choice(
            range(1, len(X_data)), 
            size=int(bootstrap_fraction * len(X_data)), 
            replace=False
        ) if bootstrap_fraction < 1 else range(1, len(X_data))
        alpha_N, intercept_N, r_value_N, _, _ = fit_linear_model(
            X_data[_idx],
            y_data[_idx]
        )
        _alphas.append(alpha_N)
        _intercepts.append(intercept_N)
        _residuals.append(r_value_N**2)

    alphas = (np.mean(_alphas), np.std(_alphas))
    intercept_N = (np.mean(_intercepts), np.std(_intercepts))
    residuals_N = (np.mean(_residuals), np.std(_residuals))

    return alphas, intercept_N, residuals_N


def get_pareto_frontier(
    df: pd.DataFrame,
    x_name="flops",
    y_name="Validation Loss",
) -> pd.DataFrame:
    """ Function to compute Pareto over FLOPs.
    """
    # NOTE: strict assumption here that x_name is maximized and y_name is minimized
    df_sorted = df.copy().sort_values(by=[x_name, y_name], ascending=[True, True])
    df_sorted = df_sorted.drop_duplicates(subset=[x_name], keep="first")
    pareto_points = []
    min_loss_so_far = float('inf')
    for _, row in df_sorted.iterrows():
        if row[y_name] < min_loss_so_far:
            pareto_points.append(row)
            min_loss_so_far = row[y_name]

    return pd.DataFrame(pareto_points)


def get_final_points_from_curve_set(
    df: pd.DataFrame,
    unique_col_list: List[str],
    x_col: str = "flops",
    y_col: str = "Validation Loss",
    get_pareto: bool = False,
) -> pd.DataFrame:
    """ Function to extract final points from a set of curves.
    """
    __df = pd.DataFrame()

    for i, x in enumerate(df.groupby(unique_col_list)):
        _df = x[1].sort_values(by=x_col)
        __df = pd.concat([__df, _df.iloc[-1:]])

    final_point_df = __df
    if get_pareto:
        final_point_df = get_pareto_frontier(final_point_df, x_col, y_col)
    
    return final_point_df


def get_pareto_frontier_by_buckets(
    df: pd.DataFrame, x_name="flops", y_name="Validation Loss", num_buckets: int=None, buck_interval: int=None
) -> pd.DataFrame: 
    raise NotImplementedError("This function is not yet implemented.")


def get_convex_hull(
    **kwargs
) -> pd.DataFrame:
    raise NotImplementedError("This function is not yet implemented.")


def functional_form_L0(
    data: list | np.ndarray,
    params: list[float],
    xlog: bool=False,
) -> list[float]:
    """ The parameteric function `L = L0 + a * (C ** b)`, where L0 is the irreducible loss.
    """
    assert len(params) == 3, "Expected 3 parameters for functional form."
    L0, a, b  = params
    assert len(data.shape)== 1, "Expected data with 1 column (C)."
    
    C = data

    if xlog:
        C = np.log(C)

    L = L0 + a * (C ** b)
    return L


def functional_form_chin3(
    data: list | np.ndarray,
    params: list[float],
    xlog: bool=False,
) -> list[float]:
    """ The parameteric function 
    """
    assert len(params) == 5, "Expected 5 parameters for functional form."
    a, alpha, b, beta, e  = params
    assert data.shape[1] == 2, "Expected data with 2 columns (N, D)."
    if xlog:
        data = np.log(data)
    N, D = data[:, 0], data[:, 1]

    L = np.exp(a - alpha * np.log(N)) + np.exp(b - beta * np.log(D)) + np.exp(e)
    return L


def functional_form_chin3_stable(
    data: list | np.ndarray,
    params: list[float],
    xlog: bool=False,
) -> list[float]:
    assert len(params) == 5, "Expected 5 parameters for functional form."
    a, alpha, b, beta, e = params
    assert data.shape[1] == 2, "Expected data with 2 columns (N, D)."
    if xlog:
        data = np.log(data)
    N, D = data[:, 0], data[:, 1]

    exponents = np.stack([
        (a - alpha * np.log(N)), 
        (b - beta * np.log(D)), 
        np.full((data.shape[0]), e)
    ], axis=-1)
    return logsumexp(exponents, axis=-1)


def fit_parametric_form(
    func_form: Callable,
    X_data: list | np.ndarray,
    y_data: list | np.ndarray,
    initial_grid: list | np.ndarray,
    delta: float=1e-3,
    use_scipy: bool=True,
    xlog: bool=False,
    ylog: bool=False,
    bounds: list[float] | None=None,
) -> Tuple[float, float]:
    """
    Fit using Huber loss (robust to outliers)
    """
    best_params = None
    best_loss = np.inf
    _loss = partial(huber_loss_scipy, ylog=ylog) if use_scipy else huber_loss
    _loss = partial(_loss, func_form=partial(func_form, xlog=xlog), delta=delta)

    for init in initial_grid:
        result = scipy.optimize.minimize(
                fun=_loss,
                x0=init,
                args=(X_data, y_data),
                method='L-BFGS-B',
                bounds=bounds,
                # options={'disp': True, 'maxiter': 100}
            )
        if result.fun < best_loss:
                best_loss = result.fun
                best_params = result.x
    
    return best_params, best_loss
