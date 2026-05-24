from functools import partial
import numpy as np
import pandas as pd
import scipy
from typing import Callable, Dict, List, Tuple
from scipy.special import logsumexp


_MISSING = object()  # sentinel value instead of None for missing functional form argument


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
    df: pd.DataFrame,
    x_name="flops",
    y_name="Validation Loss",
    num_buckets: int=None,
    buck_interval: int=None
) -> pd.DataFrame: 
    raise NotImplementedError("This function is not yet implemented.")


def get_convex_hull(
    **kwargs
) -> pd.DataFrame:
    raise NotImplementedError("This function is not yet implemented.")


def huber_loss(
    params: List | Dict,
    data: List | Dict,
    y_true: List,
    func_form: Callable=_MISSING,
    delta: float=1e-3,
) -> float:
    """ Compute the Huber loss between true and predicted values.

    NOTE: assuming this is used by scipy.optimize.minimize, the first argument is params.

    Args:
        params (List | Dict): Parameters for the functional form.
        data (List | Dict): Input data for predictions.
        y_true (List): True target values.
        func_form (Callable): The functional form to generate predictions.
            Function that takes in data and parameters and returns predicted values.
        delta (float, optional): The threshold parameter for Huber loss. Defaults to 1e-3.
    
    Returns:
        float: The computed Huber loss.
    """
    if func_form is _MISSING:
        raise TypeError("huber_loss() missing required keyword argument: 'func_form'")
    y_pred = func_form(data, params)
    residuals = np.array(y_true) - np.array(y_pred)
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
) -> float:
    """ Compute the Huber loss using SciPy.

    NOTE: assuming this is used by scipy.optimize.minimize, the first argument is params.

    Args:
        params (List | Dict): Parameters for the functional form.
            Custom ordering corresponding to the `func_form` being used.
        data (List | Dict): Input data for predictions.
            Custom ordering corresponding to the `func_form` being used.
            Shape and structure should be compatible with the `func_form` being used.
        y_true (List): True target values.
            Typically a 1D array or list of true values corresponding `func_form`.
        func_form (Callable): The functional form to generate predictions.
            Function that takes in data and parameters and returns predicted values.
        delta (float, optional): The threshold parameter for Huber loss. Defaults to 1e-3.
    
    Returns:
        float: The computed Huber loss.
    """
    if func_form is _MISSING:
        raise TypeError("huber_loss_scipy() missing required keyword argument: 'func_form'")
    y_pred = func_form(data, params)
    residuals = np.array(y_true) - np.array(y_pred)
    # scipy.special.huber handles the modulus internally
    losses = scipy.special.huber(delta, residuals)
    
    return np.sum(losses)


def fit_linear_model(
    X_data: np.ndarray | List | pd.Series,
    y_data: np.ndarray | List | pd.Series,
) -> List[float]:
    """ Fit a linear model to log-transformed data.

    Fits a simple: y = `slope` * x + `intercept` using scipy.stats.linregress.
    To be used for Chinchilla Approach 1 style fits.

    NOTE: any log-transformations should be applied to the input data before calling this function.

    Args:
        X_data (List | np.ndarray): Input data for predictions.
        y_data (List | np.ndarray): True target values.

    Returns:
        Tuple[List[float], float]: The best-fitting parameters and the corresponding loss value.
            Order of return: (slope, intercept, r_value, p_value, std_err)
    """
    if isinstance(X_data, pd.Series):
        X_data = X_data.values
    if isinstance(y_data, pd.Series):
        y_data = y_data.values

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X_data, y_data)
    
    return slope, intercept, r_value, p_value, std_err


def fit_linear_model_bootstrapped(
    X_data: np.ndarray | List | pd.Series,
    y_data: np.ndarray | List | pd.Series,
    bootstrap_iters: int=100,
    bootstrap_fraction: float=0.8
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """ Fit a linear model to log-transformed data with bootstrapping.

    Calls `fit_linear_model` multiple times on bootstrapped samples of the data 
    to estimate the variability of the fitted parameters.

    Args:
        X_data (List | np.ndarray): Input data for predictions.
        y_data (List | np.ndarray): True target values.
        bootstrap_iters (int, optional): Number of bootstrap iterations. 
            Defaults to 100.
        bootstrap_fraction (float, optional): Fraction of data to sample in each bootstrap iteration. 
            Defaults to 0.8.
        
    Returns:
        Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]: 
            Order of return: [
                (alpha_mean, alpha_std), 
                (intercept_mean, intercept_std), 
                (residuals_mean, residuals_std)
            ]
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


def functional_form_L0(
    data: list | np.ndarray,
    params: list[float],
    return_log_loss: bool=True,
) -> list[float]:
    """ The parameteric function `L = L0 + a * (C ** b)`, where L0 is the irreducible loss.

    The stable equivalent:
    Setting, L0 = exp(l0), a = exp(a) to ensure positivity.
    That is equivalent to: L = exp(l0) + exp(a + b * log(C)).
    Taking log on both sides gives: log(L) = logsumexp([l0, a + b*log(C)]).

    Therefore, the parameters input are expected to be [l0, a, b], where l0 = log(L0), a = log(a).
    
    NOTE: pass the raw `data` since the stable formulation handles the log-transform of C.

    Args:
        data (list | np.ndarray): Input data for predictions.
            Expected to be a 1D array of C values (e.g., FLOPs).
        params (list[float]): Parameters for the functional form.
            Expected order: [l0, a, b], where l0 = log(L0), a = log(a) for the stable formulation.
        return_log_loss (bool, optional): Whether to return log(L) or L.
            Defaults to True, which is recommended for numerical stability.

    Returns:
        list[float]: log(L) if return_log_loss=True, else L.

    """
    if len(params) != 3:
        raise ValueError(f"Expected 3 parameters for functional form, got {len(params)}.")
    if len(data.shape) != 1:
        raise ValueError(f"Expected data with 1 column (C), got shape {data.shape}.")
    
    l0, a, b  = params
    C = data

    exponents = np.stack([
        np.full(data.shape[0], l0),
        a + b * np.log(C)
    ], axis=-1)
    L = logsumexp(exponents, axis=-1)
    if not return_log_loss:
        L = np.exp(L)
    return L


def functional_form_chin3(
    data: list | np.ndarray,
    params: list[float],
    return_log_loss: bool=True,
) -> list[float]:
    """ The parametric function `L = A/N^alpha + B/D^beta + E` (Chinchilla Approach 3).

    The stable equivalent:
    Setting, A = exp(a), B = exp(b), E = exp(e) to ensure positivity.
    That is equivalent to: L = exp(a - α·log N) + exp(b - β·log D) + exp(e).
    Taking log on both sides gives: log(L) = logsumexp([a - α·log N, b - β·log D, e]).

    Therefore, the parameters input are expected to be [a, alpha, b, beta, e], 
    where a = log(A), b = log(B), e = log(E) for the stable formulation.
    
    NOTE: pass the raw `data` since the stable formulation handles the log-transform of N, D.

    Args:
        data (list | np.ndarray): Input data for predictions.
            Expected to be a 2D array with columns (N, D).
        params (list[float]): Parameters for the functional form.
            Expected order: [a, alpha, b, beta, e].
        return_log_loss (bool, optional): Whether to return log(L) or L.
            Defaults to True, which is recommended for numerical stability when fitting.

    Returns:
        list[float]: Predicted L values based on the functional form.
            If return_log_loss=True, returns log(L). Otherwise, returns L.

    """
    if len(params) != 5:
        raise ValueError(f"Expected 5 parameters for functional form, got {len(params)}.")
    if len(data.shape) != 2 or data.shape[1] != 2:
        raise ValueError(f"Expected data with 2 columns (N, D), got shape {data.shape}.")

    a, alpha, b, beta, e  = params
    N, D = data[:, 0], data[:, 1]

    exponents = np.stack([
        (a - alpha * np.log(N)),
        (b - beta * np.log(D)),
        np.full((data.shape[0]), e)
    ], axis=-1)
    L = logsumexp(exponents, axis=-1) 
    if not return_log_loss:
        L = np.exp(L)
    return L


def fit_parametric_form(
    func_form: Callable,
    X_data: list | np.ndarray,
    y_data: list | np.ndarray,
    initial_grid: list | np.ndarray,
    bounds: list[float] | None=None,
    delta: float=1e-3,
    use_scipy: bool=True,
) -> Tuple[float, float]:
    """
    Fit multiple parameters using Huber loss (robust to outliers).

    NOTE: the log-transformations of the data should be handled either as part of pre-processing, 
        or within the `func_form` itself, passed as the argument here.

    Args:
        func_form (Callable): The functional form to generate predictions.
        X_data (list | np.ndarray): Input data for predictions.
        y_data (list | np.ndarray): True target values.
        initial_grid (list | np.ndarray): A list of initial parameter sets to try for optimization.
            Should be in the format: [
                [param1_init, param2_init, ...], 
                [param1_init, param2_init, ...], 
                ...
            ], as per the requirements of the `func_form` being used.
            NOTE: this influences both the runtime and quality of the fit.
        bounds (list[float] | None, optional): Bounds for the parameters during optimization. 
            Defaults to None, which means no bounds.
            Should be in the format [
                (param1_lower, param1_upper),
                (param2_lower, param2_upper),
                ...
            ], corresponding to the parameters in `func_form`. 
            NOTE: this is crucial to clamp certain parameters such as the irreducible loss.
        delta (float, optional): The threshold parameter for Huber loss. Defaults to 1e-3.
        use_scipy (bool, optional): Whether to use the scipy.special.huber implementation.

    Returns:
        Tuple[float, float]: The best-fitting parameters and the corresponding loss value.
            Order of return: (best_params, best_loss)
    """
    best_params = None
    best_loss = np.inf
    _loss = huber_loss_scipy if use_scipy else huber_loss
    _loss = partial(_loss, func_form=func_form, delta=delta)

    for init in initial_grid:
        result = scipy.optimize.minimize(
                fun=_loss,
                x0=init,
                args=(X_data, y_data),
                method='L-BFGS-B',
                bounds=bounds,
            )
        if result.fun < best_loss:
                best_loss = result.fun
                best_params = result.x
    
    return best_params, best_loss
