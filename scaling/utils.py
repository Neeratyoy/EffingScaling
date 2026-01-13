import numpy as np
import pandas as pd
import scipy
from typing import Callable, Dict, List, Tuple


def huber_loss(
    func_form: Callable,
    params: List | Dict,
    data: List | Dict,
    y_true: List,
    delta: float=1e-3,
) -> float:
    """ Compute the Huber loss between true and predicted values.

    Args:
        func_form (Callable): The functional form to generate predictions.
            Function that takes in data and parameters and returns predicted values.
        params (List | Dict): Parameters for the functional form.
        data (List | Dict): Input data for predictions.
        y_true (List): True target values.
        delta (float, optional): The threshold parameter for Huber loss. Defaults to 1e-3.
    
    Returns:
        float: The computed Huber loss.
    """
    y_pred = func_form(data, params)
    residuals = np.array(y_true) - np.array(y_pred)
    abs_residuals = np.abs(residuals)
    
    return np.sum(np.where(
        abs_residuals <= delta,
        0.5 * residuals ** 2,  # Quadratic part got <= delta
        delta * (abs_residuals - 0.5 * delta)  # Linear part, otherwise
    ))


def huber_loss_scipy(
    func_form: Callable,
    params: List | Dict,
    data: List | Dict,
    y_true: List,
    delta: float=1e-3,
) -> float:
    """ Compute the Huber loss using SciPy.

    Args:
        func_form (Callable): The functional form to generate predictions.
            Function that takes in data and parameters and returns predicted values.
        params (List | Dict): Parameters for the functional form.
        data (List | Dict): Input data for predictions.
        y_true (List): True target values.
        delta (float, optional): The threshold parameter for Huber loss. Defaults to 1e-3.
    
    Returns:
        float: The computed Huber loss.
    """
    y_pred = func_form(data, params)
    residuals = np.array(y_true) - np.array(y_pred)    
    # scipy.special.huber returns the Huber function value
    # huber(delta, r) computes the Huber loss for residual r
    losses = scipy.special.huber(delta, residuals)
    
    return np.sum(losses)


def fit_model_lbfgs(
    X_data: List | np.ndarray,
    y_data: List | np.ndarray,
    initial_grid: List | np.ndarray,
    delta: float=1e-3,
    scipy: bool=True
) -> Tuple[List[float], float]:
    """ Fit model parameters by minimizing the Huber loss.

    Args:
        X_data (List | np.ndarray): Input data for predictions.
        y_data (List | np.ndarray): True target values.
        initial_grid (List | np.ndarray): Initial parameter guesses for optimization.
        delta (float, optional): The threshold parameter for Huber loss. Defaults to 1e-3.
        scipy (bool, optional): Whether to use SciPy's Huber loss implementation. Defaults to True.

    Returns:
        Tuple[List[float], float]: The best-fitting parameters and the corresponding loss value.
    """
    best_params = None
    best_loss = np.inf
    _loss = huber_loss_scipy if scipy else huber_loss
    for init in initial_grid:
        result = scipy.optimize.minimize(
                fun=_loss,
                x0=init,
                args=(X_data, y_data),
                method='L-BFGS-B',
                # options={'disp': True, 'maxiter': 100}
            )
        if result.fun < best_loss:
                best_loss = result.fun
                best_params = result.x
    
    return best_params, best_loss


def fit_linear_model(
    X_data: np.ndarray | List | pd.Series,
    y_data: np.ndarray | List | pd.Series,
) -> List[float | float | float | float | float]:
    """ Fit a linear model to log-transformed data.

    Args:
        X_data (List | np.ndarray): Input data for predictions.
        y_data (List | np.ndarray): True target values.

    Returns:
        Tuple[List[float], float]: The best-fitting parameters and the corresponding loss value.
    """
    log_C = np.log(X_data)
    log_y = np.log(y_data)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(log_C, log_y)
    
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
            size=int(bootstrap_fraction * len(pareto_df)), 
            replace=False
        ) if bootstrap_fraction < 1 else range(1, len(pareto_df))
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
