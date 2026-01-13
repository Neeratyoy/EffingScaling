import numpy as np
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


def fit_model(
    X_data: List | np.ndarray,
    y_data: List | np.ndarray,
    initial_grid: List | np.ndarray,
    delta: float=1e-3,
    scipy: bool=True
) -> Tuple[List[float], float]:
    """
    Fit using Huber loss (robust to outliers)
    """
    best_params = None
    best_loss = np.inf
    _loss = huber_loss_scipy if scipy else huber_loss
    for init in initial_grid:
        result = scipy.optimizeminimize(
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