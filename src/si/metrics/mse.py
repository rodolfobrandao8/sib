import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula o Mean Squared Error (MSE) entre os valores reais e previstos.

    Parameters
    -------
    y_true : np.ndarray
        Valores reais de y.
    y_pred : np.ndarray
        Valores previstos de y.

    Returns
    -------
    float
        O valor do erro MSE.
    """
    return float(np.mean((y_true - y_pred) ** 2))