import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula o Root Mean Squared Error (RMSE) entre os valores reais e as previsoes.

    Parameters
    -------
    y_true : np.ndarray
        Valores reais de y.
    y_pred : np.ndarray
        Valores previstos de y.

    Returns
    -------
    float
        O valor do erro RMSE.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))