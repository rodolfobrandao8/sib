import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula a percentagem de amostras corretamente classificadas.

    Parameters
    -------
    y_true : np.ndarray
        Os valores reais dos rótulos.
    y_pred : np.ndarray
        Os valores previstos dos rótulos.

    Returns
    -------
    float
        A fração de amostras corretamente classificadas.
    """
    return np.sum(y_true == y_pred) / len(y_true)