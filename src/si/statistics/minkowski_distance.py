import numpy as np


def minkowski_distance(x: np.ndarray, y: np.ndarray, p: float = 2.0) -> float:
    """
    Calcula a distancia de Minkowski de ordem p entre dois vetores.

    Parameters
    -------
    x : np.ndarray
        Primeiro vetor.
    y : np.ndarray
        Segundo vetor.
    p : float, default=2.0
        Ordem da distancia de Minkowski (p=1 para Manhattan, p=2 para Euclidiana).

    Returns
    -------
    float
        Valor da distancia de Minkowski.
    """
    return float(np.sum(np.abs(x - y) ** p) ** (1.0 / p))