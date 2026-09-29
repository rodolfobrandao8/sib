import numpy as np


def manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calcula a distancia de Manhattan (L1) entre dois vetores.

    Parameters
    -------
    x : np.ndarray
        Primeiro vetor.
    y : np.ndarray
        Segundo vetor.

    Returns
    -------
    float
        Valor da distancia de Manhattan.
    """
    return float(np.sum(np.abs(x - y)))