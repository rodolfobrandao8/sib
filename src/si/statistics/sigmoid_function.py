import numpy as np


def sigmoid_function(x: np.ndarray) -> np.ndarray:
    """
    Calcula a funcao sigmoide para o input x.

    Parameters
    -------
    x : np.ndarray
        Array com valores de entrada.

    Returns
    -------
    np.ndarray
        Array com probabilidades calculadas entre 0 e 1.
    """
    return 1.0 / (1.0 + np.exp(-x))