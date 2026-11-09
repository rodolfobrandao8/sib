import sys
from pathlib import Path

root_src = Path(__file__).resolve().parents[2]
if str(root_src) not in sys.path:
    sys.path.insert(0, str(root_src))

from abc import ABCMeta, abstractmethod
import numpy as np


class LossFunction(metaclass=ABCMeta):
    """
    Classe base abstrata para as funcoes de perda em redes neuronais.
    """

    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcula o valor da funcao de perda.

        Parameters
        --------
        y_true : np.ndarray
            Rotulos reais.
        y_pred : np.ndarray
            Valores previstos pelo modelo.

        Returns
        --------
        float
            Valor da perda.
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calcula a derivada da funcao de perda em relacao as previsoes.

        Parameters
        --------
        y_true : np.ndarray
            Rotulos reais.
        y_pred : np.ndarray
            Valores previstos pelo modelo.

        Returns
        --------
        np.ndarray
            Gradiente da perda (dE/dy_pred).
        """
        raise NotImplementedError


class MeanSquaredError(LossFunction):
    """
    Funcao de perda Mean Squared Error (MSE), indicada para regressao.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
  
        return 2 * (y_pred - y_true) / y_true.size


class BinaryCrossEntropy(LossFunction):
    """
    Funcao de perda Binary Cross-Entropy (BCE), indicada para classificacao binaria.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:

        p = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        p = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        return ((-y_true / p) + ((1.0 - y_true) / (1.0 - p))) / y_true.shape[0]


class CategoricalCrossEntropy(LossFunction):
    """
    Funcao de perda Categorical Cross-Entropy, indicada para classificacao multiclasse
    com rotulos em formato one-hot.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        p = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        return float(-np.sum(y_true * np.log(p)) / y_true.shape[0])

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        p = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        return (-y_true / p) / y_true.shape[0]


if __name__ == "__main__":
    y_t = np.array([[1.0], [0.0]])
    y_p = np.array([[0.9], [0.1]])

    mse = MeanSquaredError()
    print("MSE loss:", mse.loss(y_t, y_p))
    print("MSE deriv shape:", mse.derivative(y_t, y_p).shape)

    bce = BinaryCrossEntropy()
    print("BCE loss:", bce.loss(y_t, y_p))
    print("BCE deriv shape:", bce.derivative(y_t, y_p).shape)

    cce = CategoricalCrossEntropy()
    y_t_mc = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_p_mc = np.array([[0.8, 0.2], [0.1, 0.9]])
    print("CCE loss:", cce.loss(y_t_mc, y_p_mc))
    print("CCE deriv shape:", cce.derivative(y_t_mc, y_p_mc).shape)
    print("Modulo losses.py validado!")