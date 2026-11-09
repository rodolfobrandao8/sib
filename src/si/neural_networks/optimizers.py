import sys
from pathlib import Path

root_src = Path(__file__).resolve().parents[2]
if str(root_src) not in sys.path:
    sys.path.insert(0, str(root_src))

from abc import ABCMeta, abstractmethod
import numpy as np


class Optimizer(metaclass=ABCMeta):
    """
    Classe base abstrata para os otimizadores em redes neuronais.

    Parameters
    --------
    learning_rate : float
        Taxa de aprendizagem do otimizador.
    """

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Atualiza os parametros da camada com base no gradiente.

        Parameters
        --------
        w : np.ndarray
            Parametros atuais (pesos ou biases).
        grad_loss_w : np.ndarray
            Gradiente da perda relativamente aos parametros.

        Returns
        --------
        np.ndarray
            Parametros atualizados.
        """
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent com termo de momento (Momentum).

    Parameters
    --------
    learning_rate : float, default=0.01
        Taxa de aprendizagem.
    momentum : float, default=0.0
        Fator de momento (beta) para acumular gradientes anteriores.
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.retained_gradient = None

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Atualiza o gradiente retido e devolve os novos pesos.

        Parameters
        --------
        w : np.ndarray
            Parametros atuais.
        grad_loss_w : np.ndarray
            Gradiente da perda relativamente aos parametros.

        Returns
        --------
        np.ndarray
            Parametros atualizados.
        """
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros_like(w)

        self.retained_gradient = self.momentum * self.retained_gradient + (1.0 - self.momentum) * grad_loss_w

        return w - self.learning_rate * self.retained_gradient


class Adam(Optimizer):
    """
    Otimizador Adam (Adaptive Moment Estimation).

    Parameters
    --------
    learning_rate : float, default=0.001
        Taxa de aprendizagem.
    beta_1 : float, default=0.9
        Taxa de decaimento exponencial da media de 1.a ordem (m).
    beta_2 : float, default=0.999
        Taxa de decaimento exponencial da media de 2.a ordem (v).
    epsilon : float, default=1e-8
        Constante de estabilidade numerica para evitar divisao por zero.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8
    ):
        super().__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.m = None
        self.v = None
        self.t = 0

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Atualiza m, v, o passo temporal t e calcula os novos parametros.

        Parameters
        --------
        w : np.ndarray
            Parametros atuais.
        grad_loss_w : np.ndarray
            Gradiente da perda relativamente aos parametros.

        Returns
        --------
        np.ndarray
            Parametros atualizados.
        """
        if self.m is None:
            self.m = np.zeros_like(w)
        if self.v is None:
            self.v = np.zeros_like(w)

        self.t += 1

        self.m = self.beta_1 * self.m + (1.0 - self.beta_1) * grad_loss_w
        self.v = self.beta_2 * self.v + (1.0 - self.beta_2) * (grad_loss_w ** 2)

        m_hat = self.m / (1.0 - (self.beta_1 ** self.t))
        v_hat = self.v / (1.0 - (self.beta_2 ** self.t))

        return w - self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon))


if __name__ == "__main__":
    w_initial = np.array([[1.0, 2.0], [3.0, 4.0]])
    grad = np.array([[0.1, 0.2], [-0.1, -0.2]])

    sgd = SGD(learning_rate=0.1, momentum=0.9)
    w_sgd = sgd.update(w_initial.copy(), grad)
    print("SGD w updated:\n", w_sgd)

    adam = Adam(learning_rate=0.01)
    w_adam = adam.update(w_initial.copy(), grad)
    print("Adam w updated:\n", w_adam)
    print("Modulo optimizers.py validado com sucesso")