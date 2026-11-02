from abc import ABCMeta, abstractmethod
from typing import Tuple
import numpy as np
from si.neural_networks.layers import Layer


class ActivationLayer(Layer, metaclass=ABCMeta):
    """
    Classe base abstrata para todas as camadas de ativacao.
    """

    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input_data: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Calcula a saida aplicando a funcao de ativacao a entrada.

        Parameters
        --------
        input_data : np.ndarray
            Dados de entrada.
        training : bool, default=False
            Indica se esta em modo de treino ou inferencia.

        Returns
        --------
        np.ndarray
            Saida ativada.
        """
        self.input = input_data
        self.output = self.activation_function(self.input)
        return self.output

    def backward_propagation(self, output_error: np.ndarray) -> np.ndarray:
        """
        Retropropaga o erro multiplicando elemento a elemento pela derivada.

        Parameters
        --------
        output_error : np.ndarray
            Erro vindo da camada seguinte.

        Returns
        --------
        np.ndarray
            Erro retropropagado (dE/dX).
        """
        return self.derivative(self.input) * output_error

    @abstractmethod
    def activation_function(self, input_data: np.ndarray) -> np.ndarray:
        """
        Calcula a funcao de ativacao.
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, input_data: np.ndarray) -> np.ndarray:
        """
        Calcula a derivada da funcao de ativacao relativamente a entrada.
        """
        raise NotImplementedError

    def output_shape(self) -> Tuple[int, ...]:
        """
        As camadas de ativacao preservam a dimensao da entrada.
        """
        return self._input_shape

    def parameters(self) -> int:
        """
        Camadas de ativacao nao possuem parametros treinaveis.
        """
        return 0


class ReLUActivation(ActivationLayer):
    """
    Funcao de ativacao Rectified Linear Unit (ReLU):
    f(x) = max(0, x)
    f'(x) = 1 se x >= 0, senao 0
    """

    def activation_function(self, input_data: np.ndarray) -> np.ndarray:
        return np.maximum(0, input_data)

    def derivative(self, input_data: np.ndarray) -> np.ndarray:
        return (input_data >= 0).astype(float)


class SigmoidActivation(ActivationLayer):
    """
    Funcao de ativacao Sigmoide:
    f(x) = 1 / (1 + exp(-x))
    f'(x) = f(x) * (1 - f(x))
    """

    def activation_function(self, input_data: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-input_data))

    def derivative(self, input_data: np.ndarray) -> np.ndarray:
        sig = self.activation_function(input_data)
        return sig * (1.0 - sig)


class TanhActivation(ActivationLayer):
    """
    Funcao de ativacao Tangente Hiperbolica (Tanh):
    f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)) = np.tanh(x)
    f'(x) = 1 - f(x)^2
    """

    def activation_function(self, input_data: np.ndarray) -> np.ndarray:
        return np.tanh(input_data)

    def derivative(self, input_data: np.ndarray) -> np.ndarray:
        tanh_val = self.activation_function(input_data)
        return 1.0 - tanh_val ** 2


class SoftmaxActivation(ActivationLayer):
    """
    Funcao de ativacao Softmax estavel para classificacao multiclasse:
    f(x_i) = exp(x_i - max(x)) / sum(exp(x - max(x)))
    f'(x) = f(x) * (1 - f(x)) (aproximacao diagonal para backward element-wise)
    """

    def activation_function(self, input_data: np.ndarray) -> np.ndarray:
        shifted = input_data - np.max(input_data, axis=-1, keepdims=True)
        exps = np.exp(shifted)
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def derivative(self, input_data: np.ndarray) -> np.ndarray:
        s = self.activation_function(input_data)
        return s * (1.0 - s)


if __name__ == "__main__":
    x = np.array([-2.5, 1.2, 0.3, 0.4, 1.1, 2.7])
    eps = 1e-6

    print("Teste das Derivadas de Ativacao")
    for act in [SigmoidActivation(), ReLUActivation(), TanhActivation()]:
        num = (act.activation_function(x + eps) - act.activation_function(x - eps)) / (2 * eps)
        diff = np.abs(num - act.derivative(x)).max()
        print(f"{act.layer_name()}: diferenca maxima = {diff:.2e}")
        assert diff < 1e-7, f"Erro na derivada de {act.layer_name()}"
    print("Todas as funcoes de ativacao passaram no gradient check!")