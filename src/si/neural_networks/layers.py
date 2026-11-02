import copy
from abc import ABCMeta, abstractmethod
from typing import Tuple, Optional
import numpy as np


class Layer(metaclass=ABCMeta):
    """
    Classe base abstrata para todas as camadas da rede neuronal.
    """

    @abstractmethod
    def forward_propagation(self, input_data: np.ndarray, training: bool) -> np.ndarray:
        """
        Executa a propagacao para a frente.
        """
        raise NotImplementedError

    @abstractmethod
    def backward_propagation(self, output_error: np.ndarray) -> np.ndarray:
        """
        Executa a retropropagacao do erro.
        """
        raise NotImplementedError

    @abstractmethod
    def output_shape(self) -> Tuple[int, ...]:
        """
        Devolve a dimensao de saida da camada.
        """
        raise NotImplementedError

    @abstractmethod
    def parameters(self) -> int:
        """
        Devolve o numero total de parametros ajustaveis da camada.
        """
        raise NotImplementedError

    def set_input_shape(self, input_shape: Tuple[int, ...]) -> None:
        """
        Define a dimensao dos dados de entrada para a camada.
        """
        self._input_shape = input_shape

    def input_shape(self) -> Tuple[int, ...]:
        """
        Devolve a dimensao de entrada da camada.
        """
        return self._input_shape

    def layer_name(self) -> str:
        """
        Devolve o nome da camada.
        """
        return self.__class__.__name__


class DenseLayer(Layer):
    """
    Camada Dense (totalmente conectada / fully-connected).

    Parameters
    --------
    n_units : int
        Numero de neuronios na camada.
    input_shape : Optional[Tuple[int, ...]], default=None
        Formato da entrada (ex: (n_features,)).
    """

    def __init__(self, n_units: int, input_shape: Optional[Tuple[int, ...]] = None):
        self.n_units = n_units
        self._input_shape = input_shape

        self.input = None
        self.output = None
        self.weights = None
        self.biases = None
        self.w_opt = None
        self.b_opt = None

    def initialize(self, optimizer=None) -> 'DenseLayer':
        """
        Inicializa pesos em [-0.5, 0.5[ e bias a 0, guardando tambem o otimizador.
        """
        n_features = self._input_shape[0]

        self.weights = np.random.rand(n_features, self.n_units) - 0.5
        self.biases = np.zeros((1, self.n_units))

        if optimizer is not None:
            self.w_opt = copy.deepcopy(optimizer)
            self.b_opt = copy.deepcopy(optimizer)

        return self

    def parameters(self) -> int:
        """
        Devolve o total de parametros (pesos + biases).
        """
        return int(np.prod(self.weights.shape) + np.prod(self.biases.shape))

    def forward_propagation(self, input_data: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Calcula a saida Y = X*W + B.
        """
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward_propagation(self, output_error: np.ndarray) -> np.ndarray:
        """
        Retropropaga o erro da camada e atualiza os pesos e biases.

        Parameters
        --------
        output_error : np.ndarray
            Gradiente do erro em relacao a saida da camada (dE/dY).

        Returns
        --------
        np.ndarray
            Gradiente do erro em relacao a entrada (dE/dX).
        """
        input_error = np.dot(output_error, self.weights.T)

        weights_error = np.dot(self.input.T, output_error)

        bias_error = np.sum(output_error, axis=0, keepdims=True)

        if self.w_opt is not None:
            self.weights = self.w_opt.update(self.weights, weights_error)
        if self.b_opt is not None:
            self.biases = self.b_opt.update(self.biases, bias_error)

        return input_error

    def output_shape(self) -> Tuple[int, ...]:
        """
        Formato da saida: (n_units,).
        """
        return (self.n_units,)


class Dropout(Layer):
    """
    Camada de regularizacao Dropout (Inverted Dropout).

    Parameters
    --------
    probability : float
        Taxa de dropout (probabilidade de desativar cada neuronio, entre 0 e 1).
    """

    def __init__(self, probability: float):
        self.probability = probability
        self.mask = None
        self.input = None
        self.output = None

    def forward_propagation(self, input_data: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Aplica a mascara binomial escalada se estiver em treino; senao devolve o input inalterado.

        Parameters
        --------
        input_data : np.ndarray
            Array com os dados de entrada.
        training : bool, default=False
            Se True, aplica dropout; se False, modo de teste/inferencia.

        Returns
        --------
        np.ndarray
            Array resultante.
        """
        self.input = input_data
        if training and self.probability > 0:
            scaling_factor = 1.0 / (1.0 - self.probability)
            self.mask = np.random.binomial(1, 1.0 - self.probability, size=input_data.shape)
            self.output = input_data * self.mask * scaling_factor
            return self.output
        else:
            self.mask = None
            self.output = input_data
            return self.output

    def backward_propagation(self, output_error: np.ndarray) -> np.ndarray:
        """
        Propaga o erro multiplicando pela mascara binomial utilizada no forward.

        Parameters
        --------
        output_error : np.ndarray
            Erro vindo da camada seguinte.

        Returns
        --------
        np.ndarray
            Erro retropropagado atraves do dropout.
        """
        if self.mask is not None:
            return output_error * self.mask
        return output_error

    def output_shape(self) -> Tuple[int, ...]:
        """
        Retorna a dimensao de entrada inalterada.
        """
        return self._input_shape

    def parameters(self) -> int:
        """
        O dropout nao possui parametros ajustaveis.
        """
        return 0


if __name__ == "__main__":
    class NoUpdate:
        def update(self, w, grad):
            return w

    mse = lambda Y, P: np.mean((Y - P) ** 2)
    mse_derivative = lambda Y, P: 2 * (P - Y) / Y.size

    np.random.seed(0)
    X, Y = np.random.rand(4, 3), np.random.rand(4, 2)
    layer = DenseLayer(2, (3,)).initialize(NoUpdate())
    out = layer.forward_propagation(X, training=True)

    analytic = X.T @ mse_derivative(Y, out)
    numeric, eps = np.zeros_like(layer.weights), 1e-6

    for i in range(3):
        for j in range(2):
            layer.weights[i, j] += eps
            lp = mse(Y, layer.forward_propagation(X, True))
            layer.weights[i, j] -= 2 * eps
            lm = mse(Y, layer.forward_propagation(X, True))
            layer.weights[i, j] += eps
            numeric[i, j] = (lp - lm) / (2 * eps)

    diff = np.abs(numeric - analytic).max()
    print("Diferenca maxima:", diff)