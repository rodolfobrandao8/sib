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
    ----------
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

    def initialize(self, optimizer=None) -> 'DenseLayer':
        """
        Inicializa pesos em [-0.5, 0.5[ e bias a 0.
        """
        n_features = self._input_shape[0]

        self.weights = np.random.rand(n_features, self.n_units) - 0.5

        self.biases = np.zeros((1, self.n_units))

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
        Retropropaga o erro (sera desenvolvido nos proximos guiões).
        """
        pass

    def output_shape(self) -> Tuple[int, ...]:
        """
        Formato da saida: (n_units,).
        """
        return (self.n_units,)