import sys
from pathlib import Path

root_src = Path(__file__).resolve().parents[2]
if str(root_src) not in sys.path:
    sys.path.insert(0, str(root_src))

from typing import List, Callable, Optional, Union, Type
import numpy as np

from si.data.dataset import Dataset
from si.neural_networks.layers import Layer
from si.neural_networks.losses import LossFunction
from si.neural_networks.optimizers import Optimizer, SGD


class NeuralNetwork:
    """
    Rede neuronal sequencial modular para classificacao e regressao.

    Parameters
    --------
    epochs : int, default=100
        Numero de epocas de treino.
    batch_size : int, default=32
        Tamanho de cada mini-batch.
    optimizer : Union[Type[Optimizer], Optimizer], default=SGD
        Classe ou instancia de otimizador a utilizar.
    learning_rate : float, default=0.01
        Taxa de aprendizagem.
    verbose : bool, default=False
        Se True, imprime metricas por epoca.
    loss : Union[Type[LossFunction], LossFunction], default=None
        Funcao de perda a otimizar.
    metric : Optional[Callable[[np.ndarray, np.ndarray], float]], default=None
        Metrica de avaliacao (ex: accuracy).
    **kwargs : dict
        Argumentos adicionais passados ao otimizador.
    """

    def __init__(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        optimizer: Union[Type[Optimizer], Optimizer] = SGD,
        learning_rate: float = 0.01,
        verbose: bool = False,
        loss: Optional[Union[Type[LossFunction], LossFunction]] = None,
        metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        **kwargs
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.metric = metric

        if isinstance(optimizer, type):
            self.optimizer = optimizer(learning_rate=learning_rate, **kwargs)
        else:
            self.optimizer = optimizer

        if isinstance(loss, type):
            self.loss = loss()
        else:
            self.loss = loss

        self.layers: List[Layer] = []
        self.history = {}

    def add(self, layer: Layer) -> 'NeuralNetwork':
        """
        Adiciona uma camada a rede neuronal e inicializa os seus parametros.

        Parameters
        --------
        layer : Layer
            Camada a adicionar.

        Returns
        --------
        NeuralNetwork
            A propria instancia (permite encadeamento de chamadas).
        """
        if self.layers:
            layer.set_input_shape(self.layers[-1].output_shape())

        if hasattr(layer, "initialize"):
            layer.initialize(self.optimizer)

        self.layers.append(layer)
        return self

    def fit(self, dataset: Dataset) -> 'NeuralNetwork':
        """
        Treina a rede neuronal no dataset fornecido recorrendo a mini-batches.

        Parameters
        --------
        dataset : Dataset
            Dataset contendo X e y para treino.

        Returns
        --------
        NeuralNetwork
            A propria instancia treinada.
        """
        n_samples = dataset.X.shape[0]
        self.history = {"loss": [], "metric": []}

        for epoch in range(1, self.epochs + 1):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for start_idx in range(0, n_samples, self.batch_size):
                batch_idx = indices[start_idx : start_idx + self.batch_size]
                x_batch = dataset.X[batch_idx]
                y_batch = dataset.y[batch_idx]

                output = x_batch
                for layer in self.layers:
                    output = layer.forward_propagation(output, training=True)

                if y_batch.ndim == 1 and output.ndim == 2 and output.shape[1] == 1:
                    y_true_batch = y_batch.reshape(-1, 1)
                else:
                    y_true_batch = y_batch

                error = self.loss.derivative(y_true_batch, output)

                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error)

            y_pred_all = self.predict(dataset)
            y_true_all = dataset.y.reshape(-1, 1) if (dataset.y.ndim == 1 and y_pred_all.ndim == 2 and y_pred_all.shape[1] == 1) else dataset.y

            epoch_loss = self.loss.loss(y_true_all, y_pred_all)
            self.history["loss"].append(epoch_loss)

            epoch_metric = None
            if self.metric is not None:
                epoch_metric = self.metric(dataset.y, y_pred_all)
                self.history["metric"].append(epoch_metric)

            if self.verbose:
                metric_str = f" - metric: {epoch_metric:.4f}" if epoch_metric is not None else ""
                print(f"Epoch {epoch}/{self.epochs} - loss: {epoch_loss:.4f}{metric_str}")

        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Gera previsoes para o dataset em modo de inferencia (training=False).

        Parameters
        --------
        dataset : Dataset
            Dataset com os dados de entrada X.

        Returns
        --------
        np.ndarray
            Saida da ultima camada da rede.
        """
        output = dataset.X
        for layer in self.layers:
            output = layer.forward_propagation(output, training=False)
        return output

    def score(self, dataset: Dataset) -> float:
        """
        Calcula a metrica de avaliacao no dataset fornecido.

        Parameters
        --------
        dataset : Dataset
            Dataset de teste com X e y.

        Returns
        --------
        float
            Resultado da metrica.
        """
        if self.metric is None:
            raise ValueError("Nenhuma metrica foi especificada para a rede neuronal.")
        y_pred = self.predict(dataset)
        return float(self.metric(dataset.y, y_pred))


if __name__ == "__main__":
    from si.neural_networks.layers import DenseLayer
    from si.neural_networks.activation import SigmoidActivation
    from si.neural_networks.losses import MeanSquaredError

    def accuracy_binary(y_true, y_pred):
        preds = (y_pred >= 0.5).astype(int).flatten()
        return np.mean(y_true.flatten() == preds)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0], dtype=float)
    ds = Dataset(X, y)

    np.random.seed(1)
    net = NeuralNetwork(
        epochs=5000,
        batch_size=4,
        optimizer=SGD,
        learning_rate=0.5,
        loss=MeanSquaredError,
        metric=accuracy_binary,
        verbose=False
    )
    net.add(DenseLayer(4, (2,)))
    net.add(SigmoidActivation())
    net.add(DenseLayer(1))
    net.add(SigmoidActivation())

    net.fit(ds)
    preds = net.predict(ds)
    print("Previsoes no XOR (arredondadas):", preds.flatten().round(2))
    print("Exactidao no XOR:", net.score(ds))