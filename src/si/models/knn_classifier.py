from typing import Callable, Union
import numpy as np
from scipy.spatial.distance import euclidean

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


class KNNClassifier(Model):
    """
    K-Nearest Neighbors Classifier.

    Parameters
    -------
    k : int, default=1
        O número de vizinhos mais próximos a considerar.
    distance : Callable, default=euclidean
        Função para calcular a distância entre amostras.
    """

    def __init__(self, k: int = 1, distance: Callable = euclidean, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.distance = distance
        self.dataset = None

    def _fit(self, dataset: Dataset) -> 'KNNClassifier':
        """
        Armazena o dataset de treino.

        Parameters
        -------
        dataset : Dataset
            O dataset de treino.

        Returns
        -------
        self : KNNClassifier
        """
        self.dataset = dataset
        return self

    def _get_closest_label(self, sample: np.ndarray) -> Union[int, str]:
        """
        Obtém a classe mais frequente de entre os k vizinhos mais próximos.
        """
        distances = np.apply_along_axis(self.distance, axis=1, arr=self.dataset.X, u=sample)

        k_nearest_indices = np.argsort(distances)[:self.k]

        k_nearest_labels = self.dataset.y[k_nearest_indices]

        labels, counts = np.unique(k_nearest_labels, return_counts=True)
        return labels[np.argmax(counts)]

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Prevê as classes para todas as amostras no dataset fornecido.

        Parameters
        -------
        dataset : Dataset
            O dataset de teste.

        Returns
        -------
        predictions : np.ndarray
            Array com as classes previstas.
        """
        predictions = np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)
        return predictions

    def _score(self, dataset: Dataset) -> float:
        """
        Calcula a precisão (accuracy) do modelo no dataset fornecido.

        Parameters
        -------
        dataset : Dataset
            O dataset a avaliar.

        Returns
        -------
        error : float
            Valor de accuracy entre as previsões e os valores reais.
        """
        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)