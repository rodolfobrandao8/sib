from typing import Callable, Union
import numpy as np
from scipy.spatial.distance import euclidean

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse


class KNNRegressor(Model):
    """
    K-Nearest Neighbors Regressor.

    Parameters
    -------
    k : int, default=1
        O numero de vizinhos mais proximos a considerar.
    distance : Callable, default=euclidean
        Funcao para calcular a distancia entre amostras.
    """

    def __init__(self, k: int = 1, distance: Callable = euclidean, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.distance = distance
        self.dataset = None

    def _fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        Armazena o dataset de treino.

        Parameters
        -------
        dataset : Dataset
            O dataset de treino.

        Returns
        -------
        self : KNNRegressor
        """
        self.dataset = dataset
        return self

    def _get_closest_label(self, sample: np.ndarray) -> float:
        """
        Calcula a media dos valores de y dos k vizinhos mais proximos.
        """
        distances = np.apply_along_axis(self.distance, axis=1, arr=self.dataset.X, u=sample)

        k_nearest_indices = np.argsort(distances)[:self.k]

        k_nearest_values = self.dataset.y[k_nearest_indices]

        return np.mean(k_nearest_values)

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Preve os valores continuos para todas as amostras no dataset.

        Parameters
        -------
        dataset : Dataset
            O dataset de teste.

        Returns
        -------
        predictions : np.ndarray
            Array com os valores previstos.
        """
        predictions = np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)
        return predictions

    def _score(self, dataset: Dataset) -> float:
        """
        Calcula o erro RMSE do modelo no dataset fornecido.

        Parameters
        -------
        dataset : Dataset
            O dataset a avaliar.

        Returns
        -------
        error : float
            Valor de RMSE entre as previsoes e os valores reais.
        """
        y_pred = self.predict(dataset)
        return rmse(dataset.y, y_pred)