from typing import Callable
import numpy as np

from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectPercentile(Transformer):
    """
    Seleciona uma percentagem de caracteristicas com base nos scores estatisticos de F.

    Parameters
    -------
    score_func : Callable, default=f_classification
        Funcao de analise estatistica de variancia.
    percentile : float, default=10.0
        Percentil de caracteristicas a selecionar (de 0 a 100).
    """

    def __init__(self, score_func: Callable = f_classification, percentile: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        Estima os valores de F e p atraves da score_func.

        Parameters
        -------
        dataset : Dataset
            O dataset de treino.

        Returns
        -------
        self : SelectPercentile
        """
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Seleciona as caracteristicas com base no percentil e lida com empates.

        Parameters
        -------
        dataset : Dataset
            O dataset a transformar.

        Returns
        -------
        Dataset
            Novo Dataset apenas com as caracteristicas selecionadas.
        """
        n_features = dataset.shape()[1]
        k = int(np.ceil(n_features * (self.percentile / 100.0)))

        threshold = np.percentile(self.F, 100 - self.percentile)

        mask = self.F > threshold

        if np.sum(mask) < k:
            tied_indices = np.where(self.F == threshold)[0]
            needed = k - np.sum(mask)
            mask[tied_indices[:needed]] = True

        new_X = dataset.X[:, mask]

        new_features = None
        if dataset.features is not None:
            new_features = np.array(dataset.features)[mask].tolist()

        return Dataset(
            X=new_X,
            y=dataset.y,
            features=new_features,
            label=dataset.label
        )