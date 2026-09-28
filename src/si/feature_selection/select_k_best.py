from typing import Callable
import numpy as np

from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectKBest(Transformer):
    """
    Seleciona as k variaveis com maiores valores estatisticos F.

    Parameters
    -------
    score_func : Callable, default=f_classification
        Funcao de analise estatistica de variancia.
    k : int, default=10
        Numero de melhores features a selecionar.
    """

    def __init__(self, k: int = 10, score_func: Callable = f_classification, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.score_func = score_func
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset) -> 'SelectKBest':
        """
        Estima os valores de F e p para cada variavel usando a funcao de pontuacao.

        Parameters
        -------
        dataset : Dataset
            O dataset de treino.

        Returns
        -------
        self : SelectKBest
        """
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Seleciona as top k features com maiores valores de F.

        Parameters
        -------
        dataset : Dataset
            O dataset a transformar.

        Returns
        -------
        Dataset
            Novo Dataset apenas com as k features selecionadas.
        """
        top_k_indices = np.argsort(self.F)[-self.k:][::-1]

        new_X = dataset.X[:, top_k_indices]

        new_features = None
        if dataset.features is not None:
            new_features = [dataset.features[i] for i in top_k_indices]

        return Dataset(
            X=new_X,
            y=dataset.y,
            features=new_features,
            label=dataset.label
        )