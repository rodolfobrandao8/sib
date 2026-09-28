import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset


class VarianceThreshold(Transformer):
    """
    Seleciona features com base num limiar de variancia.

    Parameters
    -------
    threshold : float, default=0.0
        Valor de corte de variancia. Features com variancia menor ou igual serao removidas.
    """

    def __init__(self, threshold: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.variance = None

    def _fit(self, dataset: Dataset) -> 'VarianceThreshold':
        """
        Estima a variancia de cada feature.

        Parameters
        -------
        dataset : Dataset
            O dataset de entrada.

        Returns
        -------
        self : VarianceThreshold
        """
        self.variance = np.var(dataset.X, axis=0)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Seleciona as features com variancia superior ao threshold.

        Parameters
        -------
        dataset : Dataset
            O dataset a transformar.

        Returns
        -------
        Dataset
            Novo Dataset apenas com as features selecionadas.
        """
        mask = self.variance > self.threshold
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