import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegressionLeastSquares(Model):
    """
    Modelo de Regressao Linear com regularizacao L2 resolvido por Minimos Quadrados.

    Parameters
    -------
    l2_penalty : float, default=1.0
        Parametro de regularizacao L2 (lambda).
    scale : bool, default=True
        Se True, normaliza os dados com media e desvio padrao.
    """

    def __init__(self, l2_penalty: float = 1.0, scale: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.l2_penalty = l2_penalty
        self.scale = scale

        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def _fit(self, dataset: Dataset) -> 'RidgeRegressionLeastSquares':
        """
        Estima theta e theta_zero atraves da solucao analitica de Minimos Quadrados.

        Parameters
        -------
        dataset : Dataset
            O dataset de treino.

        Returns
        -------
        self : RidgeRegressionLeastSquares
        """
        m, n = dataset.shape()

        if self.scale:
            self.mean = np.nanmean(dataset.X, axis=0)
            self.std = np.nanstd(dataset.X, axis=0)
            self.std = np.where(self.std == 0, 1.0, self.std)
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        X_with_intercept = np.c_[np.ones(m), X]

        penalty_matrix = self.l2_penalty * np.eye(n + 1)

        penalty_matrix[0, 0] = 0.0

        thetas = np.linalg.inv(X_with_intercept.T.dot(X_with_intercept) + penalty_matrix).dot(X_with_intercept.T).dot(dataset.y)

        self.theta_zero = thetas[0]
        self.theta = thetas[1:]

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Preve os valores de y utilizando a multiplicacao matricial dos thetas estimados.

        Parameters
        -------
        dataset : Dataset
            Dataset para previsao.

        Returns
        -------
        predictions : np.ndarray
        predictions : np.ndarray
            Valores previstos.
        """
        m = dataset.shape()[0]

        if self.scale:
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        X_with_intercept = np.c_[np.ones(m), X]

        thetas = np.r_[self.theta_zero, self.theta]
        return X_with_intercept.dot(thetas)

    def _score(self, dataset: Dataset) -> float:
        """
        Calcula o erro MSE entre as previsoes e os valores reais.

        Parameters
        -------
        dataset : Dataset
            Dataset de avaliacao.

        Returns
        -------
        float
            Valor do erro MSE.
        """
        y_pred = self.predict(dataset)
        return mse(dataset.y, y_pred)