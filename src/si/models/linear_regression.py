import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegression(Model):
    """
    Modelo de Regressao Linear com regularizacao L2 (Ridge Regression)
    otimizado via Gradient Descent.

    Parameters
    -------
    l2_penalty : float, default=1.0
        Parametro de regularizacao L2 (lambda).
    alpha : float, default=0.001
        Taxa de aprendizagem (learning rate).
    max_iter : int, default=1000
        Numero maximo de iteracoes do Gradient Descent.
    patience : int, default=5
        Numero maximo de iteracoes consecutivas sem melhoria no custo antes de parar.
    scale : bool, default=True
        Se True, normaliza os dados com media e desvio padrao.
    """

    def __init__(self, l2_penalty: float = 1.0, alpha: float = 0.001,
                 max_iter: int = 1000, patience: int = 5, scale: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale

        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = {}

    def _fit(self, dataset: Dataset) -> 'RidgeRegression':
        """
        Treina o modelo Ridge Regression estimando os coeficientes theta e theta_zero.

        Parameters
        -------
        dataset : Dataset
            O dataset de treino.

        Returns
        -------
        self : RidgeRegression
        """
        m, n = dataset.shape()

        if self.scale:
            self.mean = np.nanmean(dataset.X, axis=0)
            self.std = np.nanstd(dataset.X, axis=0)
            self.std = np.where(self.std == 0, 1.0, self.std)
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        y = dataset.y

        self.theta = np.zeros(n)
        self.theta_zero = 0.0
        self.cost_history = {}

        patience_count = 0
        best_cost = np.inf

        for i in range(self.max_iter):
            y_pred = np.dot(X, self.theta) + self.theta_zero
            error = y_pred - y

            gradient_zero = np.sum(error) / m

            gradient_theta = (np.dot(X.T, error) / m) + (self.l2_penalty * self.theta / m)

            self.theta_zero = self.theta_zero - self.alpha * gradient_zero
            self.theta = self.theta - self.alpha * gradient_theta

            current_cost = (np.sum((y_pred - y) ** 2) + self.l2_penalty * np.sum(self.theta ** 2)) / (2 * m)
            self.cost_history[i] = current_cost

            if current_cost < best_cost:
                best_cost = current_cost
                patience_count = 0
            else:
                patience_count += 1

            if patience_count >= self.patience:
                break

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Preve a variavel dependente y utilizando os coeficientes estimados.

        Parameters
        -------
        dataset : Dataset
            Dataset para previsao.

        Returns
        -------

        predictions : np.ndarray
            Valores previstos.
        """
        if self.scale:
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        return np.dot(X, self.theta) + self.theta_zero

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

    def cost(self, dataset: Dataset) -> float:
        """
        Calcula a funcao de custo J(theta) com regularizacao L2 num dado dataset.

        Parameters
        -------
        dataset : Dataset
            Dataset para calcular a funcao de custo.

        Returns
        -------
        float
            Valor da funcao de custo J(theta).
        """
        m = dataset.shape()[0]
        y_pred = self.predict(dataset)
        reg_term = self.l2_penalty * np.sum(self.theta ** 2)
        return float((np.sum((y_pred - dataset.y) ** 2) + reg_term) / (2 * m))