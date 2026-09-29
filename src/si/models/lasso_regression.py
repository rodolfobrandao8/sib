import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class LassoRegression(Model):
    """
    Modelo de Regressao Linear com regularizacao L1 (Lasso)
    e otimizador Gradient Descent com Momentum opcional.

    Parameters
    -------
    l1_penalty : float, default=1.0
        Parametro de regularizacao L1 (lambda).
    alpha : float, default=0.001
        Taxa de aprendizagem (learning rate).
    max_iter : int, default=1000
        Numero maximo de iteracoes.
    patience : int, default=5
        Numero maximo de iteracoes sem melhoria antes de parar.
    scale : bool, default=True
        Se True, normaliza as features com media e desvio padrao.
    momentum : float, default=0.0
        Fator de momentum (beta), tipicamente entre 0.0 e 0.9.
    """

    def __init__(self, l1_penalty: float = 1.0, alpha: float = 0.001,
                 max_iter: int = 1000, patience: int = 5, scale: bool = True,
                 momentum: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.l1_penalty = l1_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale
        self.momentum = momentum

        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = {}

    def _fit(self, dataset: Dataset) -> 'LassoRegression':
        """
        Treina o modelo Lasso estimando os coeficientes via Gradient Descent com Momentum.

        Parameters
        -------
        dataset : Dataset
            O dataset de treino.

        Returns
        -------
        self : LassoRegression
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

        v_theta = np.zeros(n)
        v_theta_zero = 0.0

        patience_count = 0
        best_cost = np.inf

        for i in range(self.max_iter):
            y_pred = np.dot(X, self.theta) + self.theta_zero
            error = y_pred - y

            grad_zero = np.sum(error) / m
            grad_theta = (np.dot(X.T, error) / m) + (self.l1_penalty * np.sign(self.theta) / m)

            v_theta_zero = self.momentum * v_theta_zero + self.alpha * grad_zero
            v_theta = self.momentum * v_theta + self.alpha * grad_theta

            self.theta_zero -= v_theta_zero
            self.theta -= v_theta

            current_cost = (np.sum((y_pred - y) ** 2) / (2 * m)) + (self.l1_penalty * np.sum(np.abs(self.theta)) / m)
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
        Gera previsoes para o dataset fornecido.

        Parameters
        -------
        dataset : Dataset
            Dataset para previsao.

        Returns
        -------
        np.ndarray
        np.ndarray
            Valores previstos.
        """
        if self.scale:
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        return np.dot(X, self.theta) + self.theta_zero

    def _score(self, dataset: Dataset) -> float:
        """
        Calcula o erro MSE nas previsoes.

        Parameters
        -------
        dataset : Dataset
            Dataset de teste.

        Returns
        -------
        float
            MSE.
        """
        y_pred = self.predict(dataset)
        return mse(dataset.y, y_pred)

    def cost(self, dataset: Dataset) -> float:
        """
        Calcula a funcao de custo com regularizacao L1.

        Parameters
        -------
        dataset : Dataset
            Dataset para avaliacao.

        Returns
        -------
        float
            Custo calculado.
        """
        m = dataset.shape()[0]
        y_pred = self.predict(dataset)
        l1_term = self.l1_penalty * np.sum(np.abs(self.theta))
        return float((np.sum((y_pred - dataset.y) ** 2) / (2 * m)) + (l1_term / m))