import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.sigmoid_function import sigmoid_function


class LogisticRegression(Model):
    """
    Modelo de Regressao Logistica com regularizacao L2
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

    def _fit(self, dataset: Dataset) -> 'LogisticRegression':
        """
        Treina o modelo de regressao logistica estimando os coeficientes theta e theta_zero.

        Parameters
        -------
        dataset : Dataset
            O dataset de treino.

        Returns
        -------
        self : LogisticRegression
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
            linear_output = np.dot(X, self.theta) + self.theta_zero
            y_pred = sigmoid_function(linear_output)
            error = y_pred - y

            gradient_zero = np.sum(error) / m
            gradient_theta = (np.dot(X.T, error) / m) + (self.l2_penalty * self.theta / m)

            self.theta_zero = self.theta_zero - self.alpha * gradient_zero
            self.theta = self.theta - self.alpha * gradient_theta

            updated_output = np.dot(X, self.theta) + self.theta_zero
            updated_probs = sigmoid_function(updated_output)

            eps = 1e-15
            probs_clipped = np.clip(updated_probs, eps, 1.0 - eps)

            cost_log = -np.mean(y * np.log(probs_clipped) + (1.0 - y) * np.log(1.0 - probs_clipped))
            reg_term = (self.l2_penalty / (2 * m)) * np.sum(self.theta ** 2)
            current_cost = cost_log + reg_term
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
        Preve as classes binarias (0 ou 1) para cada amostra do dataset.

        Parameters
        -------
        dataset : Dataset
            Dataset para previsao.

        Returns
        -------
        np.ndarray
            Classes previstas (0 ou 1).
        """
        if self.scale:
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        probabilities = sigmoid_function(np.dot(X, self.theta) + self.theta_zero)
        return (probabilities >= 0.5).astype(int)

    def _score(self, dataset: Dataset) -> float:
        """
        Calcula a acuracia entre as previsoes e os valores reais.

        Parameters
        -------
        dataset : Dataset
            Dataset de avaliacao.

        Returns
        -------
        float
            Valor da acuracia.
        """
        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)

    def cost(self, dataset: Dataset) -> float:
        """
        Calcula a funcao de custo J(theta) com regularizacao L2 num dado dataset.

        Parameters
        -------
        dataset : Dataset
            Dataset para avaliacao do custo.

        Returns
        -------
        float
            Valor da funcao de custo J(theta).
        """
        m = dataset.shape()[0]
        if self.scale:
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        probs = sigmoid_function(np.dot(X, self.theta) + self.theta_zero)
        eps = 1e-15
        probs_clipped = np.clip(probs, eps, 1.0 - eps)

        cost_log = -np.mean(dataset.y * np.log(probs_clipped) + (1.0 - dataset.y) * np.log(1.0 - probs_clipped))
        reg_term = (self.l2_penalty / (2 * m)) * np.sum(self.theta ** 2)
        return float(cost_log + reg_term)