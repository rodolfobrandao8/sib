from abc import ABC, abstractmethod
from si.base.estimator import Estimator
from si.data.dataset import Dataset


class Model(Estimator, ABC):
    """
    Abstract base class for models.
    A model is an object that can predict the target values of a Dataset object.
    """

    def __init__(self, **kwargs):
        """
        Initialize the model.
        """
        super().__init__(**kwargs)

    def predict(self, dataset: Dataset):
        """
        Predict the target values of the dataset.
        The model needs to be fitted before calling this method.

        Parameters
        -------
        dataset: Dataset
            The dataset to predict the target values of.

        Returns
        -------
        predictions: np.ndarray
            The predicted target values.
        """
        if not self.is_fitted:
            raise ValueError('Model needs to be fitted before calling predict()')
        return self._predict(dataset)

    @abstractmethod
    def _predict(self, dataset: Dataset):
        """
        Predict the target values of the dataset.
        Abstract method that needs to be implemented by all subclasses.

        Parameters
        -------
        dataset: Dataset
            The dataset to predict the target values of.

        Returns
        -------
        predictions: np.ndarray
            The predicted target values.
        """
        pass

    def fit_predict(self, dataset: Dataset):
        """
        Fit the model to the dataset and predict the target values.
        Equivalent to calling fit(dataset) and then predict(dataset).

        Parameters
        -------
        dataset: Dataset
            The dataset to fit and predict the target values of.

        Returns
        -------
        predictions: np.ndarray
            The predicted target values.
        """
        self.fit(dataset)
        return self.predict(dataset)

    @abstractmethod
    def _score(self, dataset: Dataset):
        """
        Abstract method to calculate the error/score of the model on a given dataset.

        Parameters
        -------
        dataset: Dataset
            The dataset to evaluate the model on.

        Returns
        -------
        score: float
            The calculated error/score.
        """
        pass

    def score(self, dataset: Dataset):
        """
        Verifies if the model is fitted and calculates the error metric.

        Parameters
        ------
        dataset: Dataset
            The dataset to evaluate the model on.

        Returns
        -------
        score: float
            The calculated error/score.
        """
        if not self.is_fitted:
            raise ValueError('Model needs to be fitted before calling score()')
        return self._score(dataset)