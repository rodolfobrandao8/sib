from typing import Tuple
import numpy as np
from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = None) -> Tuple[Dataset, Dataset]:
    """
    Divide um objeto Dataset em dois datasets: treino e teste.

    Parameters
    -------
    dataset : Dataset
        O objeto Dataset a ser dividido.
    test_size : float, default=0.2
        A proporção do dataset a alocar para o conjunto de teste (ex.: 0.2 para 20%).
    random_state : int, default=None
        Semente para o gerador de números aleatórios (para reprodutibilidade).

    Returns
    -------
    Tuple[Dataset, Dataset]
        Um tuplo com (dataset_de_treino, dataset_de_teste).
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = dataset.shape()[0]
    n_test = int(n_samples * test_size)

    permutations = np.random.permutation(n_samples)

    test_indices = permutations[:n_test]
    train_indices = permutations[n_test:]

    train_X = dataset.X[train_indices]
    train_y = dataset.y[train_indices] if dataset.has_label() else None
    train_dataset = Dataset(X=train_X, y=train_y, features=dataset.features, label=dataset.label)

    test_X = dataset.X[test_indices]
    test_y = dataset.y[test_indices] if dataset.has_label() else None
    test_dataset = Dataset(X=test_X, y=test_y, features=dataset.features, label=dataset.label)

    return train_dataset, test_dataset


def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = None) -> Tuple[Dataset, Dataset]:
    """
    Divide um objeto Dataset de forma estratificada mantendo a proporção de cada classe.

    Parameters
    -------
    dataset : Dataset
        O objeto Dataset a ser dividido.
    test_size : float, default=0.2
        A proporção do dataset a alocar para o conjunto de teste.
    random_state : int, default=None
        Semente para o gerador de números aleatórios.

    Returns
    -------
    Tuple[Dataset, Dataset]
        Um tuplo com (dataset_de_treino, dataset_de_teste).
    """
    if random_state is not None:
        np.random.seed(random_state)

    labels, counts = np.unique(dataset.y, return_counts=True)

    train_indices = []
    test_indices = []

    for label, count in zip(labels, counts):
        label_indices = np.where(dataset.y == label)[0]
        np.random.shuffle(label_indices)

        n_test = int(count * test_size)

        test_indices.extend(label_indices[:n_test])
        train_indices.extend(label_indices[n_test:])

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    train_dataset = Dataset(
        X=dataset.X[train_indices],
        y=dataset.y[train_indices],
        features=dataset.features,
        label=dataset.label
    )
    test_dataset = Dataset(
        X=dataset.X[test_indices],
        y=dataset.y[test_indices],
        features=dataset.features,
        label=dataset.label
    )

    return train_dataset, test_dataset