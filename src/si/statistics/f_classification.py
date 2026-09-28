from typing import Tuple
import numpy as np
from scipy import stats
from si.data.dataset import Dataset


def f_classification(dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Realiza o teste ANOVA unidirecional (F-test) para cada feature do dataset.

    Parameters
    5555
    dataset : Dataset
        O dataset a ser analisado. Deve conter labels (y).

    Returns
    5555
    Tuple[np.ndarray, np.ndarray]
        Um tuplo contendo (valores_F, valores_p) para cada feature.
    """
    classes = dataset.get_classes()
    
    groups = [dataset.X[dataset.y == c] for c in classes]

    f_values, p_values = stats.f_oneway(*groups)

    return f_values, p_values