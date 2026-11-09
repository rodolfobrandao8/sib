import sys
from pathlib import Path

root_src = Path(__file__).resolve().parent / "src"
if str(root_src) not in sys.path:
    sys.path.insert(0, str(root_src))

import numpy as np
from si.data.dataset import Dataset
from si.neural_networks.neural_network import NeuralNetwork
from si.neural_networks.layers import DenseLayer
from si.neural_networks.activation import ReLUActivation, SigmoidActivation
from si.neural_networks.optimizers import SGD
from si.neural_networks.losses import BinaryCrossEntropy


def accuracy_binary(y_true, y_pred):
    preds = (y_pred >= 0.5).astype(int).flatten()
    return float(np.mean(y_true.flatten() == preds))

np.random.seed(42)
n_samples = 200
n_features = 32

X = np.random.randn(n_samples, n_features)
y = ((X[:, :5].sum(axis=1) + np.random.randn(n_samples) * 0.3) > 0).astype(float)


def create_model():
    model = NeuralNetwork(
        epochs=100,
        batch_size=16,
        optimizer=SGD,
        learning_rate=0.01,
        loss=BinaryCrossEntropy,
        metric=accuracy_binary,
        verbose=False
    )
    # Camada 1: reduz de 32 para 16
    model.add(DenseLayer(n_units=16, input_shape=(32,)))
    model.add(ReLUActivation())
    # Camada 2: reduz de 16 para 8
    model.add(DenseLayer(n_units=8))
    model.add(ReLUActivation())
    # Camada de saida: 8 para 1
    model.add(DenseLayer(n_units=1))
    model.add(SigmoidActivation())
    return model


if __name__ == "__main__":
    print("Execucao do Exercicio 16: K-Fold Cross-Validation (k=5)")
    k = 5
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)

    scores = []
    for fold_idx, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(indices, test_idx)

        train_ds = Dataset(X=X[train_idx], y=y[train_idx])
        test_ds = Dataset(X=X[test_idx], y=y[test_idx])

        model = create_model()
        model.fit(train_ds)

        acc = model.score(test_ds)
        scores.append(acc)
        print(f"Fold {fold_idx + 1}/{k} - Exactidao: {acc:.4f}")

    print(f"\nExactidao Media nos 5 folds: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")