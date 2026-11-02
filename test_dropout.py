import numpy as np
from si.neural_networks.layers import Dropout

np.random.seed(42)

X = np.ones((5, 10))
dropout_layer = Dropout(probability=0.5)

out_train = dropout_layer.forward_propagation(X, training=True)
print(" Modo de Treino (training=True) ")
print("Saída do Dropout (amostra):\n", out_train[0])
print(f"Número de zeros gerados: {np.sum(out_train == 0)} de {out_train.size}")
print(f"Valores não nulos reescalados para: {np.unique(out_train[out_train != 0])}")

grad = np.ones_like(X)
back_err = dropout_layer.backward_propagation(grad)
print("\nBackward Propagation (após forward de treino)")
print(f"O gradiente foi 0 onde o neurónio foi desligado? {np.all((back_err == 0) == (out_train == 0))}")

out_eval = dropout_layer.forward_propagation(X, training=False)
print("\nModo de Inferência (training=False)")
print(f"A saída é idêntica à entrada? {np.all(out_eval == X)}")