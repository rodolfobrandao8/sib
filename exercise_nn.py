import numpy as np

X = np.random.rand(5, 2)
print(f"Shape de X: {X.shape}")

W1 = np.random.rand(2, 3) - 0.5
b1 = np.zeros((1, 3))

out1 = np.dot(X, W1) + b1
print(f"Shape da saída da Camada 1: {out1.shape}")

W2 = np.random.rand(3, 1) - 0.5
b2 = np.zeros((1, 1))

out2 = np.dot(out1, W2) + b2
print(f"Shape da saída final (Camada 2): {out2.shape}")
print("Saída final:\n", out2)