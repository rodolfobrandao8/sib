import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

import numpy as np
from si.neural_networks.layers import DenseLayer
layer = DenseLayer(4, (3,)).initialize()

X_dummy = np.random.rand(10, 3)

out = layer.forward_propagation(X_dummy)

assert out.shape == (10, 4), f"Erro na dimensão da saída: esperado (10, 4), obtido {out.shape}"
assert layer.parameters() == 3 * 4 + 4, f"Erro no cálculo de parâmetros: esperado 16, obtido {layer.parameters()}"
assert layer.output_shape() == (4,), f"Erro em output_shape: esperado (4,), obtido {layer.output_shape()}"

print("Validação concluída com sucesso!")
print(f"Shape da saída: {out.shape}")
print(f"Número de parâmetros: {layer.parameters()}")
print(f"Output shape registado: {layer.output_shape()}")