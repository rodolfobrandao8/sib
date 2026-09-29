import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import numpy as np
from si.data.dataset import Dataset
from si.model_selection.split import train_test_split
from si.models.linear_regression import RidgeRegression
from si.models.ridge_regression_least_squares import RidgeRegressionLeastSquares

file_path = "datasets/cpu/cpu.csv"

data = np.genfromtxt(file_path, delimiter=",", skip_header=1)

X = data[:, :-1]
y = data[:, -1]

with open(file_path, "r") as f:
    header = f.readline().strip().split(",")
features = header[:-1]
label = header[-1]

dataset = Dataset(X=X, y=y, features=features, label=label)

train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

ridge_gd = RidgeRegression(l2_penalty=1.0, alpha=0.001, max_iter=2000, patience=10, scale=True)
ridge_gd.fit(train_dataset)

mse_score_gd = ridge_gd.score(test_dataset)
final_cost_gd = ridge_gd.cost(test_dataset)

print("--- Ridge Regression (Gradient Descent) ---")
print(f"MSE Score: {mse_score_gd}")
print(f"Cost: {final_cost_gd}")

ridge_ls = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)
ridge_ls.fit(train_dataset)
mse_score_ls = ridge_ls.score(test_dataset)

print("\n--- Ridge Regression (Least Squares) ---")
print(f"MSE Score: {mse_score_ls}")