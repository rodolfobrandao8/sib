import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import numpy as np
from si.data.dataset import Dataset
from si.model_selection.split import train_test_split
from si.models.logistic_regression import LogisticRegression

file_path = "datasets/breast_bin/breast-bin.csv"

data = np.genfromtxt(file_path, delimiter=",", skip_header=1)
X = data[:, :-1]
y = data[:, -1]

with open(file_path, "r") as f:
    header = f.readline().strip().split(",")
features = header[:-1]
label = header[-1]

dataset = Dataset(X=X, y=y, features=features, label=label)

train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

log_reg = LogisticRegression(l2_penalty=1.0, alpha=0.001, max_iter=2000, patience=10, scale=True)
log_reg.fit(train_dataset)

acc_score = log_reg.score(test_dataset)
final_cost = log_reg.cost(test_dataset)

print("--- Logistic Regression ---")
print(f"Accuracy Score: {acc_score:.4f}")
print(f"Cost: {final_cost:.4f}")