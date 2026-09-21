import numpy as np
from data.dataset import Dataset

def read_data_file(filename, sep=',', label=False):
    """Lê um ficheiro de dados usando numpy e retorna um objeto Dataset."""
    data = np.genfromtxt(filename, delimiter=sep)
    
    if label:
        X = data[:, :-1]
        y = data[:, -1]
    else:
        X = data
        y = None
        
    return Dataset(X=X, y=y)

def write_data_file(filename, dataset, sep=',', label=False):
    """Escreve um objeto Dataset num ficheiro de dados usando numpy."""
    if label and dataset.has_label():
        data = np.column_stack((dataset.X, dataset.y))
    else:
        data = dataset.X
        
    np.savetxt(filename, data, delimiter=sep)