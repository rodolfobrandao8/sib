import pandas as pd
from data.dataset import Dataset 

def read_csv(filename, sep=',', features=False, label=False):
    """Lê um ficheiro CSV e retorna um objeto Dataset."""
    df = pd.read_csv(filename, sep=sep, header=0 if features else None)
    
    if label:
        X = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()
    else:
        X = df.to_numpy()
        y = None
        
    feature_names = None
    label_name = None
    
    if features:
        if label:
            feature_names = df.columns[:-1].tolist()
            label_name = df.columns[-1]
        else:
            feature_names = df.columns.tolist()
            
    return Dataset(X=X, y=y, features=feature_names, label=label_name)

def write_csv(filename, dataset, sep=',', features=False, label=False):
    """Escreve um objeto Dataset num ficheiro CSV."""
    df = pd.DataFrame(dataset.X)
    
    if features and dataset.features is not None:
        df.columns = dataset.features
        
    if label and dataset.has_label():
        label_col = dataset.label if dataset.label is not None else 'y'
        df[label_col] = dataset.y
        
    df.to_csv(filename, sep=sep, index=False, header=features)