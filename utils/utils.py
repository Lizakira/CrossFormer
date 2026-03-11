import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

def build_cluster_indices(source_columns, cluser_dict):
    
    col_index = {g: i for i, g in enumerate(source_columns)}
    cluster_names, cluster_idx = [], []

    for p, c in cluser_dict.items():
        idx = [col_index[g] for g in c if g in col_index]
    
        cluster_names.append(p)
        cluster_idx.append(np.array(idx, dtype=np.int64))
    return cluster_names, cluster_idx

def load_gmt(path):
    x_sets = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            name = parts[0]
            c = parts[2:]
            x_sets[name] = c
    return x_sets


class PairedOmicsDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])