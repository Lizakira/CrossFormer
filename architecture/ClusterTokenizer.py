import torch
import torch.nn as nn

class ClusterTokenizer(nn.Module):

    """
    Converts the source  feature vector into a sequence of
    cluster-based token embeddings for use in a Transformer encoder.

    Parameters:

    cluster_idx : list of list of int
        Indices of input features grouped into clusters. Each sublist
        defines one cluster. For example, [[0, 1, 2], [3, 4]] defines
        two clusters of sizes 3 and 2 respectively.
    d_model : int, optional
        Output embedding dimension for each token. Default: 128.
    hidden : int, optional
        Hidden layer size in the MLP encoder for each cluster. Default: 256.
    dropout : float, optional
        Dropout rate applied after the first linear layer. Default: 0.1
    """

    

    def __init__(self, cluster_idx, d_model=128, hidden=256, dropout=0.1):
        super().__init__()
        self.cluster_idx = cluster_idx
        self.P = len(cluster_idx)

        self.encoders = nn.ModuleList()
        for idx in cluster_idx:
            Gp = len(idx)
            self.encoders.append(nn.Sequential(
                nn.Linear(Gp, hidden),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden, d_model),
            )
            )

        self.ln = nn.LayerNorm(d_model)

    def forward(self, x_genes):
        toks = []
        for p, idx in enumerate(self.cluster_idx):
            xp = x_genes[:, idx]          
            tp = self.encoders[p](xp)     
            toks.append(tp.unsqueeze(1))  
        T = torch.cat(toks, dim=1)        
        return self.ln(T)