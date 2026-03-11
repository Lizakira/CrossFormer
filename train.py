import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from utils.utils import build_cluster_indices, PairedOmicsDataset
from architecture.TransformerGenerator import TransformerGenerator

def train(
        
    modality_source_train: pd.DataFrame,
    modality_target_train: pd.DataFrame,
    cluster_dict: dict,
    device="cuda" if torch.cuda.is_available() else "cpu",
    d_model=128,
    n_heads=8,
    n_layers=2,
    n_queries=32,
    z_dim=64,
    tok_hidden=256,
    dropout=0.1,
    lr=3e-4,
    weight_decay=1e-4,
    batch_size=32,
    epochs=100
):
    assert (modality_source_train.index == modality_target_train.index).all(), "modality_source_train and modality_target_train must be aligned by index."

    cluster_names, cluster_idx = build_cluster_indices(modality_source_train.columns, cluster_dict)
    print("Using pathways:", len(cluster_names))

    # scale X and Y
    x_imputer = SimpleImputer(strategy="mean")
    y_imputer = SimpleImputer(strategy="mean")
    
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X = x_imputer.fit_transform(modality_source_train.values)
    Y = y_imputer.fit_transform(modality_target_train.values)

    X = x_scaler.fit_transform(X).astype(np.float32)
    Y = y_scaler.fit_transform(Y).astype(np.float32)

    ds = PairedOmicsDataset(X, Y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = TransformerGenerator(
        cluster_idx=cluster_idx,
        y_dim=Y.shape[1],
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        n_queries=n_queries,
        z_dim=z_dim,
        tok_hidden=tok_hidden,
        dropout=dropout
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            y_hat = model(xb)

            loss = F.smooth_l1_loss(y_hat, yb, beta=1.0)
            
            opt.zero_grad()
            loss.backward()
            
            opt.step()

            bs = xb.size(0)
            total_loss += loss.item() * bs
            n += bs

        print(f"Epoch {epoch:03d} | loss {total_loss/n:.4f}")

    return model, x_scaler, x_imputer, y_scaler,y_imputer, cluster_names