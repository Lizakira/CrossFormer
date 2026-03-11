import torch
import pandas as pd
import numpy as np

@torch.no_grad()

def infer_and_generate(
    model,
    x_imputer,
    x_scaler,
    y_scaler,
    modality_1_test: pd.DataFrame,      
    target_columns=None,             
    n_samples: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):

    model.eval()

    # Scale X_test
    X = x_imputer.transform(modality_1_test.values) 
    X = x_scaler.transform(X).astype(np.float32)  
    xb = torch.from_numpy(X).to(device)

    # Generate S samples (scaled space)
    ys_scaled = model.generate(xb, n_samples=n_samples) 
    ys_scaled_np = ys_scaled.cpu().numpy()

    # Inverse-transform each sample
    S, N, M = ys_scaled_np.shape
    ys_inv = np.empty_like(ys_scaled_np, dtype=np.float32)
    for s in range(S):
        ys_inv[s] = y_scaler.inverse_transform(ys_scaled_np[s])

    y_mean = ys_inv.mean(axis=0)  

    if target_columns is None:
        target_columns = [f"target_{j}" for j in range(M)]

    y_hat = pd.DataFrame(y_mean, index=modality_1_test.index, columns=target_columns)

    return y_hat