from architecture.ClusterTokenizer import ClusterTokenizer
import torch
import torch.nn as nn

class TransformerGenerator(nn.Module):
    """
    CrossFormer: a modality-agnostic transformer-based generator for
    cross-modal generation in tabular multimodal data.

    Given a source omics profile x, the model generates a target profile y_hat.

    Parameters:
    
    cluster_idx : list of list of int
        Indices of input features grouped into clusters. Each sublist
        defines one cluster, which will be encoded as a single token.
    y_dim : int
        Dimensionality of the target modality (number of output features).
    d_model : int, optional
        Embedding dimension used throughout the transformer. Default: 128.
    n_heads : int, optional
        Number of attention heads in all multi-head attention layers. Default: 8.
    n_layers : int, optional
        Number of layers in both the encoder and the query transformer. Default: 2.
    n_queries : int, optional
        Number of learnable target profile queries. Default: 32.
    z_dim : int
        Dimensionality of the latent noise vector z. Default: 64.
    tok_hidden : int, optional
        Hidden size of the MLP used in the cluster tokenizer. Default: 256.
    dropout : float, optional
        Dropout rate applied throughout the architecture. Default: 0.3.
    """

    def __init__(
        self,
        cluster_idx,
        y_dim,
        d_model=128,
        n_heads=8,
        n_layers=2,
        n_queries=32,
        z_dim=64,
        tok_hidden=256,
        dropout=0.3
    ):
        super().__init__()
        self.y_dim = y_dim
        self.d_model = d_model
        self.z_dim = z_dim
        self.n_queries = n_queries

        self.tokenizer = ClusterTokenizer(
            cluster_idx=cluster_idx,
            d_model=d_model,
            hidden=tok_hidden,
            dropout=dropout
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.law_pool = nn.Linear(d_model, d_model)
        self.z_mu = nn.Linear(d_model, z_dim)
        self.z_logvar = nn.Linear(d_model, z_dim)

        # Target Profile queries
        self.object_queries = nn.Parameter(torch.randn(n_queries, d_model))
        
        self.q_proj = nn.Linear(z_dim, d_model)
        self.q_ln = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.cross_att = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
            dropout=dropout
        )

        q_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4*d_model,
            batch_first=True,
            dropout=dropout,
            activation="relu"
        )
        self.transformer = nn.TransformerEncoder(q_layer, num_layers=n_layers)

        self.out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, y_dim)
        )

        self.pool_w = nn.Linear(d_model,1)

    def encode(self, x):
        T = self.tokenizer(x)   # (B, P, d)
        H = self.encoder(T)           # (B, P, d)
        return H

    def sample_z(self, H):

        h = H.mean(dim=1)
        nq = self.n_queries
        law = self.law_pool(h)
        
        mu = self.z_mu(law).unsqueeze(1).expand(-1, nq, -1)
        logvar = self.z_logvar(law).unsqueeze(1).expand(-1, nq, -1)
        eps = torch.randn_like(mu)

        z = mu + eps * torch.exp(0.5 * logvar)

        return z

    def decode(self, H, z):

        B = H.size(0)
        Q0 = self.object_queries.unsqueeze(0).expand(B, -1, -1) 
        Q = Q0 + self.q_proj(z)

        attn_out, _ = self.cross_att(Q, H, H)
        Q = self.q_ln(Q + attn_out)
        Q = self.dropout(Q)

        Q = self.transformer(Q)

        weights = torch.softmax(self.pool_w(Q).squeeze(-1), dim=1)
        pooled = (Q*weights.unsqueeze(-1)).sum(dim=1)
        y_hat = self.out(pooled)

        return y_hat

    def forward(self, x):

        H = self.encode(x)
        z = self.sample_z(H)
        y_hat = self.decode(H, z)

        return y_hat
    
    @torch.no_grad()
    def generate(self, x, n_samples=10):
        """
        x: (B, G) float
        returns: (n_samples, B, y_dim) in SCALED space
        """
        H = self.encode(x)
        outs = []
        for _ in range(n_samples):
            z  = self.sample_z(H)
            y_hat = self.decode(H, z)
            outs.append(y_hat)
        return torch.stack(outs, dim=0)