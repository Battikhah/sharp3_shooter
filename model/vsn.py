import torch
import torch.nn as nn


class VariableSelectionNetwork(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int, ticker_embed_dim: int | None = None):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.ticker_embed_dim = ticker_embed_dim

        # Nonlinear per-feature embedding (paper Appendix B.5 Eq. 44: "nonlinear embedding function")
        self.feature_embeddings = nn.ModuleList([
            nn.Sequential(nn.Linear(1, hidden_dim), nn.ELU())
            for _ in range(num_features)
        ])

        # Gating network — optionally conditioned on ticker identity so each asset
        # learns its own feature importance weights
        gating_in = num_features * hidden_dim + (ticker_embed_dim or 0)
        self.gating_network = nn.Sequential(
            nn.Linear(gating_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_features),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor, ticker_emb: torch.Tensor | None = None) -> torch.Tensor:
        # x:          (batch, seq_len, num_features)
        # ticker_emb: (batch, ticker_embed_dim) or None
        batch_size, seq_len, _ = x.shape

        embedded = torch.stack(
            [self.feature_embeddings[i](x[:, :, i:i + 1]) for i in range(self.num_features)],
            dim=2,
        )  # (batch, seq_len, num_features, hidden_dim)

        flat = embedded.view(batch_size, seq_len, -1)  # (batch, seq_len, num_features * hidden_dim)

        if ticker_emb is not None and self.ticker_embed_dim is not None:
            # Broadcast ticker identity across every timestep in the window
            ticker_expanded = ticker_emb.unsqueeze(1).expand(-1, seq_len, -1)
            gating_input = torch.cat([flat, ticker_expanded], dim=-1)
        else:
            gating_input = flat

        weights = self.gating_network(gating_input)  # (batch, seq_len, num_features)
        return (weights.unsqueeze(-1) * embedded).sum(dim=2)  # (batch, seq_len, hidden_dim)


if __name__ == "__main__":
    from config import NUM_FEATURES, HIDDEN_DIM, SEQUENCE_LENGTH

    batch, seq_len = 4, SEQUENCE_LENGTH
    vsn = VariableSelectionNetwork(NUM_FEATURES, HIDDEN_DIM, ticker_embed_dim=HIDDEN_DIM)
    x = torch.randn(batch, seq_len, NUM_FEATURES)
    ticker_emb = torch.randn(batch, HIDDEN_DIM)

    out = vsn(x, ticker_emb)

    assert out.shape == (batch, seq_len, HIDDEN_DIM), f"Bad output shape: {out.shape}"

    flat_emb = torch.stack(
        [vsn.feature_embeddings[i](x[:, :, i:i + 1]) for i in range(NUM_FEATURES)], dim=2
    )
    flat_cat = flat_emb.view(batch, seq_len, -1)
    ticker_expanded = ticker_emb.unsqueeze(1).expand(-1, seq_len, -1)
    gating_in = torch.cat([flat_cat, ticker_expanded], dim=-1)
    weights = vsn.gating_network(gating_in)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch, seq_len), atol=1e-5), \
        "Gating weights don't sum to 1"

    out.sum().backward()
    assert all(p.grad is not None for p in vsn.parameters()), "Some parameters have no gradient"

    print(f"Output shape: {out.shape}  PASS")
    print(f"Gating weights sum to 1  PASS")
    print(f"Gradients flow  PASS")
