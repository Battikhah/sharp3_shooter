import torch.nn as nn
import torch

class VariableSelectionNetwork(nn.Module):
    def __init__(self, num_features, hidden_dim):
        # Per-feature embedding: Scaler feature to hidden_dim vector
        super(VariableSelectionNetwork, self).__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(num_features)
        ])

        # Gating Network: concatenated features to num_features weights (softmax weights)
        self.gating_network = nn.Sequential(
            nn.Linear(num_features * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_features),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # X Shape
        batch_size, seq_len, num_features = x.shape

        # 1. Per-feature embedding
        embedded_features = []
        for i in range(self.num_features):
            embedded = self.feature_embeddings[i](x[:, :, i:i+1])
            embedded_features.append(embedded)
        embedded_features = torch.stack(embedded_features, dim=2) # Shape: (batch, seq_len, num_features, hidden_dim)
        
        # 2. Gating Network
        flat_embedded = embedded_features.view(batch_size, seq_len, -1) # Shape: (batch, seq_len, num_features * hidden_dim)
        gating_weights = self.gating_network(flat_embedded) # Shape: (batch, seq_len, num_features)

        # 3. Weighted Sum of Embedded Features
        weighted_features = (gating_weights.unsqueeze(-1) * embedded_features).sum(dim=2) # Shape: (batch, seq_len, hidden_dim)

        # Return batch, seq_len, hidden_dim
        return weighted_features


if __name__ == "__main__":
    from config import NUM_FEATURES, HIDDEN_DIM, SEQUENCE_LENGTH

    batch, seq_len = 4, SEQUENCE_LENGTH
    vsn = VariableSelectionNetwork(NUM_FEATURES, HIDDEN_DIM)
    x = torch.randn(batch, seq_len, NUM_FEATURES)

    out = vsn(x)

    assert out.shape == (batch, seq_len, HIDDEN_DIM), f"Bad output shape: {out.shape}"

    # Gating weights must sum to 1 across features at every timestep
    flat = x.view(batch, seq_len, -1)
    flat_emb = torch.stack([vsn.feature_embeddings[i](x[:, :, i:i+1]) for i in range(NUM_FEATURES)], dim=2)
    flat_cat = flat_emb.view(batch, seq_len, -1)
    weights = vsn.gating_network(flat_cat)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch, seq_len), atol=1e-5), "Gating weights don't sum to 1"

    # Gradients must flow back through the network
    out.sum().backward()
    assert all(p.grad is not None for p in vsn.parameters()), "Some parameters have no gradient"

    print(f"Output shape: {out.shape}  PASS")
    print(f"Gating weights sum to 1  PASS")
    print(f"Gradients flow  PASS")
