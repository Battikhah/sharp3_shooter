import torch
import torch.nn as nn
from model.vsn import VariableSelectionNetwork


class VLSTM(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        num_tickers: int,
        dropout: float,
        num_lstm_layers: int = 1,
    ):
        super().__init__()
        self.ticker_embedding = nn.Embedding(num_tickers, hidden_dim)
        # VSN now conditioned on ticker identity so each asset learns its own feature weights
        self.vsn = VariableSelectionNetwork(num_features, hidden_dim, ticker_embed_dim=hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )
        # Temporal attention: learns which timesteps in the lookback window matter most
        self.temporal_attn = nn.Linear(hidden_dim, 1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, ticker_id: torch.Tensor) -> torch.Tensor:
        # x:         (batch, seq_len, num_features)
        # ticker_id: (batch,)

        ticker_emb = self.ticker_embedding(ticker_id)           # (batch, hidden_dim)

        vsn_out = self.vsn(x, ticker_emb)                       # (batch, seq_len, hidden_dim)

        # Ticker identity also added to every LSTM timestep (in addition to VSN conditioning)
        lstm_in = vsn_out + ticker_emb.unsqueeze(1)             # (batch, seq_len, hidden_dim)

        lstm_out, _ = self.lstm(lstm_in)                        # (batch, seq_len, hidden_dim)

        # Soft attention over the full lookback window — avoids relying solely on the last step
        attn_w = torch.softmax(self.temporal_attn(lstm_out), dim=1)  # (batch, seq_len, 1)
        attended = (attn_w * lstm_out).sum(dim=1)               # (batch, hidden_dim)

        out = self.head(self.norm(attended))                    # (batch, 1)
        return torch.tanh(out).squeeze(-1)                      # (batch,) in [-1, 1]


if __name__ == "__main__":
    from config import NUM_FEATURES, HIDDEN_DIM, DROPOUT, SEQUENCE_LENGTH, TICKERS, NUM_LSTM_LAYERS

    num_tickers = len(TICKERS)
    model = VLSTM(NUM_FEATURES, HIDDEN_DIM, num_tickers, DROPOUT, NUM_LSTM_LAYERS)

    batch_size = 4
    x = torch.randn(batch_size, SEQUENCE_LENGTH, NUM_FEATURES)
    ticker_id = torch.randint(0, num_tickers, (batch_size,))

    out = model(x, ticker_id)

    assert out.shape == (batch_size,), f"Bad output shape: {out.shape}"
    assert out.min() >= -1 and out.max() <= 1, f"Output outside [-1, 1]: [{out.min():.4f}, {out.max():.4f}]"

    out.sum().backward()
    assert all(p.grad is not None for p in model.parameters()), "Some parameters have no gradient"

    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 500_000, f"Parameter count unexpectedly low: {n_params:,}"

    print(f"Output shape: {out.shape}  PASS")
    print(f"Output range: [{out.min().item():.4f}, {out.max().item():.4f}]  PASS")
    print(f"Gradients flow  PASS")
    print(f"Parameter count: {n_params:,}  PASS")
