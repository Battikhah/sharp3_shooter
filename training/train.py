import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime

from model.vlstm import VLSTM
from model.losses import sharpe_loss


def train_model(train_dataset, val_dataset, config) -> nn.Module:
    """
    Trains VLSTM with early stopping on validation Sharpe loss.
    Returns the model with best validation performance.
    """
    num_tickers = len(config.TICKERS)
    model = VLSTM(
        config.NUM_FEATURES,
        config.HIDDEN_DIM,
        num_tickers,
        config.DROPOUT,
        config.NUM_LSTM_LAYERS,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, drop_last=False)

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = checkpoint_dir / f"{run_id}.pt"

    best_val_loss = float("inf")
    patience_count = 0
    best_state = None

    for epoch in range(config.EPOCHS):
        model.train()
        for X, _target, ticker_id, fwd_return, vs_factor in train_loader:
            optimizer.zero_grad()
            positions = model(X, ticker_id)
            loss = sharpe_loss(positions, fwd_return, vs_factor, config.TARGET_VOL)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, _target, ticker_id, fwd_return, vs_factor in val_loader:
                positions = model(X, ticker_id)
                val_loss = sharpe_loss(positions, fwd_return, vs_factor, config.TARGET_VOL)
                val_losses.append(val_loss.item())

        if not val_losses:
            continue

        mean_val_loss = sum(val_losses) / len(val_losses)

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_state, checkpoint_path)
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= config.PATIENCE:
                print(f"  Early stop at epoch {epoch + 1}")
                break

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1:3d}  val_loss={mean_val_loss:.4f}  patience={patience_count}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from training.dataset import FuturesDataset

    TICKERS_SMALL = ["SPY", "QQQ"]
    N_DAYS = 120

    class _FastConfig:
        TICKERS = TICKERS_SMALL
        NUM_FEATURES = 9
        HIDDEN_DIM = 32
        DROPOUT = 0.0
        NUM_LSTM_LAYERS = 1
        LEARNING_RATE = 1e-3
        BATCH_SIZE = 16
        EPOCHS = 6
        PATIENCE = 3
        TARGET_VOL = 0.1
        SEQUENCE_LENGTH = 20

    cfg = _FastConfig()
    tickers_2_id = {t: i for i, t in enumerate(cfg.TICKERS)}

    rng = np.random.default_rng(0)
    features_dict = {}
    for ticker in cfg.TICKERS:
        df = pd.DataFrame(
            rng.standard_normal((N_DAYS, cfg.NUM_FEATURES)),
            columns=[f"f{i}" for i in range(cfg.NUM_FEATURES)],
        )
        df["target"] = rng.standard_normal(N_DAYS)
        df["forward_return"] = rng.standard_normal(N_DAYS) * 0.01
        df["vs_factor"] = rng.standard_normal(N_DAYS) ** 2 + 1.0
        features_dict[ticker] = df

    train_ds = FuturesDataset(features_dict, cfg.SEQUENCE_LENGTH, tickers_2_id)
    val_ds = FuturesDataset(features_dict, cfg.SEQUENCE_LENGTH, tickers_2_id)

    assert len(train_ds) > 0, "Empty train dataset"
    print(f"Train dataset size: {len(train_ds)}")

    model = train_model(train_ds, val_ds, cfg)

    model.eval()
    X_s, _, tid_s, _, _ = train_ds[0]
    out = model(X_s.unsqueeze(0), tid_s.unsqueeze(0))
    assert out.shape == (1,), f"Bad output shape: {out.shape}"
    assert -1.0 <= out.item() <= 1.0, f"Output outside [-1, 1]: {out.item()}"
    print(f"Output shape: {out.shape}  PASS")
    print(f"Output value: {out.item():.4f}  PASS")
    print(f"Checkpoint saved: {list(Path('checkpoints').glob('*.pt'))}  PASS")
