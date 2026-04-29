import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime

from model.vlstm import VLSTM
from model.losses import portfolio_sharpe_loss


def train_model(train_dataset, val_dataset, config) -> tuple[nn.Module, float]:
    """
    Train VLSTM with portfolio-level Sharpe loss and early stopping.

    Expects TemporalDataset instances: each batch item is one trading date with
    tensors shaped (K, seq_len, C) for K tickers.

    Returns (best_model, best_val_loss).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Training on {device}")

    num_tickers = len(config.TICKERS)
    model = VLSTM(
        config.NUM_FEATURES,
        config.HIDDEN_DIM,
        num_tickers,
        config.DROPOUT,
        config.NUM_LSTM_LAYERS,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS, eta_min=config.LEARNING_RATE / 10
    )

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
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
        for X_batch, tid_batch, fwd_batch, _vs_batch in train_loader:
            # X_batch:   (T, K, seq_len, C)   T = temporal batch size
            # tid_batch: (T, K)
            # fwd_batch: (T, K)
            T, K, seq_len, C = X_batch.shape
            X_flat = X_batch.view(T * K, seq_len, C).to(device)
            tid_flat = tid_batch.view(T * K).to(device)
            fwd_flat = fwd_batch.view(T * K).to(device)

            optimizer.zero_grad()
            positions_flat = model(X_flat, tid_flat)        # (T*K,)
            positions = positions_flat.view(T, K)
            fwd_returns = fwd_flat.view(T, K)

            loss = portfolio_sharpe_loss(positions, fwd_returns)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # Validation — accumulate all timesteps then compute portfolio Sharpe
        model.eval()
        all_positions, all_fwd_returns = [], []
        with torch.no_grad():
            for X_batch, tid_batch, fwd_batch, _vs_batch in val_loader:
                T, K, seq_len, C = X_batch.shape
                X_flat = X_batch.view(T * K, seq_len, C).to(device)
                tid_flat = tid_batch.view(T * K).to(device)
                positions = model(X_flat, tid_flat).cpu().view(T, K)
                all_positions.append(positions)
                all_fwd_returns.append(fwd_batch)

        if not all_positions:
            continue

        val_loss = portfolio_sharpe_loss(
            torch.cat(all_positions, dim=0),    # (total_T, K)
            torch.cat(all_fwd_returns, dim=0),  # (total_T, K)
        ).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, checkpoint_path)
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= config.PATIENCE:
                print(f"  Early stop at epoch {epoch + 1}")
                break

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1:3d}  val_loss={val_loss:.4f}  patience={patience_count}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_loss


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from training.dataset import TemporalDataset

    TICKERS_SMALL = ["SPY", "QQQ"]
    N_DAYS = 120

    class _FastConfig:
        TICKERS = TICKERS_SMALL
        NUM_FEATURES = 9
        HIDDEN_DIM = 32
        DROPOUT = 0.0
        NUM_LSTM_LAYERS = 1
        LEARNING_RATE = 1e-3
        WEIGHT_DECAY = 0.01
        BATCH_SIZE = 8          # timesteps per batch
        EPOCHS = 6
        PATIENCE = 3
        TARGET_VOL = 0.1
        SEQUENCE_LENGTH = 20

    cfg = _FastConfig()
    tickers_2_id = {t: i for i, t in enumerate(cfg.TICKERS)}

    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=N_DAYS)
    features_dict = {}
    for ticker in cfg.TICKERS:
        df = pd.DataFrame(
            rng.standard_normal((N_DAYS, cfg.NUM_FEATURES)),
            index=dates,
            columns=[f"f{i}" for i in range(cfg.NUM_FEATURES)],
        )
        df["target"] = rng.standard_normal(N_DAYS)
        df["forward_return"] = rng.standard_normal(N_DAYS) * 0.01
        df["vs_factor"] = rng.standard_normal(N_DAYS) ** 2 + 1.0
        features_dict[ticker] = df

    train_ds = TemporalDataset(features_dict, cfg.SEQUENCE_LENGTH, tickers_2_id)
    val_ds = TemporalDataset(features_dict, cfg.SEQUENCE_LENGTH, tickers_2_id)

    assert len(train_ds) > 0, "Empty train dataset"
    print(f"Train dataset size: {len(train_ds)} timesteps")

    model, val_loss = train_model(train_ds, val_ds, cfg)

    model.eval()
    X_s, tid_s, _, _ = train_ds[0]
    out = model(X_s.unsqueeze(0).view(1, len(cfg.TICKERS), cfg.SEQUENCE_LENGTH, cfg.NUM_FEATURES)
                .view(len(cfg.TICKERS), cfg.SEQUENCE_LENGTH, cfg.NUM_FEATURES),
                tid_s)
    assert out.shape == (len(cfg.TICKERS),), f"Bad output shape: {out.shape}"
    assert all(-1.0 <= v <= 1.0 for v in out.tolist()), "Output outside [-1, 1]"
    print(f"val_loss: {val_loss:.4f}  PASS")
    print(f"Output shape: {out.shape}  PASS")
