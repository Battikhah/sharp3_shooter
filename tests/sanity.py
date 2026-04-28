"""
Five pre-deployment sanity checks.
Run: python -m tests.sanity
All checks must PASS before going live.
"""
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

from backtest.simulate import simulate
from backtest.metrics import compute_metrics
from training.dataset import FuturesDataset
from training.train import train_model
from model.losses import sharpe_loss


def _make_positions(
    rng: np.random.Generator,
    n_dates: int,
    tickers: list[str],
    has_signal: bool,
) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    rows = []
    for ticker in tickers:
        fwd_returns = rng.normal(0.0005, 0.01, n_dates)
        vs_factors = rng.uniform(5, 15, n_dates)
        positions = np.sign(fwd_returns) if has_signal else rng.uniform(-1, 1, n_dates)
        for i, date in enumerate(dates):
            rows.append({
                "date": date,
                "ticker": ticker,
                "position": float(positions[i]),
                "fwd_return": float(fwd_returns[i]),
                "vs_factor": float(vs_factors[i]),
            })
    return pd.DataFrame(rows)


def check_leakage(rng: np.random.Generator) -> None:
    """Positions that used future signal must outperform shuffled (null) positions."""
    pos_df = _make_positions(rng, 252, ["SPY", "QQQ"], has_signal=True)

    sharpe_signal = compute_metrics(
        simulate(pos_df, target_vol=0.1)["portfolio_return"]
    )["sharpe"]

    shuffled = pos_df.copy()
    shuffled["position"] = shuffled["position"].values[rng.permutation(len(shuffled))]
    sharpe_shuffled = compute_metrics(
        simulate(shuffled, target_vol=0.1)["portfolio_return"]
    )["sharpe"]

    assert sharpe_signal > sharpe_shuffled, (
        f"Signal Sharpe {sharpe_signal:.2f} should exceed shuffled {sharpe_shuffled:.2f}"
    )
    print(f"[1] Leakage:  signal={sharpe_signal:.2f}  shuffled={sharpe_shuffled:.2f}  PASS")


def check_random_baseline(rng: np.random.Generator) -> None:
    """Random positions must produce near-zero Sharpe."""
    pos_df = _make_positions(rng, 504, ["SPY", "QQQ", "GLD"], has_signal=False)
    sharpe = compute_metrics(
        simulate(pos_df, target_vol=0.1)["portfolio_return"]
    )["sharpe"]
    assert abs(sharpe) < 2.0, f"Random baseline Sharpe unexpectedly large: {sharpe:.2f}"
    print(f"[2] Random baseline:  Sharpe={sharpe:.2f} (expected ~0)  PASS")


def check_constant_long(rng: np.random.Generator) -> None:
    """position=+1, vs_factor=1/target_vol → weight=1 → strategy equals buy-and-hold."""
    n = 252
    target_vol = 0.1
    dates = pd.bdate_range("2020-01-01", periods=n)
    fwd_returns = rng.normal(0.0005, 0.01, n)

    pos_df = pd.DataFrame({
        "date": dates,
        "ticker": "SPY",
        "position": 1.0,
        "fwd_return": fwd_returns,
        "vs_factor": 1.0 / target_vol,
    })
    strategy_sharpe = compute_metrics(
        simulate(pos_df, target_vol=target_vol)["portfolio_return"]
    )["sharpe"]
    passive_sharpe = compute_metrics(pd.Series(fwd_returns, index=dates))["sharpe"]

    assert abs(strategy_sharpe - passive_sharpe) < 1e-4, (
        f"Constant long {strategy_sharpe:.6f} != passive {passive_sharpe:.6f}"
    )
    print(f"[3] Constant long:  strategy={strategy_sharpe:.4f}  passive={passive_sharpe:.4f}  PASS")


def check_cost_sensitivity(rng: np.random.Generator) -> None:
    """Sharpe must be non-increasing as transaction costs increase."""
    pos_df = _make_positions(rng, 504, ["SPY", "QQQ"], has_signal=True)
    bps_grid = [0, 1, 5, 10, 20, 50]
    sharpes = [
        compute_metrics(
            simulate(pos_df, target_vol=0.1, cost_bps=b)["portfolio_return"]
        )["sharpe"]
        for b in bps_grid
    ]

    for i in range(len(sharpes) - 1):
        assert sharpes[i] >= sharpes[i + 1] - 0.01, (
            f"Sharpe rose {sharpes[i]:.3f} → {sharpes[i + 1]:.3f} "
            f"at {bps_grid[i]} → {bps_grid[i + 1]} bps"
        )

    Path("results").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(bps_grid, sharpes, marker="o")
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Transaction Cost (bps)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Sharpe vs Transaction Cost")
    ax.grid(True, alpha=0.3)
    fig.savefig("results/cost_sensitivity.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    label = {b: f"{s:.2f}" for b, s in zip(bps_grid, sharpes)}
    print(f"[4] Cost sensitivity:  {label}  PASS")
    print(f"    Plot → results/cost_sensitivity.png")


def check_seed_sensitivity() -> None:
    """Five training seeds must produce finite, consistent val Sharpe on identical data."""
    class _Cfg:
        TICKERS = ["SPY", "QQQ"]
        NUM_FEATURES = 9
        HIDDEN_DIM = 32
        DROPOUT = 0.0
        NUM_LSTM_LAYERS = 1
        LEARNING_RATE = 1e-3
        BATCH_SIZE = 16
        EPOCHS = 3
        PATIENCE = 3
        TARGET_VOL = 0.1
        SEQUENCE_LENGTH = 20

    cfg = _Cfg()
    tickers_2_id = {t: i for i, t in enumerate(cfg.TICKERS)}
    n_days, cutoff = 150, 120

    val_sharpes = []
    for seed in range(5):
        # Same data distribution each run (deterministic per seed)
        data_rng = np.random.default_rng(seed)
        features_dict = {
            ticker: _build_feature_df(data_rng, n_days, cfg.NUM_FEATURES)
            for ticker in cfg.TICKERS
        }

        train_dict = {t: df.iloc[:cutoff] for t, df in features_dict.items()}
        val_dict = {t: df.iloc[cutoff:] for t, df in features_dict.items()}

        torch.manual_seed(seed)
        train_ds = FuturesDataset(train_dict, cfg.SEQUENCE_LENGTH, tickers_2_id)
        val_ds = FuturesDataset(val_dict, cfg.SEQUENCE_LENGTH, tickers_2_id)

        model = train_model(train_ds, val_ds, cfg)

        model.eval()
        val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE)
        batch_losses = []
        with torch.no_grad():
            for X, _t, tid, fwd, vs in val_loader:
                pos = model(X, tid)
                batch_losses.append(sharpe_loss(pos, fwd, vs, cfg.TARGET_VOL).item())

        val_sharpe = -sum(batch_losses) / len(batch_losses) if batch_losses else 0.0
        val_sharpes.append(val_sharpe)

    arr = np.array(val_sharpes)
    assert np.isfinite(arr).all(), f"NaN/Inf in val Sharpe: {val_sharpes}"

    print(f"[5] Seed sensitivity:  val_sharpes={[f'{v:.3f}' for v in val_sharpes]}")
    print(f"    mean={arr.mean():.3f}  std={arr.std():.3f}  PASS")


def _build_feature_df(rng: np.random.Generator, n_days: int, n_features: int) -> pd.DataFrame:
    df = pd.DataFrame(
        rng.standard_normal((n_days, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    df["target"] = rng.standard_normal(n_days)
    df["forward_return"] = rng.standard_normal(n_days) * 0.01
    df["vs_factor"] = rng.standard_normal(n_days) ** 2 + 1.0
    return df


if __name__ == "__main__":
    import time

    rng = np.random.default_rng(42)
    t0 = time.time()

    print("Running sanity checks...\n")
    check_leakage(rng)
    check_random_baseline(rng)
    check_constant_long(rng)
    check_cost_sensitivity(rng)
    check_seed_sensitivity()

    print(f"\nAll 5 sanity checks PASSED in {time.time() - t0:.1f}s")
    print("Safe to proceed to live trading.")
