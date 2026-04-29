"""
Full training + backtest pipeline.

Usage:
  python run_pipeline.py            # train everything from scratch
  python run_pipeline.py --resume   # skip folds with existing checkpoints
"""
import argparse
import time
from pathlib import Path

import config
from data.fetch import fetch_data
from data.features import compute_features
from training.walkforward import run_walkforward
from backtest.simulate import simulate
from backtest.metrics import compute_metrics, plot_pnl

START_YEAR = int(config.START_DATE[:4]) + config.TRAIN_YEARS   # 2015
END_YEAR   = int(config.END_DATE[:4])                           # 2025
N_FOLDS    = END_YEAR - START_YEAR + 1                          # 11


def main(resume: bool) -> None:
    t0 = time.time()

    # ── 1. Load data ────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1 / 4  Load market data")
    print("=" * 60)
    raw = fetch_data(config.TICKERS, config.START_DATE, config.END_DATE)
    n_tickers = raw.columns.get_level_values(0).nunique()
    print(f"  {len(raw):,} trading days × {n_tickers} tickers  (from cache)\n")

    # ── 2. Compute features ─────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 2 / 4  Compute features")
    print("=" * 60)
    features = compute_features(raw)
    for ticker, df in features.items():
        print(f"  {ticker:<6}  {len(df):>5} rows  {len(df.columns)} cols")
    print()

    # ── 3. Walk-forward training ─────────────────────────────────────────────
    print("=" * 60)
    print(f"STEP 3 / 4  Walk-forward training  ({N_FOLDS} folds: {START_YEAR}–{END_YEAR})")
    if resume:
        from pathlib import Path as _P
        done = sorted(
            int(p.stem.split("_")[1])
            for p in _P("checkpoints").glob("model_*.pt")
            if p.stem.split("_")[1].isdigit()
        )
        remaining = N_FOLDS - len(done)
        print(f"  Resume mode — {len(done)} checkpoints found, {remaining} folds to train")
    print("=" * 60)

    positions = run_walkforward(features, config, resume=resume)
    print(f"\nWalk-forward complete  {len(positions):,} position rows\n")

    # ── 4. Backtest ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 4 / 4  Backtest")
    print("=" * 60)
    daily_returns = simulate(positions, target_vol=config.TARGET_VOL)
    metrics = compute_metrics(daily_returns["portfolio_return"])

    print(f"\n{'Metric':<30} {'Value':>10}")
    print("-" * 42)
    for k, v in metrics.items():
        print(f"  {k:<28} {v:>10.4f}")

    Path("results").mkdir(exist_ok=True)
    plot_pnl(daily_returns["portfolio_return"], out_path="results/pnl.png")
    positions.to_parquet("results/positions.parquet")
    daily_returns.to_parquet("results/daily_returns.parquet")

    elapsed = time.time() - t0
    print(f"\nFinished in {elapsed / 3600:.2f}h")
    print("  Checkpoints  →  checkpoints/model_{{year}}.pt")
    print("  PnL chart    →  results/pnl.png")
    print("  Positions    →  results/positions.parquet")
    print("  Daily PnL    →  results/daily_returns.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip folds where checkpoints/model_{year}.pt already exists",
    )
    args = parser.parse_args()
    main(resume=args.resume)
