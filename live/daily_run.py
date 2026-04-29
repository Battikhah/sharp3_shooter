"""
live/daily_run.py — End-of-day trading pipeline.

Usage:
    python -m live.daily_run            # live: connects to IBKR and submits orders
    python -m live.daily_run --dry-run  # print orders without submitting
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import torch
import yfinance as yf

import config
from data.features import compute_features
from model.vlstm import VLSTM
from training.walkforward import _feature_cols
from live.ibkr_connect import connect, get_account_summary, get_positions
from live.execute import Order, submit_orders

SIGNALS_PATH = Path("results/signals.csv")
CHECKPOINT_DIR = Path("checkpoints")
LOOKBACK_DAYS = 600  # calendar days — gives ~500 trading days, enough for 252-day MACD burn-in


def _load_latest_checkpoint() -> torch.nn.Module:
    checkpoints = sorted(CHECKPOINT_DIR.glob("model_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No model_*.pt checkpoints found in {CHECKPOINT_DIR}")
    latest = checkpoints[-1]
    logging.info("Loading checkpoint: %s", latest)
    model = VLSTM(
        config.NUM_FEATURES,
        config.HIDDEN_DIM,
        len(config.TICKERS),
        config.DROPOUT,
        config.NUM_LSTM_LAYERS,
    )
    model.load_state_dict(torch.load(latest, weights_only=True))
    model.eval()
    return model


def _fetch_recent_prices() -> pd.DataFrame:
    """Download the last LOOKBACK_DAYS of prices directly (bypasses the stale training cache)."""
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=LOOKBACK_DAYS)).isoformat()
    logging.info("Fetching prices %s → %s", start, end)
    return yf.download(config.TICKERS, start=start, end=end, group_by="ticker", progress=False)


def _load_prev_signals() -> dict[str, float]:
    if not SIGNALS_PATH.exists():
        return {}
    df = pd.read_csv(SIGNALS_PATH, index_col="ticker")
    return df["signal"].to_dict()


def _save_signals(signals: dict[str, float]) -> None:
    SIGNALS_PATH.parent.mkdir(exist_ok=True)
    pd.DataFrame(
        [{"ticker": t, "signal": s} for t, s in signals.items()]
    ).set_index("ticker").to_csv(SIGNALS_PATH)


def run_daily(dry_run: bool = False) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

    # 1. Fetch recent prices and compute features
    prices = _fetch_recent_prices()
    logging.info("Computing features")
    features_df = compute_features(prices)

    if not features_df:
        logging.error("compute_features returned empty dict — aborting")
        return

    # 2. Load model
    model = _load_latest_checkpoint()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    tickers_to_id = {t: i for i, t in enumerate(config.TICKERS)}
    feature_cols = _feature_cols(features_df)
    seq_len = config.SEQUENCE_LENGTH
    prev_signals = _load_prev_signals()

    # 3. IBKR account state (or placeholders for dry-run)
    if dry_run:
        logging.info("DRY-RUN: NAV=$%,.0f, no open positions assumed", config.INITIAL_CAPITAL)
        nav = float(config.INITIAL_CAPITAL)
        current_positions: dict[str, float] = {}
        ib = None
    else:
        ib = connect(readonly=False)
        summary = get_account_summary(ib)
        nav = summary["NetLiquidation"]
        pos_df = get_positions(ib)
        current_positions = dict(zip(pos_df["ticker"], pos_df["position"]))
        logging.info("Connected  NAV=$%,.0f  %d open positions", nav, len(current_positions))

    # 4. Last close prices
    last_close: dict[str, float] = {}
    for ticker in config.TICKERS:
        try:
            last_close[ticker] = float(prices[ticker]["Close"].iloc[-1])
        except (KeyError, IndexError):
            logging.warning("%s: close price unavailable", ticker)

    # 5. Generate signals and build orders
    orders: list[Order] = []
    new_signals: dict[str, float] = {}

    with torch.no_grad():
        for ticker, df in features_df.items():
            if ticker not in tickers_to_id:
                continue
            if len(df) < seq_len:
                logging.warning("%s: %d rows < seq_len=%d — skipping", ticker, len(df), seq_len)
                continue

            window = torch.from_numpy(df[feature_cols].iloc[-seq_len:].values.astype("float32"))
            tid = torch.tensor(tickers_to_id[ticker], dtype=torch.long)
            signal = float(
                model(window.unsqueeze(0).to(device), tid.unsqueeze(0).to(device)).cpu().item()
            )
            new_signals[ticker] = signal

            price = last_close.get(ticker)
            if not price or price <= 0:
                logging.warning("%s: no valid price — skipping", ticker)
                continue

            vs_factor = float(df["vs_factor"].iloc[-1])
            target_weight = signal * config.TARGET_VOL * vs_factor
            target_shares = (target_weight * nav) / price
            current_shares = current_positions.get(ticker, 0.0)
            delta_shares = target_shares - current_shares

            logging.info(
                "%-6s signal=%+.4f  vs=%.2f  weight=%+.4f  target=%+.0fsh  delta=%+.0fsh",
                ticker, signal, vs_factor, target_weight, target_shares, delta_shares,
            )

            orders.append(Order(
                ticker=ticker,
                delta_shares=delta_shares,
                current_shares=current_shares,
                signal=signal,
                prev_signal=prev_signals.get(ticker),
                price=price,
            ))

    # 6. Save signals for tomorrow's flip detection
    _save_signals(new_signals)

    # 7. Submit (or dry-print) orders
    submit_orders(ib, orders, nav, dry_run=dry_run)

    if ib is not None:
        ib.disconnect()
        logging.info("Disconnected from IBKR")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-of-day trading pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Print orders without submitting")
    args = parser.parse_args()
    run_daily(dry_run=args.dry_run)
