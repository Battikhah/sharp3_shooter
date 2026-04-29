"""
live/monitor.py — Live trading performance monitoring.

Usage:
    python -m live.monitor --log      # append today's NAV + PnL to results/live_pnl.csv
    python -m live.monitor --compare  # print live vs backtest Sharpe over the live window
    python -m live.monitor --plot     # save results/live_vs_backtest.png
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LIVE_PNL_PATH = Path("results/live_pnl.csv")
BACKTEST_PATH = Path("results/daily_returns.parquet")
PLOT_PATH = Path("results/live_vs_backtest.png")


def _annualized_sharpe(returns: pd.Series) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return float("nan")
    return float(returns.mean() / returns.std() * np.sqrt(252))


def _load_live_returns() -> pd.Series | None:
    """Return daily pct-change of NAV from live_pnl.csv. Returns None on failure."""
    if not LIVE_PNL_PATH.exists():
        logging.error("No live PnL log at %s — run --log first", LIVE_PNL_PATH)
        return None
    df = pd.read_csv(LIVE_PNL_PATH, parse_dates=["date"]).sort_values("date")
    returns = df.set_index("date")["nav"].pct_change().dropna()
    if len(returns) < 2:
        logging.error("Need ≥2 days of live data for analysis (have %d)", len(returns))
        return None
    return returns


def log_today(ib) -> None:
    """Read current NAV from IBKR and append a row to results/live_pnl.csv."""
    from live.ibkr_connect import get_account_summary

    summary = get_account_summary(ib)
    nav = summary["NetLiquidation"]
    today_str = date.today().isoformat()

    LIVE_PNL_PATH.parent.mkdir(exist_ok=True)

    if LIVE_PNL_PATH.exists():
        hist = pd.read_csv(LIVE_PNL_PATH, parse_dates=["date"])
        if today_str in hist["date"].astype(str).values:
            logging.info("Already logged for %s — skipping", today_str)
            return
        prev_nav = float(hist["nav"].iloc[-1])
        daily_pnl = nav - prev_nav
    else:
        hist = pd.DataFrame(columns=["date", "nav", "daily_pnl"])
        daily_pnl = 0.0

    new_row = pd.DataFrame([{"date": today_str, "nav": nav, "daily_pnl": daily_pnl}])
    hist = pd.concat([hist, new_row], ignore_index=True)
    hist.to_csv(LIVE_PNL_PATH, index=False)
    logging.info("Logged %s  NAV=$%,.0f  daily_pnl=$%+,.0f", today_str, nav, daily_pnl)


def compare_sharpe() -> None:
    """Print live Sharpe vs backtest Sharpe over the same date window."""
    live_returns = _load_live_returns()
    if live_returns is None:
        return

    live_sharpe = _annualized_sharpe(live_returns)
    start, end = live_returns.index.min(), live_returns.index.max()
    n_days = len(live_returns)

    print(f"\n{'─' * 46}")
    print(f"  Live window : {start.date()} → {end.date()} ({n_days} days)")
    print(f"  Live Sharpe : {live_sharpe:+.4f}")

    if BACKTEST_PATH.exists():
        bt = pd.read_parquet(BACKTEST_PATH)
        bt.index = pd.to_datetime(bt.index)
        bt_window = bt.loc[start:end, "portfolio_return"]
        if len(bt_window) >= 2:
            bt_sharpe = _annualized_sharpe(bt_window)
            print(f"  BT Sharpe   : {bt_sharpe:+.4f}  (same window)")
        else:
            print(f"  BT Sharpe   : n/a (no backtest data for this window)")
    else:
        print(f"  BT Sharpe   : n/a ({BACKTEST_PATH} missing — run run_pipeline.py first)")

    print(f"{'─' * 46}\n")


def plot_comparison() -> None:
    """Save cumulative return chart: live vs backtest over the live trading window."""
    live_returns = _load_live_returns()
    if live_returns is None:
        return

    start, end = live_returns.index.min(), live_returns.index.max()
    live_cum = (1 + live_returns).cumprod()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(live_cum.index, live_cum.values, label="Live", color="steelblue", linewidth=1.8)

    if BACKTEST_PATH.exists():
        bt = pd.read_parquet(BACKTEST_PATH)
        bt.index = pd.to_datetime(bt.index)
        bt_window = bt.loc[start:end, "portfolio_return"]
        if not bt_window.empty:
            bt_cum = (1 + bt_window).cumprod()
            ax.plot(bt_cum.index, bt_cum.values, label="Backtest",
                    color="coral", linewidth=1.8, linestyle="--")

    ax.axhline(1.0, color="black", linewidth=0.6, linestyle=":")
    ax.set_title("Live vs Backtest — Cumulative Return")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    fig.tight_layout()

    PLOT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    logging.info("Plot saved → %s", PLOT_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

    parser = argparse.ArgumentParser(description="Live trading monitor")
    parser.add_argument("--log", action="store_true", help="Log today's NAV and PnL")
    parser.add_argument("--compare", action="store_true", help="Compare live vs backtest Sharpe")
    parser.add_argument("--plot", action="store_true", help="Save live_vs_backtest.png")
    args = parser.parse_args()

    if not (args.log or args.compare or args.plot):
        parser.print_help()
    else:
        if args.log:
            from live.ibkr_connect import connect
            ib = connect(readonly=True)
            log_today(ib)
            ib.disconnect()
        if args.compare:
            compare_sharpe()
        if args.plot:
            plot_comparison()
