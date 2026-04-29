# sharp3-shooter

A systematic ETF trading system built on the VLSTM architecture from *Deep Learning for Financial Time Series: A Large-Scale Benchmark of Risk-Adjusted Performance* (Saly-Kaufmann et al., 2026). The model uses a Variable Selection Network + LSTM to output daily position signals in [-1, 1], trained end-to-end on a differentiable Sharpe ratio loss, and executed via IBKR paper trading.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Step-by-Step Usage](#step-by-step-usage)
  - [1. Download Market Data](#1-download-market-data)
  - [2. Run the Full Training Pipeline](#2-run-the-full-training-pipeline)
  - [3. Resume a Partial Run](#3-resume-a-partial-run)
  - [4. Set Up IBKR Paper Trading](#4-set-up-ibkr-paper-trading)
  - [5. Test the IBKR Connection](#5-test-the-ibkr-connection)
  - [6. Run the Daily Pipeline (Dry Run)](#6-run-the-daily-pipeline-dry-run)
  - [7. Go Live](#7-go-live)
  - [8. Monitor Performance](#8-monitor-performance)
- [Project Structure](#project-structure)
- [Output Files](#output-files)
- [Safeguards](#safeguards)
- [Known Limitations](#known-limitations)

---

## How It Works

```
Market Data (yfinance)
       ↓
Feature Engineering          9 features per ticker per day:
  • 6 normalized returns     ret_norm_{1,5,21,63,126,252}
  • 3 MACD signals           macd_signal_{8_24,16_48,32_96}
       ↓
Walk-Forward Training        Annual rolling window:
  • Train on 5 years           train model_2015.pt on 2010–2014
  • Test on next year          generate positions for 2015
  • Slide forward 1 year       repeat through 2025
       ↓
VLSTM Model                  Per-timestep feature selection (VSN)
                             + ticker embedding + LSTM + tanh head
                             → position signal in [-1, 1]
       ↓
Position Sizing              weight = signal × target_vol × vs_factor
                             vs_factor = 1 / σ_t (volatility parity)
       ↓
Backtest / Live Trading      Simulate PnL or submit orders to IBKR
```

The model is trained to directly maximize risk-adjusted returns (Sharpe loss), not to predict prices. A positive signal means go long; negative means go short. The volatility-parity sizing keeps each ticker contributing roughly equal risk regardless of its absolute volatility.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | 3.11 recommended |
| PyTorch | 2.x | CPU works; CUDA auto-detected |
| IBKR TWS or IB Gateway | Any | For live/paper trading steps only |
| IBKR paper account | — | Free to open at interactivebrokers.com |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Battikhah/sharp3-shooter.git
cd sharp3-shooter
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For CUDA support, install the matching PyTorch build **before** the above command:

```bash
# Example for CUDA 12.1 — find your exact command at pytorch.org/get-started
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

The training and inference code auto-detects CUDA at runtime — no config change needed.

### 4. Verify the installation

```bash
python -c "import torch; import ib_insync; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
```

---

## Configuration

All parameters live in `config.py`. Edit this file before running anything.

| Parameter | Default | Description |
|---|---|---|
| `TICKERS` | 12 ETFs | Universe traded (SPY, QQQ, IWM, EFA, EEM, VNQ, AGG, BND, GLD, SLV, USO, UNG) |
| `START_DATE` | `2010-01-01` | Start of historical data download |
| `END_DATE` | `2025-12-31` | End of historical data download |
| `SEQUENCE_LENGTH` | `84` | Trading days per model input window |
| `HIDDEN_DIM` | `256` | LSTM and VSN hidden size |
| `NUM_LSTM_LAYERS` | `2` | LSTM depth |
| `DROPOUT` | `0.3` | Dropout rate (applied between LSTM layers) |
| `BATCH_SIZE` | `128` | Training batch size |
| `LEARNING_RATE` | `1e-4` | Adam optimizer LR |
| `EPOCHS` | `50` | Max epochs per fold |
| `PATIENCE` | `20` | Early stopping patience (validation Sharpe) |
| `TARGET_VOL` | `0.10` | Annualized target volatility for position sizing (10%) |
| `TRAIN_YEARS` | `5` | Rolling training window length |
| `INITIAL_CAPITAL` | `1,000,000` | NAV used in dry-run mode |
| `IBKR_HOST` | `127.0.0.1` | TWS / IB Gateway host |
| `IBKR_PORT` | `7497` | 7497 = TWS paper, 4002 = Gateway paper, 7496 = live |
| `IBKR_CLIENT_ID` | `1` | Must be unique per connected client |

---

## Step-by-Step Usage

### 1. Download Market Data

The data is fetched automatically by the pipeline. To pre-download and inspect:

```bash
python -m data.fetch
```

This saves a parquet cache to `data/cache/raw_data.parquet`. Subsequent runs load from cache. To force a fresh download, delete the cache file:

```bash
rm data/cache/raw_data.parquet
```

To inspect computed features:

```bash
python -m data.features
```

---

### 2. Run the Full Training Pipeline

```bash
python run_pipeline.py
```

This runs 4 steps automatically:

1. **Load data** — reads from cache (or downloads if cache is missing)
2. **Compute features** — 9 features per ticker per day
3. **Walk-forward training** — trains one model per test year, saves checkpoints
4. **Backtest** — simulates portfolio returns, prints metrics, saves plots

Expected output:

```
============================================================
STEP 3 / 4  Walk-forward training  (11 folds: 2015–2025)
============================================================
[2015] train 2010-01-01 → 2014-12-31 | test 2015-01-01 → 2015-12-31
  Training on cuda
  Early stop at epoch 31
[2016] train 2011-01-01 → 2015-12-31 | test 2016-01-01 → 2016-12-31
  ...

Metric                              Value
------------------------------------------
  annualized_return                0.0842
  annualized_vol                   0.0731
  sharpe                           1.1517
  max_drawdown                    -0.0943
  calmar                           0.8928
  hit_rate                         0.5312
  avg_abs_daily_return             0.0046
```

Training time is roughly 2–6 hours on CPU, 30–60 minutes on a single GPU.

---

### 3. Resume a Partial Run

If training was interrupted, skip already-completed folds:

```bash
python run_pipeline.py --resume
```

This detects existing `checkpoints/model_{year}.pt` files and skips those folds.

---

### 4. Set Up IBKR Paper Trading

Before running the live pipeline, configure TWS or IB Gateway:

1. Open **TWS** and log into your paper account
2. Go to **File → Global Configuration → API → Settings**
3. Enable:
   - `Enable ActiveX and Socket Clients`
   - `Allow connections from localhost only`
4. Set **Socket port** to `7497` (TWS paper default)

For **IB Gateway** instead of TWS:
- Use port `4002` for paper trading
- Update `IBKR_PORT = 4002` in `config.py`

---

### 5. Test the IBKR Connection

```bash
python -m live.ibkr_connect
```

Expected output:

```
Connecting to 127.0.0.1:7497  client_id=1 ...
Connected  client_id=1

Account Summary:
  BuyingPower                   1,000,000.00
  GrossPositionValue                    0.00
  NetLiquidation                1,000,000.00
  RealizedPnL                           0.00
  TotalCashValue                1,000,000.00
  UnrealizedPnL                         0.00

Open Positions (0 rows):
  (none — clean paper account)

All assertions  PASS
Disconnected.
```

If you see `Connection refused`, check:
- TWS / IB Gateway is running and logged in
- API access is enabled in Global Configuration
- Port matches `IBKR_PORT` in `config.py`
- `127.0.0.1` is in the trusted IP list (or leave the list blank)

---

### 6. Run the Daily Pipeline (Dry Run)

Test the end-of-day pipeline without submitting any orders:

```bash
python -m live.daily_run --dry-run
```

This will:
- Download the last ~600 calendar days of prices (always fresh, not from training cache)
- Compute features for each ticker
- Load the latest checkpoint from `checkpoints/`
- Generate a position signal per ticker
- Print what orders *would* be submitted, including safeguard outcomes

Example output:

```
2026-04-29 17:00:01 INFO     Fetching prices 2024-10-11 → 2026-04-29
2026-04-29 17:00:04 INFO     Computing features
2026-04-29 17:00:05 INFO     Loading checkpoint: checkpoints/model_2025.pt
2026-04-29 17:00:05 INFO     DRY-RUN: NAV=$1,000,000, no open positions assumed
2026-04-29 17:00:05 INFO     SPY    signal=+0.3142  vs=14.21  weight=+0.4462  target=+893sh  delta=+893sh
2026-04-29 17:00:05 INFO     QQQ    signal=-0.1083  vs=12.84  weight=-0.1388  target=-310sh  delta=-310sh
...
2026-04-29 17:00:06 INFO     submit_orders: 12/12 approved  (NAV=$1,000,000)
2026-04-29 17:00:06 INFO     DRY    SPY    BUY 893 sh  [signal=+0.3142 delta=+893sh $418,274]
2026-04-29 17:00:06 INFO     DRY    QQQ    SELL 310 sh  [signal=-0.1083 delta=-310sh $173,220]
```

---

### 7. Go Live

Once you are comfortable with the dry-run output and have been paper trading for at least 2–3 months:

```bash
python -m live.daily_run
```

Run this after market close each trading day (5:00–5:30 PM ET). Orders are placed as **Market-on-Open** for the next session's open.

To automate with cron (macOS/Linux), add to your crontab (`crontab -e`):

```
0 17 * * 1-5 cd /path/to/sharp3-shooter && .venv/bin/python -m live.daily_run >> results/daily_run.log 2>&1
```

---

### 8. Monitor Performance

**Log today's NAV and PnL** — run once per trading day after market close:

```bash
python -m live.monitor --log
```

Appends a row to `results/live_pnl.csv`:

```
date,nav,daily_pnl
2026-04-01,1000000,0.0
2026-04-02,1003241,3241.0
2026-04-03,1001887,-1354.0
```

**Compare live vs backtest Sharpe** — run weekly:

```bash
python -m live.monitor --compare
```

```
──────────────────────────────────────────────
  Live window : 2026-04-01 → 2026-04-29 (20 days)
  Live Sharpe : +0.8734
  BT Sharpe   : +1.1042  (same window)
──────────────────────────────────────────────
```

If live Sharpe diverges significantly from backtest, investigate before scaling up.

**Plot cumulative returns:**

```bash
python -m live.monitor --plot
```

Saves `results/live_vs_backtest.png`.

**Combine all three flags in one call:**

```bash
python -m live.monitor --log --compare --plot
```

---

## Project Structure

```
sharp3-shooter/
├── config.py                   # All hyperparameters and settings
│
├── data/
│   ├── fetch.py                # Download + cache market data (yfinance)
│   ├── features.py             # Compute normalized returns, MACD, vs_factor
│   └── cache/
│       └── raw_data.parquet    # Auto-generated price cache (gitignored)
│
├── model/
│   ├── vsn.py                  # Variable Selection Network
│   ├── vlstm.py                # Full VLSTM model (VSN + LSTM + tanh head)
│   └── losses.py               # Differentiable Sharpe ratio loss
│
├── training/
│   ├── dataset.py              # FuturesDataset — sliding window PyTorch dataset
│   ├── train.py                # Training loop with early stopping + gradient clipping
│   └── walkforward.py          # Annual rolling-window orchestration
│
├── backtest/
│   ├── simulate.py             # Convert positions → daily portfolio returns
│   └── metrics.py              # Sharpe, drawdown, Calmar, hit rate, PnL plot
│
├── live/
│   ├── ibkr_connect.py         # IBKR connection helpers
│   ├── daily_run.py            # End-of-day pipeline
│   ├── execute.py              # Order submission with safeguards
│   └── monitor.py              # PnL logger and live vs backtest comparison
│
├── tests/
│   └── sanity.py               # Leakage, random baseline, cost sensitivity checks
│
├── checkpoints/                # Trained model weights (gitignored)
│   └── model_{year}.pt         # One file per walk-forward fold
│
├── results/                    # All outputs (gitignored)
│   ├── pnl.png                 # Backtest cumulative PnL chart
│   ├── positions.parquet       # Walk-forward position signals
│   ├── daily_returns.parquet   # Backtest daily portfolio returns
│   ├── live_pnl.csv            # Live trading NAV log
│   ├── signals.csv             # Yesterday's signals (for flip detection)
│   ├── orders.log              # Full order history with safeguard outcomes
│   └── live_vs_backtest.png    # Monitor comparison chart
│
├── run_pipeline.py             # Main entry point: fetch → features → train → backtest
├── requirements.txt
└── documentation.md            # Detailed technical documentation
```

---

## Output Files

| File | Created by | Description |
|---|---|---|
| `data/cache/raw_data.parquet` | `data/fetch.py` | Cached OHLCV prices |
| `checkpoints/model_{year}.pt` | `training/walkforward.py` | Model weights per fold |
| `results/positions.parquet` | `run_pipeline.py` | Walk-forward position signals |
| `results/daily_returns.parquet` | `run_pipeline.py` | Backtest daily returns |
| `results/pnl.png` | `run_pipeline.py` | Backtest PnL chart |
| `results/signals.csv` | `live/daily_run.py` | Latest signals (flip detection) |
| `results/orders.log` | `live/execute.py` | Every order decision with timestamp |
| `results/live_pnl.csv` | `live/monitor.py --log` | Daily NAV and PnL |
| `results/live_vs_backtest.png` | `live/monitor.py --plot` | Comparison chart |

---

## Safeguards

Three risk controls are enforced on every call to `live/execute.py`:

| Safeguard | Threshold | Behavior |
|---|---|---|
| Signal flip | `\|signal_t − signal_{t-1}\| > 1.5` | Order blocked; logged as BLOCK |
| Position cap | `\|target_$\| > 20% of NAV` | Order blocked; logged as BLOCK |
| Daily turnover cap | Cumulative `\|Δ$\| > 200% of NAV` | Remaining orders blocked in sequence |

All decisions (approved and blocked) are appended to `results/orders.log` with a UTC timestamp. Orders smaller than 1 share after rounding are silently skipped.

---

## Known Limitations

- **ETF universe, not futures.** The paper reports a 2.39 Sharpe on 50+ liquid futures. With 12 ETFs and real bid-ask spreads, expect 0.5–1.2 if everything works.
- **Annual retraining.** The model is retrained once per year. Regime changes mid-year are not handled until the next fold.
- **Cache staleness.** `data/fetch.py` loads from cache if the file exists. Delete `data/cache/raw_data.parquet` to refresh historical data. The live pipeline always fetches fresh prices independently.
- **Paper trading only.** Change `IBKR_PORT = 7496` in `config.py` to switch to a live account. Do not do this without months of paper trading validation.
- **No short-selling constraints.** The model can output negative positions on any ticker. Borrow costs and availability are not modeled.
