# VLSTM Trading System — Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Configuration (`config.py`)](#configuration)
5. [Data Layer](#data-layer)
   - [Fetcher (`data/fetch.py`)](#fetcher)
   - [Feature Engineering (`data/features.py`)](#feature-engineering)
6. [Model Layer](#model-layer)
   - [Variable Selection Network (`model/vsn.py`)](#variable-selection-network)
   - [VLSTM (`model/vlstm.py`)](#vlstm)
   - [Sharpe Loss (`model/losses.py`)](#sharpe-loss)
7. [Training Layer](#training-layer)
   - [Dataset (`training/dataset.py`)](#dataset)
   - [Training Loop (`training/train.py`)](#training-loop)
   - [Walk-Forward Orchestrator (`training/walkforward.py`)](#walk-forward-orchestrator)
8. [Backtest Layer](#backtest-layer)
   - [Simulator (`backtest/simulate.py`)](#simulator)
   - [Metrics (`backtest/metrics.py`)](#metrics)
9. [Sanity Checks (`tests/sanity.py`)](#sanity-checks)
10. [Live Trading Layer](#live-trading-layer)
    - [IBKR Connection (`live/ibkr_connect.py`)](#ibkr-connection)
    - [Daily Run (`live/daily_run.py`)](#daily-run)
    - [Order Execution (`live/execute.py`)](#order-execution)
    - [Monitoring (`live/monitor.py`)](#monitoring)
11. [Pipeline Entry Point (`run_pipeline.py`)](#pipeline-entry-point)
12. [Feature Reference](#feature-reference)
13. [Training Methodology](#training-methodology)
14. [Backtesting Methodology](#backtesting-methodology)
15. [CUDA / Device Support](#cuda--device-support)
16. [Setup & Installation](#setup--installation)
17. [Running the System](#running-the-system)
18. [Known Limitations](#known-limitations)

---

## Overview

This system implements the **VLSTM** model from *Deep Learning for Financial Time Series: A Large-Scale Benchmark of Risk-Adjusted Performance* (Saly-Kaufmann et al., 2026). VLSTM (Variable Selection Network + LSTM) is the highest-performing model in that benchmark, reporting an annualized Sharpe ratio of 2.39 on 50+ futures instruments under ideal execution.

This implementation targets **ETF proxies** (SPY, QQQ, IWM, etc.) via Yahoo Finance, with live paper trading through Interactive Brokers (IBKR). Expect a Sharpe of 0.5–1.2 in practice — the paper's 2.39 reflects gross returns on 50+ liquid futures with no execution costs.

The system is built end-to-end:
- Historical data is downloaded and cached from Yahoo Finance
- Features are computed per the paper's Appendix A specification
- A VLSTM is trained on a 5-year rolling window and tested on the next year (walk-forward)
- Backtest converts model positions into portfolio returns
- The live layer connects to IBKR, generates daily positions, and submits orders

---

## Architecture

```
Yahoo Finance
      │
      ▼
data/fetch.py  ──► data/cache/raw_data.parquet
      │
      ▼
data/features.py  ──► dict[ticker → DataFrame(features)]
      │
      ▼
training/dataset.py  ──► FuturesDataset (PyTorch)
      │
      ▼
training/train.py  ──► VLSTM (trained model)
      │
      ▼
training/walkforward.py  ──► positions DataFrame [date, ticker, position, fwd_return, vs_factor]
      │
      ▼
backtest/simulate.py  ──► daily_returns DataFrame
      │
      ▼
backtest/metrics.py  ──► Sharpe, Calmar, drawdown + PnL plot
      │
      ▼
live/daily_run.py  ──► target weights per ticker
      │
      ▼
live/execute.py  ──► IBKR orders (via ibkr_connect.py)
```

---

## Project Structure

```
sharp3-shooter/
├── config.py                  # All hyperparameters and settings
├── run_pipeline.py            # Full train + backtest pipeline entry point
├── documentation.md           # This file
├── requirements.txt           # Pinned dependencies
│
├── data/
│   ├── fetch.py               # Download OHLCV from Yahoo Finance + local cache
│   ├── features.py            # Compute normalized returns, MACD, vol-scaling factor
│   └── cache/
│       └── raw_data.parquet   # Cached OHLCV (auto-created, gitignored)
│
├── model/
│   ├── vsn.py                 # Variable Selection Network
│   ├── vlstm.py               # Full VLSTM model (VSN + LSTM + head)
│   └── losses.py              # Differentiable negative Sharpe loss
│
├── training/
│   ├── dataset.py             # PyTorch Dataset wrapping the feature dict
│   ├── train.py               # Single train/val training loop with early stopping
│   └── walkforward.py         # Annual rolling-window orchestration
│
├── backtest/
│   ├── simulate.py            # Convert positions to portfolio returns
│   └── metrics.py             # Sharpe, drawdown, Calmar, hit rate, PnL plot
│
├── tests/
│   └── sanity.py              # Five pre-deployment sanity checks
│
├── live/
│   ├── ibkr_connect.py        # IBKR connection + account/position query helpers
│   ├── daily_run.py           # End-of-day pipeline: fetch → features → positions → orders
│   ├── execute.py             # Order submission with safeguards
│   └── monitor.py             # Daily PnL logger + live vs backtest comparison
│
├── checkpoints/               # Saved model weights per walk-forward fold
│   └── model_{year}.pt        # e.g., model_2020.pt trained on 2015–2019
│
└── results/
    ├── pnl.png                # Cumulative PnL chart
    ├── positions.parquet      # Walk-forward positions output
    └── daily_returns.parquet  # Daily portfolio returns
```

---

## Configuration

**File:** `config.py`

Single source of truth for all parameters. Every module imports from here.

### Universe

```python
TICKERS = [
    "SPY",  # S&P 500 ETF
    "QQQ",  # Nasdaq 100 ETF
    "IWM",  # Russell 2000 ETF
    "EFA",  # MSCI EAFE ETF
    "EEM",  # MSCI Emerging Markets ETF
    "VNQ",  # Vanguard Real Estate ETF
    "AGG",  # U.S. Aggregate Bond ETF
    "BND",  # Total Bond Market ETF
    "GLD",  # Gold
    "SLV",  # Silver
    "USO",  # Oil
    "UNG",  # Natural Gas
]
```

### Data Window

| Parameter    | Value        | Description                        |
|--------------|--------------|------------------------------------|
| `START_DATE` | `2010-01-01` | First date fetched from Yahoo      |
| `END_DATE`   | `2025-12-31` | Last date fetched from Yahoo       |

### Feature Parameters

| Parameter         | Value               | Description                                   |
|-------------------|---------------------|-----------------------------------------------|
| `VOL_SPAN`        | `60`                | EWMA span (trading days) for volatility       |
| `RETURN_HORIZONS` | `[1,5,21,63,126,252]` | Horizons (days) for normalized returns      |
| `MACD_PAIRS`      | `[(8,24),(16,48),(32,96)]` | (fast, slow) EWMA periods for MACD   |
| `NUM_FEATURES`    | `9`                 | `len(RETURN_HORIZONS) + len(MACD_PAIRS)`     |

### Model Parameters

| Parameter        | Value | Description                                  |
|------------------|-------|----------------------------------------------|
| `SEQUENCE_LENGTH`| `84`  | Input window size in trading days (~4 months)|
| `HIDDEN_DIM`     | `256` | Hidden dimension for VSN, LSTM, embeddings   |
| `NUM_LSTM_LAYERS`| `2`   | Number of stacked LSTM layers                |
| `DROPOUT`        | `0.3` | Dropout rate (applied between LSTM layers)   |

### Training Parameters

| Parameter       | Value  | Description                              |
|-----------------|--------|------------------------------------------|
| `BATCH_SIZE`    | `128`  | Mini-batch size                          |
| `LEARNING_RATE` | `1e-4` | Adam learning rate                       |
| `EPOCHS`        | `50`   | Maximum training epochs per fold         |
| `PATIENCE`      | `20`   | Early-stopping patience (val loss)       |
| `TARGET_VOL`    | `0.10` | Annualized volatility target for sizing  |

### Walk-Forward Parameters

| Parameter        | Value  | Description                               |
|------------------|--------|-------------------------------------------|
| `TRAIN_YEARS`    | `5`    | Training window length in years           |
| `TEST_YEARS`     | `1`    | Test window length in years               |
| `ROLLING_STEP`   | `1`    | Slide forward by this many years per fold |
| `INITIAL_CAPITAL`| `1,000,000` | Starting NAV for position sizing    |

### IBKR Parameters

| Parameter        | Value       | Description                                    |
|------------------|-------------|------------------------------------------------|
| `IBKR_HOST`      | `127.0.0.1` | TWS / IB Gateway hostname                      |
| `IBKR_PORT`      | `7497`      | `7497` = TWS paper, `7496` = TWS live, `4002` = Gateway paper |
| `IBKR_CLIENT_ID` | `1`         | Client ID for the IB API connection            |

---

## Data Layer

### Fetcher

**File:** `data/fetch.py`

```python
def fetch_data(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame
```

Downloads daily OHLCV data for all tickers using `yfinance` and caches the result locally.

**Behavior:**
- If `data/cache/raw_data.parquet` exists, loads from cache without hitting the network.
- Otherwise downloads via `yfinance.download(group_by='ticker')` and saves to cache.
- Returns a `DataFrame` with a two-level column index: `(ticker, field)` where `field` is one of `Open`, `High`, `Low`, `Close`, `Volume`.

**Cache invalidation:** The cache is never automatically invalidated. Delete `data/cache/raw_data.parquet` to force a fresh download.

**Output shape:** approximately `(3,900 trading days × 12 tickers × 5 fields)`.

**Self-test:**
```bash
python -m data.fetch
```

---

### Feature Engineering

**File:** `data/features.py`

```python
def compute_features(prices: pd.DataFrame) -> dict[str, pd.DataFrame]
```

Converts raw OHLCV data into the feature set described in Appendix A of the paper.

**Returns:** A dict mapping each ticker symbol to a `DataFrame` with the following columns:

| Column              | Formula                                          | Description                          |
|---------------------|--------------------------------------------------|--------------------------------------|
| `return_norm_1`     | `r_{t-1:t} / (σ_t · √1)`                        | 1-day normalized return              |
| `return_norm_5`     | `r_{t-5:t} / (σ_t · √5)`                        | 5-day normalized return              |
| `return_norm_21`    | `r_{t-21:t} / (σ_t · √21)`                      | 1-month normalized return            |
| `return_norm_63`    | `r_{t-63:t} / (σ_t · √63)`                      | 3-month normalized return            |
| `return_norm_126`   | `r_{t-126:t} / (σ_t · √126)`                    | 6-month normalized return            |
| `return_norm_252`   | `r_{t-252:t} / (σ_t · √252)`                    | 1-year normalized return             |
| `macd_signal_8_24`  | `q / Std_{252}(q)`, where `q = MACD / Std_{63}(P)` | Fast MACD signal                  |
| `macd_signal_16_48` | same formula with spans 16/48                    | Medium MACD signal                   |
| `macd_signal_32_96` | same formula with spans 32/96                    | Slow MACD signal                     |
| `vs_factor`         | `1 / σ_t`                                        | Volatility-scaling factor (metadata) |
| `target`            | `clip(r_{t+1} / σ_t, -20, 20)`                  | Training label (next-day vol-scaled return) |
| `forward_return`    | `r_{t+1}`                                        | Raw next-day return (backtest only)  |

**Look-ahead safety:** All features at time `t` use only data up to and including `t`. The `target` and `forward_return` columns use `t+1` and are only used as labels/backtest inputs — never as model input features.

**Burn-in:** The first ~252 rows per ticker are dropped (`dropna()`), removing the period before all rolling windows are populated.

**Self-test:**
```bash
python -m data.features
```

---

## Model Layer

### Variable Selection Network

**File:** `model/vsn.py`

Implements the VSN component from the paper. It learns to weight input features dynamically per timestep, allowing the model to focus on the most relevant signals at each point in time.

**Architecture:**
1. **Per-feature embeddings** — each scalar feature is projected independently through a `Linear(1 → hidden_dim)` layer, yielding `num_features` vectors of shape `hidden_dim`.
2. **Gating network** — a two-layer MLP takes all feature embeddings concatenated (`num_features × hidden_dim`) and outputs a `softmax` distribution over the `num_features` features.
3. **Weighted sum** — gating weights are applied to the embeddings, collapsing the feature dimension to a single `hidden_dim` vector per timestep.

**Signature:**
```python
class VariableSelectionNetwork(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int)
    def forward(self, x: Tensor) -> Tensor
    # x:      (batch, seq_len, num_features)
    # output: (batch, seq_len, hidden_dim)
```

**Self-test:**
```bash
python -m model.vsn
```
Checks output shape, gating weights sum to 1, and gradients flow to all parameters.

---

### VLSTM

**File:** `model/vlstm.py`

Full end-to-end model: VSN → ticker embedding injection → LSTM → linear head → tanh.

**Architecture:**

```
x (batch, seq_len, num_features)
        │
        ▼
  VariableSelectionNetwork
        │
        ▼  (batch, seq_len, hidden_dim)
        + ticker_embedding(ticker_id).unsqueeze(1)   ← broadcast across seq_len
        │
        ▼  (batch, seq_len, hidden_dim)
  LSTM (num_layers=2, batch_first=True, dropout=0.3)
        │
        ▼  take last timestep: (batch, hidden_dim)
  Linear(hidden_dim → 1)
        │
        ▼
  tanh  →  (batch,)  in [-1, 1]
```

The ticker embedding allows the model to learn asset-specific behavior while sharing parameters across the entire universe.

**Signature:**
```python
class VLSTM(nn.Module):
    def __init__(self, num_features, hidden_dim, num_tickers, dropout, num_lstm_layers)
    def forward(self, x: Tensor, ticker_id: Tensor) -> Tensor
    # x:         (batch, seq_len, num_features)
    # ticker_id: (batch,)  — integer index into TICKERS list
    # output:    (batch,)  in [-1, 1]
```

**Parameter count:** ~3–4M parameters at `hidden_dim=256`, `num_layers=2`, `num_tickers=12`.

**Self-test:**
```bash
python -m model.vlstm
```
Checks output shape, range `[-1, 1]`, gradient flow, and parameter count.

---

### Sharpe Loss

**File:** `model/losses.py`

Differentiable negative annualized Sharpe ratio, computed on the entire batch.

```python
def sharpe_loss(
    positions: Tensor,    # (batch,) model output in [-1, 1]
    fwd_returns: Tensor,  # (batch,) next-day asset returns
    vs_factors: Tensor,   # (batch,) 1/σ_t
    target_vol: float = 0.10,
    eps: float = 1e-6,
) -> Tensor  # scalar
```

**Math:**
```
weight         = position × target_vol × vs_factor
portfolio_ret  = weight × fwd_return
sharpe         = mean(portfolio_ret) / std(portfolio_ret) × √252
loss           = −sharpe
```

**Why this loss:** Directly maximizing Sharpe makes the model learn to produce positions that, in aggregate across the batch, yield good risk-adjusted returns. It is end-to-end differentiable with respect to the model's position outputs.

**Self-test:**
```bash
python -m model.losses
```
Checks scalar output, `requires_grad`, backward pass, and a known-Sharpe verification.

---

## Training Layer

### Dataset

**File:** `training/dataset.py`

```python
class FuturesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        features_df: dict[str, pd.DataFrame],
        sequence_length: int,
        tickers_2_id: dict[str, int],
    )
```

Wraps the per-ticker feature dicts into a PyTorch `Dataset`.

**Indexing:** Builds a flat list of `(ticker, end_row_idx)` pairs — one entry for every valid window of length `sequence_length` across all tickers.

**`__getitem__` returns:**
```
X          — (sequence_length, num_features) float32 tensor
target     — scalar float32 (vol-scaled next-day return, clipped ±20)
ticker_id  — scalar int64 (index into TICKERS)
fwd_return — scalar float32 (raw next-day return, for Sharpe loss)
vs_factor  — scalar float32 (1/σ_t, for Sharpe loss)
```

Note that `target` is returned but not used during training — the Sharpe loss uses `fwd_return` and `vs_factor` instead.

**Length:** approximately `num_tickers × (num_trading_days − sequence_length)`.

**Self-test:**
```bash
python -m training.dataset
```

---

### Training Loop

**File:** `training/train.py`

```python
def train_model(
    train_dataset: FuturesDataset,
    val_dataset: FuturesDataset,
    config,
) -> nn.Module
```

Trains a single VLSTM instance on a fixed train/val split and returns the best checkpoint.

**Algorithm:**
1. Instantiate a fresh `VLSTM` and move it to the detected device (CUDA or CPU).
2. Create `DataLoader`s (training with `shuffle=True`, validation without).
3. For each epoch:
   - Training pass: compute Sharpe loss, backward, clip gradients at norm 1.0, step Adam.
   - Validation pass: accumulate Sharpe loss with `torch.no_grad()`.
   - Track best validation loss; if improved, save `best_state` (as CPU tensors).
   - If validation does not improve for `PATIENCE` epochs, stop early.
4. Load `best_state` back into the model and return it.

**Checkpoint:** An intermediate checkpoint `checkpoints/{timestamp}.pt` is saved during training so the best state is recoverable even if the process is killed.

**Device:** Prints `Training on cuda` or `Training on cpu` at the start of each call.

**Self-test:**
```bash
python -m training.train
```

---

### Walk-Forward Orchestrator

**File:** `training/walkforward.py`

```python
def run_walkforward(
    features_df: dict[str, pd.DataFrame],
    config,
    resume: bool = False,
) -> pd.DataFrame
```

Implements the annual rolling-window evaluation scheme from the paper.

**Scheme:** For each test year `Y` from `START_DATE + TRAIN_YEARS` to `END_DATE`:
1. **Train window:** `Y − TRAIN_YEARS` to `Y − 1` (5 years).
2. **Val split:** last 10% of training window by date.
3. **Train** a fresh VLSTM on the training portion; save `checkpoints/model_{Y}.pt`.
4. **Inference** on test year `Y`: slide a `SEQUENCE_LENGTH`-day window through each ticker's test data and collect predicted positions.

**Resume mode (`--resume`):** If `checkpoints/model_{Y}.pt` already exists for a fold, load it and skip training. Useful for restarting interrupted runs.

**Look-ahead guard:** An assertion verifies that the training data's last date is strictly before the test year's start date. This will raise immediately if leakage is introduced.

**Output DataFrame columns:**

| Column         | Type    | Description                                   |
|----------------|---------|-----------------------------------------------|
| `date`         | date    | Trading day                                   |
| `ticker`       | str     | Ticker symbol                                 |
| `position`     | float   | Model output in `[-1, 1]`                     |
| `fwd_return`   | float   | Actual next-day return (from features)        |
| `vs_factor`    | float   | `1/σ_t` on that date                          |

**Self-test:**
```bash
python training/walkforward.py
```

---

## Backtest Layer

### Simulator

**File:** `backtest/simulate.py`

```python
def simulate(
    positions_df: pd.DataFrame,
    target_vol: float = 0.10,
    cost_bps: float = 0,
) -> pd.DataFrame
```

Converts the walk-forward positions into daily portfolio returns.

**Math per row:**
```
weight         = position × target_vol × vs_factor
ticker_return  = weight × fwd_return
```

**Portfolio return per day:** mean of `ticker_return` across all tickers on that date.

**Transaction costs:** If `cost_bps > 0`, the absolute change in weight for each ticker (vs. the previous day) is multiplied by `cost_bps / 10,000` and subtracted from `ticker_return`.

**Output:** `DataFrame` with a single column `portfolio_return`, indexed by `date`.

**Self-test:**
```bash
python -m backtest.simulate
```
Tests output shape, cost monotonicity, and a known-return case (`position=1, vs_factor=1/target_vol` → `weight=1` → `ticker_return = fwd_return`).

---

### Metrics

**File:** `backtest/metrics.py`

```python
def compute_metrics(daily_returns: pd.Series) -> dict
def plot_pnl(daily_returns: pd.Series, out_path: str = "results/pnl.png") -> None
```

**`compute_metrics` returns:**

| Key                    | Formula                                            |
|------------------------|----------------------------------------------------|
| `annualized_return`    | `mean(r) × 252`                                    |
| `annualized_vol`       | `std(r) × √252`                                    |
| `sharpe`               | `annualized_return / annualized_vol`               |
| `max_drawdown`         | Min of `(cumulative − rolling_max) / rolling_max`  |
| `calmar`               | `annualized_return / |max_drawdown|`               |
| `hit_rate`             | Fraction of days with positive return              |
| `avg_abs_daily_return` | Mean of `|r|` (proxy for turnover)                 |

**`plot_pnl`:** Saves a `(12 × 5)` inch cumulative PnL chart (growth of $1) to `out_path`. Uses the `Agg` matplotlib backend (no display required).

**Self-test:**
```bash
python -m backtest.metrics
```

---

## Sanity Checks

**File:** `tests/sanity.py`

Five pre-deployment checks that must all pass before going live. Run with:
```bash
python -m tests.sanity
```

| # | Check | What it verifies |
|---|-------|-----------------|
| 1 | **Leakage** | Positions derived from future signal (`sign(fwd_return)`) must have higher Sharpe than shuffled positions. Fails if features accidentally encode future data. |
| 2 | **Random baseline** | Purely random positions must produce `|Sharpe| < 2.0`. Guards against data or implementation errors that would make even noise look profitable. |
| 3 | **Constant long** | `position=+1, vs_factor=1/target_vol` must exactly match a passive buy-and-hold Sharpe. Validates the `simulate()` math. |
| 4 | **Cost sensitivity** | Sharpe must be non-increasing as `cost_bps` increases from 0 → 50. Saves `results/cost_sensitivity.png`. Reveals the strategy's breakeven cost. |
| 5 | **Seed sensitivity** | Five training seeds on identical data must all produce finite validation Sharpe. Fragile strategies will produce `NaN` or huge variance here. |

---

## Live Trading Layer

### IBKR Connection

**File:** `live/ibkr_connect.py`

Helpers for connecting to TWS / IB Gateway and querying account state.

```python
def connect(host, port, client_id, timeout=10, readonly=True) -> IB
def get_account_summary(ib: IB) -> dict[str, float]
def get_positions(ib: IB) -> pd.DataFrame
```

**Pre-requisites before running any live scripts:**
1. TWS or IB Gateway is open and logged into a paper account.
2. API access enabled: `File → Global Configuration → API → Settings → Enable ActiveX and Socket Clients`.
3. Port: `7497` for TWS paper trading, `4002` for IB Gateway paper trading.

**`get_account_summary` returns:** A dict with keys `NetLiquidation`, `TotalCashValue`, `BuyingPower`, `UnrealizedPnL`, `RealizedPnL`, `GrossPositionValue`.

**`get_positions` returns:** `DataFrame` with columns `ticker, sec_type, currency, position, avg_cost`.

**Self-test (requires TWS running):**
```bash
python -m live.ibkr_connect
```

---

### Daily Run

**File:** `live/daily_run.py`

End-of-day pipeline that generates target positions and submits orders to IBKR. Intended to run after market close (e.g., 5 PM ET).

**Workflow:**
1. Fetch latest prices (appends today's data to the feature pipeline).
2. Recompute features for each ticker.
3. Load the most recent model checkpoint from `checkpoints/`.
4. For each ticker, build the `SEQUENCE_LENGTH`-day input window ending today.
5. Run the model → position signal in `[-1, 1]`.
6. Compute target weight: `w = signal × target_vol / σ_t`.
7. Convert weight to dollar amount: `dollars = w × account_NAV`.
8. Convert dollars to shares: `shares = dollars / current_price`.
9. Diff against current IBKR positions to get the required order quantity.
10. Submit orders via `live/execute.py`.

**Dry-run mode:** Pass `--dry-run` to print orders without submitting them to IBKR.

**Usage:**
```bash
python -m live.daily_run              # submit orders
python -m live.daily_run --dry-run    # print only
```

---

### Order Execution

**File:** `live/execute.py`

Robust order submission with mandatory safeguards.

```python
def submit_orders(
    ib: IB,
    orders: list[dict],    # [{ticker, target_shares, current_shares, price}]
    nav: float,
    dry_run: bool = True,
) -> list[dict]            # order receipts with status
```

**Safeguards (all checked before any order is submitted):**

| Safeguard | Limit | Description |
|-----------|-------|-------------|
| Max position per ticker | 20% of NAV | Prevents concentration in a single asset |
| Max daily turnover | 200% of NAV | Prevents runaway churning |
| Signal flip detection | Δposition > 1.5 | Blocks if a position flipped from near +1 to near −1 in a single day with no news |
| Logging | Always | Every order is written to `results/orders.log` with timestamp, ticker, size, dollar value, and reason |

**Order type:** Market-on-Open (MOO) orders submitted for the next session's open. This matches the paper's assumption of next-day execution.

**Self-test (dry-run, no IBKR required):**
```bash
python -m live.execute
```

---

### Monitoring

**File:** `live/monitor.py`

Daily logger and live-vs-backtest comparison tool.

**Daily log (`results/live_pnl.csv`):**

| Column        | Description                          |
|---------------|--------------------------------------|
| `date`        | Trading date                         |
| `nav`         | Account net liquidation value        |
| `daily_pnl`   | Day's PnL in dollars                 |
| `daily_return`| Day's PnL as a fraction of prior NAV |
| `positions`   | JSON snapshot of held positions      |

**Weekly comparison:** Computes the rolling Sharpe over the live period and compares it to the backtest Sharpe over the same date range. A large divergence (>0.5 Sharpe points) is flagged as a warning.

**Usage:**
```bash
python -m live.monitor --log      # append today's PnL to live_pnl.csv
python -m live.monitor --compare  # print live vs backtest Sharpe comparison
python -m live.monitor --plot     # save results/live_vs_backtest.png
```

---

## Pipeline Entry Point

**File:** `run_pipeline.py`

Runs the full historical train + backtest pipeline in one command.

```bash
python run_pipeline.py            # train all folds from scratch
python run_pipeline.py --resume   # skip folds with existing checkpoints
```

**Steps executed:**
1. `fetch_data` — load from cache or download.
2. `compute_features` — build per-ticker feature DataFrames.
3. `run_walkforward` — train 11 models (2015–2025), generate positions.
4. `simulate` → `compute_metrics` → `plot_pnl` — backtest and report.

**Outputs:**
- `checkpoints/model_{year}.pt` — one per fold.
- `results/pnl.png` — cumulative PnL chart.
- `results/positions.parquet` — all walk-forward positions.
- `results/daily_returns.parquet` — daily portfolio returns.

---

## Feature Reference

All 9 model input features (columns fed to the VLSTM):

| Index | Name                  | Paper reference        | Intuition                                    |
|-------|-----------------------|------------------------|----------------------------------------------|
| 0     | `return_norm_1`       | Appendix A, Eq. 1     | Very short-term momentum / mean-reversion    |
| 1     | `return_norm_5`       | Appendix A, Eq. 1     | Weekly momentum                              |
| 2     | `return_norm_21`      | Appendix A, Eq. 1     | Monthly momentum                             |
| 3     | `return_norm_63`      | Appendix A, Eq. 1     | Quarterly momentum                           |
| 4     | `return_norm_126`     | Appendix A, Eq. 1     | Semi-annual momentum                         |
| 5     | `return_norm_252`     | Appendix A, Eq. 1     | Annual momentum                              |
| 6     | `macd_signal_8_24`    | Appendix A.4          | Fast trend-following signal                  |
| 7     | `macd_signal_16_48`   | Appendix A.4          | Medium trend-following signal                |
| 8     | `macd_signal_32_96`   | Appendix A.4          | Slow trend-following signal                  |

`vs_factor` and `forward_return` are metadata passed through the pipeline for sizing and loss computation — they are never fed as model inputs.

---

## Training Methodology

### Walk-Forward Validation

The system uses a strict **expanding/rolling walk-forward** scheme to prevent any future information from contaminating training:

```
Fold 2015:  train 2010–2014  │  test 2015
Fold 2016:  train 2011–2015  │  test 2016
...
Fold 2025:  train 2020–2024  │  test 2025
```

Each fold trains a completely fresh model from random initialization. No warm-starting from the previous year's weights (a valid extension for v2).

### Train / Validation Split

Within each training window, the last 10% of dates form a held-out validation set. The split is done **by date** (not randomly) so the validation set always represents a more recent period than the training set.

### Early Stopping

Training stops if the validation Sharpe loss does not improve for `PATIENCE = 20` consecutive epochs. The model state at the epoch with the lowest validation loss is restored and returned.

### Gradient Clipping

Gradients are clipped to a maximum L2 norm of 1.0 before each optimizer step to prevent exploding gradients in the LSTM.

---

## Backtesting Methodology

### Position → Weight Conversion

```
weight = position × target_vol × vs_factor
       = position × target_vol × (1 / σ_t)
```

This scales each position so that, if held at full size (`position = 1`), the expected annual volatility of that single-asset bet equals `target_vol = 10%`. This is **volatility-parity sizing** — assets with lower realized volatility receive larger dollar positions.

### Portfolio Return

The portfolio return on each day is the **equal-weighted mean** of all ticker returns:
```
portfolio_return_t = mean_over_tickers(weight_t × fwd_return_t)
```

### Transaction Costs

The simulator supports transaction cost modeling via `cost_bps`. The cost is proportional to the absolute change in portfolio weight for each ticker between consecutive days. At `cost_bps = 10` (1 basis point), a position that goes from weight 0 to weight 1 incurs a 0.1% cost.

---

## CUDA / Device Support

The system automatically detects and uses CUDA if available, falling back to CPU otherwise.

**Where device selection happens:**
- `training/train.py` — detects device at the start of `train_model`, moves model and all batch tensors to device. Prints `Training on cuda` or `Training on cpu`.
- `training/walkforward.py` — detects device once at the start of `run_walkforward`, moves model to device after both training and checkpoint loading.

**Checkpoint portability:** Best-state tensors are always saved to CPU (`v.cpu().clone()`), so checkpoints saved on a GPU machine can be loaded on a CPU machine without `map_location` workarounds.

**No code changes required** to switch between CPU and GPU — device selection is fully automatic.

---

## Setup & Installation

### 1. Clone and create virtual environment

```bash
git clone <repo-url>
cd sharp3-shooter
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify core imports

```bash
python -c "import torch; import ib_insync; import yfinance; print('ok')"
```

### 4. (Optional) IBKR setup for live trading

1. Download and install TWS or IB Gateway from Interactive Brokers.
2. Log into a **paper trading** account.
3. Enable API: `File → Global Configuration → API → Settings → Enable ActiveX and Socket Clients`.
4. Confirm port `7497` (TWS paper) or `4002` (IB Gateway paper) is open.

---

## Running the System

### Full historical backtest

```bash
python run_pipeline.py
```

### Resume an interrupted backtest

```bash
python run_pipeline.py --resume
```

### Run individual module self-tests

```bash
python -m model.losses
python -m model.vsn
python -m model.vlstm
python -m backtest.simulate
python -m backtest.metrics
python training/walkforward.py
python -m tests.sanity
```

### Pre-deployment sanity checks

```bash
python -m tests.sanity
```
All 5 checks must print `PASS` before going live.

### Live paper trading (daily, after market close)

```bash
python -m live.daily_run --dry-run   # verify orders without submitting
python -m live.daily_run             # submit orders to IBKR
```

### Monitoring

```bash
python -m live.monitor --log       # record today's PnL
python -m live.monitor --compare   # live vs backtest Sharpe
python -m live.monitor --plot      # generate comparison chart
```

---

## Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| ETFs instead of futures | Lower Sharpe (ETFs have lower volatility, higher tracking error vs paper's instruments) | Acceptable for prototyping; swap for IBKR futures contracts in v2 |
| Annual retraining only | Model may lag regime changes mid-year | Quarterly retraining is a planned v2 feature |
| Single model per fold | High variance in Sharpe across seeds | Seed ensembling (average 5 models) is a planned v2 feature |
| Yahoo Finance data | May have survivorship bias, split errors, or stale prices | Cross-validate critical periods against Bloomberg or Refinitiv |
| No warm-starting | Each fold trains from scratch; 5 × 50 epochs × 11 folds is slow without GPU | Warm-start from previous year's checkpoint as a v2 optimization |
| Market-on-open orders | Assumes fills at the open price; actual fills may differ | Acceptable for paper trading; revisit for live with limit orders |
| No position limits in backtest | Simulator allows arbitrarily large weights | The live `execute.py` enforces a 20% NAV cap; the backtest does not |
| Paper's Sharpe of 2.39 | Reflects 50+ futures, gross of costs, ideal execution | Realistic expectation with ETFs + costs: **0.5–1.2** |
