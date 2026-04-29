# ETF universe — 14 tickers across decorrelated asset classes
# Removed BND (≈0.99 corr with AGG); added TLT (duration), UUP (USD), DBC (commodities)
TICKERS = [
    # US Equity
    "SPY",  # S&P 500
    "QQQ",  # Nasdaq 100
    "IWM",  # Russell 2000 small-cap
    "EFA",  # MSCI EAFE (developed ex-US)
    "EEM",  # MSCI Emerging Markets
    "VNQ",  # US Real Estate
    # Fixed Income
    "AGG",  # US Aggregate Bond
    "TLT",  # 20+ Year Treasury (long duration)
    # Commodities
    "GLD",  # Gold
    "SLV",  # Silver
    "USO",  # Crude Oil
    "UNG",  # Natural Gas
    "DBC",  # Broad Commodity basket
    # FX / Macro
    "UUP",  # US Dollar Index
]

# Data
START_DATE = "2010-01-01"
END_DATE = "2025-12-31"

# Training
# BATCH_SIZE = number of TIMESTEPS per batch in TemporalDataset (each timestep has K tickers)
BATCH_SIZE = 64
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01          # AdamW weight decay
EPOCHS = 100
PATIENCE = 50
TARGET_VOL = 0.10            # Annualised target portfolio volatility (paper default)

# Seed ensembling — train NUM_SEEDS models per fold, keep TOP_SEEDS by val Sharpe
NUM_SEEDS = 5
TOP_SEEDS = 3

# Validation split fraction (applied to the training window)
VAL_FRAC = 0.20              # 80 % train / 20 % val

# Features
VOL_SPAN = 60                # EWMA span for volatility (Appendix A)
RETURN_HORIZONS = [1, 5, 21, 63, 126, 252]
MACD_PAIRS = [(8, 24), (16, 48), (32, 96)]

# Model (VLSTM)
SEQUENCE_LENGTH = 84
HIDDEN_DIM = 256             # paper uses 256 (was 128)
NUM_LSTM_LAYERS = 2
DROPOUT = 0.2
NUM_FEATURES = len(RETURN_HORIZONS) + len(MACD_PAIRS)  # vs_factor is metadata, not a model input

# Walk-forward
TRAIN_YEARS = 5
TEST_YEARS = 1
ROLLING_STEP = 1
INITIAL_CAPITAL = 1_000_000

# IBKR
IBKR_HOST = "127.0.0.1"
IBKR_PORT = 7497             # 7497 = TWS paper; 4002 = Gateway paper; 7496 = live
IBKR_CLIENT_ID = 1