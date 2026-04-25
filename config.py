# Future ETFs
TICKERS = [
    # Equity ETFs
    "SPY", # S&P 500 ETF
    "QQQ", # Nasdaq 100 ETF
    "IWM", # Russell 2000 ETF
    "EFA", # MSCI EAFE ETF
    "EEM", # MSCI Emerging Markets ETF
    "VNQ", # Vanguard Real Estate ETF

    # Bond ETFs
    "AGG",   # iShares Core U.S. Aggregate Bond ETF
    "BND",   # Vanguard Total Bond Market ETF

    # Commodities
    "GLD",   # Gold
    "SLV",   # Silver
    "USO",   # Oil
    "UNG",   # Natural gas
]

# Data
START_DATE = "2010-01-01"
END_DATE = "2025-12-31"

# Training
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
EPOCHS = 50
PATIENCE = 20 # Early stopping patience
TARGET_VOL = 0.1 # Target volatility for dynamic position sizing

# Features, more will be added, just basing off of the original paper for now
VOL_SPAN = 20 # Lookback period for volatility calculation, might be changed later
RETURN_HORIZONS = [1, 5, 21, 63, 126, 252] # Return horizons in trading days
MACD_PAIRS = [(8, 24), (16, 48), (32, 96)] # MACD short and long periods

# Model (VLSTM)
# TO be filled out later

# Walk forward, rolling window, and backtesting
TRAIN_YEARS = 5
TEST_YEARS = 1
ROLLING_STEP = 1 # Roll forward by 1 year each iteration
INITIAL_CAPITAL = 1000000 # Starting capital for backtesting

# IBKR
IBKR_HOST = "127.0.0.1"
IBKR_PORT = 7497 # Default TWS port, 7496 fo live accounts
IBKR_CLIENT_ID = 1 # Unique client ID for IBKR connection