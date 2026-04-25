# ETFs
TICKERS = [
    "SPY",  # S&P 500 ETF
    "IVV",  # iShares Core S&P 500 ETF
    "VOO",  # Vanguard S&P 500 ETF
    "QQQ",  # Invesco QQQ Trust (tracks the Nasdaq-100)
    "VTI",  # Vanguard Total Stock Market ETF
    "IWM",  # iShares Russell 2000 ETF
    "EFA",  # iShares MSCI EAFE ETF

    #Halal ETFs
    "SPUS",  # SPDR Portfolio S&P 500 ETF
    "HLAL",  # Wahed FTSE USA Shariah ETF
    "MNZL",  # Manzil Russell Halal USA Broad Market ETF
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