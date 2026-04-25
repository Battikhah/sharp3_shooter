from config import TICKERS, START_DATE, END_DATE
from pathlib import Path

# Chache Directory and File
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "raw_data.parquet"


# Inputs:
# - Ticker List
# - Start and End Data

# Outputs:
# - DataFrame with MultiIndex columns: (ticker, field)

def fetch_data(tickers: list[str], start_date: str, end_date: str):
    import yfinance as yf
    import pandas as pd

    # Check cache first
    if CACHE_FILE.exists():
        data = pd.read_parquet(CACHE_FILE)
    else:
        # Download and cache
        data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
        data.to_parquet(CACHE_FILE)

    return data

if __name__ == "__main__":
    data = fetch_data(
        tickers=TICKERS,
        start_date=START_DATE,
        end_date=END_DATE
    )

    print(data.head())