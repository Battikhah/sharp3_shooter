import pandas as pd
from config import RETURN_HORIZONS, MACD_PAIRS, VOL_SPAN

def compute_features(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    results = {}
    
    for ticker in prices.columns.get_level_values(0).unique():
        p = prices[ticker]['Close'].dropna() # Closing Prices
        r = p.pct_change().dropna() # Daily Returns
        ewma_vol = r.ewm(span=VOL_SPAN).std()

        features = pd.DataFrame(index=r.index)

        # Normalized Returns
        for h in RETURN_HORIZONS:
            features[f'return_norm_{h}'] = r.rolling(window=h).sum() / (ewma_vol * (h ** 0.5))

        # MACD Signals
        for short, long in MACD_PAIRS:
            macd = p.ewm(span=short).mean() - p.ewm(span=long).mean()
            q = macd / p.rolling(window=63).std()
            features[f'macd_signal_{short}_{long}'] = q / q.rolling(window=252).std()

        # Volatility-Scaling Factor
        features['vs_factor'] = 1 / ewma_vol

        # Look Into the Future
        features['target'] = (r.shift(-1) / ewma_vol).clip(-20, 20) # Acts as the Label for training
        features['forward_return'] = r.shift(-1) # Used for backtesting, PnL calculation

        results[ticker] = features.dropna()

    return results

if __name__ == "__main__":
    from config import TICKERS, START_DATE, END_DATE
    from data.fetch import fetch_data

    # Fetch raw data
    raw_data = fetch_data(TICKERS, START_DATE, END_DATE)

    # Compute features
    features = compute_features(raw_data)

    print(features.items())