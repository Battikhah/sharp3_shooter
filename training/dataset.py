import torch
import pandas as pd

class FuturesDataset(torch.utils.data.Dataset):
    def __init__(self, 
                features_df: dict[str, pd.DataFrame],
                sequence_length: int,
                tickers_2_id: dict[str, int]
                ):
        self.sequence_length = sequence_length
        self.tickers_2_id = tickers_2_id

        # Identify feature columns (everything that isn't metadata)
        temp_df = next(iter(features_df.values()))
        self.feature_cols = [col for col in temp_df.columns if col not in ['target', 'forward_return', 'vs_factor']]

        # Build Index (ticker, end_row_idx) (valid windows)
        self.index = []

        for ticker, df in features_df.items():
            for i in range(len(df) - sequence_length + 1):
                self.index.append((ticker, i + sequence_length - 1))

        # Store ticker arrays for quick access
        self._data = {}

        for ticker, df in features_df.items():
            if ticker in tickers_2_id:
                self._data[ticker] = {
                    'features': df[self.feature_cols].values.astype('float32'),
                    'target': df['target'].values.astype('float32'),
                    'forward_return': df['forward_return'].values.astype('float32'),
                    'vs_factor': df['vs_factor'].values.astype('float32')
                }

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ticker, end_idx = self.index[idx]
        df = self._data[ticker]

        start_row = end_idx - self.sequence_length + 1

        X = torch.from_numpy(df['features'][start_row:end_idx + 1])

        ticker_id = torch.tensor(self.tickers_2_id[ticker], dtype=torch.long)
        fwd_return = torch.tensor(df['forward_return'][end_idx], dtype=torch.float32)
        vs_factor = torch.tensor(df['vs_factor'][end_idx], dtype=torch.float32)
        target = torch.tensor(df['target'][end_idx], dtype=torch.float32)

        return X, target, ticker_id, fwd_return, vs_factor
    
class TemporalDataset(torch.utils.data.Dataset):
    """
    Each item is one trading date; returns tensors for ALL K tickers at that date.

    This enables portfolio-level Sharpe optimisation: the training loss can compute
    the cross-sectional average return across tickers before computing Sharpe, which
    is the paper's "pooled portfolio returns" objective (Eq. 8–10).

    Returns per item:
        X   (K, seq_len, C)   — lookback windows for every ticker
        tid (K,)              — ticker integer IDs
        fwd (K,)              — next-day forward returns
        vs  (K,)              — volatility-scaling factors
    """
    def __init__(
        self,
        features_df: dict[str, "pd.DataFrame"],
        sequence_length: int,
        tickers_2_id: dict[str, int],
    ):
        self.sequence_length = sequence_length
        self.tickers_2_id = tickers_2_id

        sample = next(iter(features_df.values()))
        self.feature_cols = [c for c in sample.columns if c not in ("target", "forward_return", "vs_factor")]

        self._data: dict[str, dict] = {}
        valid_per_ticker: dict[str, set] = {}

        for ticker, df in features_df.items():
            if ticker not in tickers_2_id or len(df) < sequence_length:
                continue
            self._data[ticker] = {
                "features": df[self.feature_cols].values.astype("float32"),
                "forward_return": df["forward_return"].values.astype("float32"),
                "vs_factor": df["vs_factor"].values.astype("float32"),
                "date_to_idx": {d: i for i, d in enumerate(df.index)},
            }
            # Dates that have at least seq_len rows of prior history
            valid_per_ticker[ticker] = set(df.index[sequence_length - 1:])

        # Only keep dates where every ticker has a full lookback window
        if valid_per_ticker:
            common = set.intersection(*valid_per_ticker.values())
            self.valid_dates = sorted(common)
        else:
            self.valid_dates = []

        # Stable ticker ordering (follows TICKERS list order via tickers_2_id insertion order)
        self.tickers = [t for t in tickers_2_id if t in self._data]

    def __len__(self) -> int:
        return len(self.valid_dates)

    def __getitem__(self, idx: int):
        date = self.valid_dates[idx]
        X_list, tid_list, fwd_list, vs_list = [], [], [], []

        for ticker in self.tickers:
            data = self._data[ticker]
            end = data["date_to_idx"][date]
            start = end - self.sequence_length + 1
            X_list.append(torch.from_numpy(data["features"][start:end + 1]))
            tid_list.append(self.tickers_2_id[ticker])
            fwd_list.append(data["forward_return"][end])
            vs_list.append(data["vs_factor"][end])

        return (
            torch.stack(X_list),                               # (K, seq_len, C)
            torch.tensor(tid_list, dtype=torch.long),          # (K,)
            torch.tensor(fwd_list, dtype=torch.float32),       # (K,)
            torch.tensor(vs_list, dtype=torch.float32),        # (K,)
        )


if __name__ == "__main__":

    from config import TICKERS, START_DATE, END_DATE
    from data.fetch import fetch_data
    from data.features import compute_features

    # Fetch raw data
    raw_data = fetch_data(TICKERS, START_DATE, END_DATE)

    # Compute features
    features = compute_features(raw_data)

    # Create ticker to ID mapping
    tickers_2_id = {ticker: idx for idx, ticker in enumerate(TICKERS)}

    # Create Dataset
    dataset = FuturesDataset(features, sequence_length=252, tickers_2_id=tickers_2_id)

    print(f"Dataset Length: {len(dataset)}")
    X, y, ticker_id, fwd_return, vs_factor = dataset[0]
    print(f"X shape: {X.shape}, y: {y}, ticker_id: {ticker_id}, fwd_return: {fwd_return}, vs_factor: {vs_factor}")