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
        # Returns a tuple of (X, y, ticker_id, fwd_return)
        ticker, end_idx = self.index[idx]
        df = self._data[ticker]

        steart_row = end_idx - self.sequence_length + 1

        X = df['features'][steart_row:end_idx + 1] 

        ticker_id = torch.tensor(self.tickers_2_id[ticker], dtype=torch.long)
        fwd_return = torch.tensor(df['forward_return'][end_idx], dtype=torch.float32)
        target = torch.tensor(df['target'][end_idx], dtype=torch.float32)

        return X, target, ticker_id, fwd_return