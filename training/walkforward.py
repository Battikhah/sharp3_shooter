import torch
import pandas as pd
from pathlib import Path

from training.dataset import FuturesDataset
from training.train import train_model


def _feature_cols(features_df: dict[str, pd.DataFrame]) -> list[str]:
    sample = next(iter(features_df.values()))
    return [c for c in sample.columns if c not in ("target", "forward_return", "vs_factor")]


def _predict_ticker(
    model: torch.nn.Module,
    df: pd.DataFrame,
    feature_cols: list[str],
    ticker_id: int,
    seq_len: int,
    batch_size: int,
) -> list[dict]:
    feat = torch.from_numpy(df[feature_cols].values.astype("float32"))
    n_windows = len(df) - seq_len + 1
    if n_windows <= 0:
        return []

    tid = torch.tensor(ticker_id, dtype=torch.long)
    rows = []

    for batch_start in range(0, n_windows, batch_size):
        batch_end = min(batch_start + batch_size, n_windows)
        windows = torch.stack([feat[i : i + seq_len] for i in range(batch_start, batch_end)])
        tids = tid.expand(batch_end - batch_start)

        with torch.no_grad():
            positions = model(windows, tids)

        for j, pos in enumerate(positions.tolist()):
            end_idx = batch_start + j + seq_len - 1
            rows.append({
                "date": df.index[end_idx],
                "position": pos,
                "fwd_return": float(df["forward_return"].iloc[end_idx]),
                "vs_factor": float(df["vs_factor"].iloc[end_idx]),
            })

    return rows


def run_walkforward(features_df: dict[str, pd.DataFrame], config) -> pd.DataFrame:
    """
    Annual rolling 5-year-train / 1-year-test walk-forward.
    Returns DataFrame with columns [date, ticker, position, fwd_return, vs_factor].
    """
    tickers_2_id = {t: i for i, t in enumerate(config.TICKERS)}
    feat_cols = _feature_cols(features_df)

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    start_year = int(config.START_DATE[:4]) + config.TRAIN_YEARS
    end_year = int(config.END_DATE[:4])

    all_results = []

    for test_year in range(start_year, end_year + 1):
        train_start = f"{test_year - config.TRAIN_YEARS}-01-01"
        train_end = f"{test_year - 1}-12-31"
        test_start = f"{test_year}-01-01"
        test_end = f"{test_year}-12-31"

        print(f"[{test_year}] train {train_start} → {train_end} | test {test_start} → {test_end}")

        train_feat = {t: df.loc[train_start:train_end] for t, df in features_df.items()}
        test_feat = {t: df.loc[test_start:test_end] for t, df in features_df.items()}

        # Verify no look-ahead leakage
        for t, df_train in train_feat.items():
            df_test = test_feat.get(t, pd.DataFrame())
            if len(df_train) and len(df_test):
                assert df_train.index.max() < pd.Timestamp(test_start), (
                    f"Leakage detected: training data for {t} bleeds into test year {test_year}"
                )

        # 90 / 10 train / val split by date
        ref = next(iter(train_feat.values()))
        cutoff_date = ref.index[int(len(ref) * 0.9)]
        sub_train = {t: df[df.index < cutoff_date] for t, df in train_feat.items()}
        sub_val = {t: df[df.index >= cutoff_date] for t, df in train_feat.items()}

        train_ds = FuturesDataset(sub_train, config.SEQUENCE_LENGTH, tickers_2_id)
        val_ds = FuturesDataset(sub_val, config.SEQUENCE_LENGTH, tickers_2_id)

        model = train_model(train_ds, val_ds, config)
        torch.save(model.state_dict(), checkpoint_dir / f"model_{test_year}.pt")

        model.eval()
        for ticker, df in test_feat.items():
            if ticker not in tickers_2_id or len(df) < config.SEQUENCE_LENGTH:
                continue
            rows = _predict_ticker(
                model, df, feat_cols,
                tickers_2_id[ticker],
                config.SEQUENCE_LENGTH,
                config.BATCH_SIZE,
            )
            for row in rows:
                row["ticker"] = ticker
            all_results.extend(rows)

    result = pd.DataFrame(all_results, columns=["date", "ticker", "position", "fwd_return", "vs_factor"])
    return result.sort_values(["date", "ticker"]).reset_index(drop=True)


if __name__ == "__main__":
    import numpy as np

    TICKERS_SMALL = ["SPY", "QQQ"]

    class _FastConfig:
        TICKERS = TICKERS_SMALL
        NUM_FEATURES = 9
        HIDDEN_DIM = 32
        DROPOUT = 0.0
        NUM_LSTM_LAYERS = 1
        LEARNING_RATE = 1e-3
        BATCH_SIZE = 16
        EPOCHS = 2
        PATIENCE = 2
        TARGET_VOL = 0.1
        SEQUENCE_LENGTH = 20
        START_DATE = "2019-01-01"
        END_DATE = "2021-12-31"
        TRAIN_YEARS = 1

    cfg = _FastConfig()

    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2019-01-01", "2021-12-31")
    features_dict = {}
    for ticker in cfg.TICKERS:
        df = pd.DataFrame(
            rng.standard_normal((len(dates), cfg.NUM_FEATURES)),
            index=dates,
            columns=[f"f{i}" for i in range(cfg.NUM_FEATURES)],
        )
        df["target"] = rng.standard_normal(len(dates))
        df["forward_return"] = rng.standard_normal(len(dates)) * 0.01
        df["vs_factor"] = rng.standard_normal(len(dates)) ** 2 + 1.0
        features_dict[ticker] = df

    out = run_walkforward(features_dict, cfg)

    assert isinstance(out, pd.DataFrame), "Output must be a DataFrame"
    expected_cols = {"date", "ticker", "position", "fwd_return", "vs_factor"}
    assert set(out.columns) == expected_cols, f"Wrong columns: {out.columns.tolist()}"
    assert len(out) > 0, "No output rows"
    assert out["position"].between(-1, 1).all(), "Positions outside [-1, 1]"
    assert out["ticker"].isin(cfg.TICKERS).all(), "Unknown ticker in output"

    saved = list(Path("checkpoints").glob("model_*.pt"))
    assert len(saved) >= 2, f"Expected 2 checkpoint files, got {len(saved)}"

    print(f"Output: {len(out)} rows, {out['date'].nunique()} dates, {out['ticker'].nunique()} tickers  PASS")
    print(f"Columns: {out.columns.tolist()}  PASS")
    print(f"Position range: [{out['position'].min():.4f}, {out['position'].max():.4f}]  PASS")
    print(f"Checkpoints saved: {[p.name for p in saved]}  PASS")
