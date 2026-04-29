import torch
import pandas as pd
from pathlib import Path
import pickle

from sklearn.preprocessing import RobustScaler

from training.dataset import TemporalDataset
from training.train import train_model


def _feature_cols(features_df: dict[str, pd.DataFrame]) -> list[str]:
    sample = next(iter(features_df.values()))
    return [c for c in sample.columns if c not in ("target", "forward_return", "vs_factor")]


def _fit_fold_scaler(train_feat: dict[str, pd.DataFrame], feature_cols: list[str]) -> RobustScaler:
    scaler = RobustScaler()
    train_matrix = pd.concat([df[feature_cols] for df in train_feat.values() if len(df)], axis=0)
    scaler.fit(train_matrix.values)
    return scaler


def _apply_fold_scaler(
    fold_feat: dict[str, pd.DataFrame],
    scaler: RobustScaler,
    feature_cols: list[str],
) -> dict[str, pd.DataFrame]:
    scaled_feat: dict[str, pd.DataFrame] = {}
    for ticker, df in fold_feat.items():
        if len(df) == 0:
            scaled_feat[ticker] = df.copy()
            continue
        scaled_values = scaler.transform(df[feature_cols].values)
        scaled_df = df.copy()
        scaled_df.loc[:, feature_cols] = scaled_values
        scaled_feat[ticker] = scaled_df
    return scaled_feat


def _predict_ticker_ensemble(
    models: list[torch.nn.Module],
    df: pd.DataFrame,
    feature_cols: list[str],
    ticker_id: int,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    test_start: pd.Timestamp,
) -> list[dict]:
    """
    Generate position signals for one ticker by averaging across an ensemble of models.
    Returns raw signals (model tanh output) — scaling is handled by simulate.py.
    """
    feat = torch.from_numpy(df[feature_cols].values.astype("float32"))
    n_windows = len(df) - seq_len + 1
    if n_windows <= 0:
        return []

    tid = torch.tensor(ticker_id, dtype=torch.long)
    rows = []

    for batch_start in range(0, n_windows, batch_size):
        batch_end = min(batch_start + batch_size, n_windows)
        windows = torch.stack([feat[i:i + seq_len] for i in range(batch_start, batch_end)]).to(device)
        tids = tid.expand(batch_end - batch_start).to(device)

        with torch.no_grad():
            pos_sum = sum(model(windows, tids).cpu() for model in models)
            positions = pos_sum / len(models)

        for j, pos in enumerate(positions.tolist()):
            end_idx = batch_start + j + seq_len - 1
            date = df.index[end_idx]
            if date < test_start:
                continue
            rows.append({
                "date": date,
                # Raw signal in (-1, 1) — simulate.py applies target_vol × vs_factor scaling
                "position": float(pos),
                "fwd_return": float(df["forward_return"].iloc[end_idx]),
                "vs_factor": float(df["vs_factor"].iloc[end_idx]),
            })

    return rows


def run_walkforward(
    features_df: dict[str, pd.DataFrame],
    config,
    resume: bool = False,
) -> pd.DataFrame:
    """
    Annual rolling 5-year-train / 1-year-test walk-forward.

    For each fold trains NUM_SEEDS independent models (different random seeds),
    keeps the TOP_SEEDS with the best validation Sharpe, and averages their
    position signals at inference time (seed ensembling, paper Section 3.6).

    Returns DataFrame with columns [date, ticker, position, fwd_return, vs_factor].
    The 'position' column holds the raw model signal in (-1, 1); downstream
    backtest and live code apply target_vol × vs_factor scaling (paper Eq. 6).
    """
    from model.vlstm import VLSTM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        checkpoint_path = checkpoint_dir / f"model_{test_year}.pt"
        scaler_path = checkpoint_dir / f"scaler_{test_year}.pkl"

        print(f"[{test_year}] train {train_start} → {train_end} | test {test_start} → {test_end}")

        train_feat = {t: df.loc[train_start:train_end] for t, df in features_df.items()}

        warmup_start = pd.Timestamp(test_start) - pd.tseries.offsets.BDay(config.SEQUENCE_LENGTH)
        test_feat = {t: df.loc[warmup_start:test_end] for t, df in features_df.items()}

        # 80 / 20 train / val split by date (wider val window = more stable Sharpe signal)
        ref = next(iter(train_feat.values()))
        cutoff_date = ref.index[int(len(ref) * (1.0 - config.VAL_FRAC))]
        sub_train = {t: df[df.index < cutoff_date] for t, df in train_feat.items()}
        sub_val = {t: df[df.index >= cutoff_date] for t, df in train_feat.items()}

        scaler = _fit_fold_scaler(sub_train, feat_cols)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        sub_train = _apply_fold_scaler(sub_train, scaler, feat_cols)
        sub_val = _apply_fold_scaler(sub_val, scaler, feat_cols)
        test_feat_scaled = _apply_fold_scaler(test_feat, scaler, feat_cols)

        if resume and checkpoint_path.exists():
            print(f"  Checkpoint found — loading, skipping training")
            model = VLSTM(
                config.NUM_FEATURES,
                config.HIDDEN_DIM,
                len(tickers_2_id),
                config.DROPOUT,
                config.NUM_LSTM_LAYERS,
            )
            model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
            top_models = [model.to(device)]
        else:
            for t, df_train in train_feat.items():
                df_test = test_feat.get(t, pd.DataFrame())
                if len(df_train) and len(df_test):
                    assert df_train.index.max() < pd.Timestamp(test_start), (
                        f"Leakage detected: training data for {t} bleeds into test year {test_year}"
                    )

            train_ds = TemporalDataset(sub_train, config.SEQUENCE_LENGTH, tickers_2_id)
            val_ds = TemporalDataset(sub_val, config.SEQUENCE_LENGTH, tickers_2_id)

            seed_results: list[tuple[float, torch.nn.Module]] = []

            for seed in range(config.NUM_SEEDS):
                torch.manual_seed(seed)
                model, val_loss = train_model(train_ds, val_ds, config)
                model = model.to(device)
                seed_ckpt = checkpoint_dir / f"model_{test_year}_seed{seed}.pt"
                torch.save(model.state_dict(), seed_ckpt)
                seed_results.append((val_loss, model))
                print(f"  Seed {seed}  val_loss={val_loss:.4f}")

            # Select top TOP_SEEDS models (lower Sharpe loss = higher Sharpe)
            seed_results.sort(key=lambda x: x[0])
            top_models = [m for _, m in seed_results[: config.TOP_SEEDS]]

            # Save the best single-seed weights as the canonical checkpoint for live trading
            torch.save(top_models[0].state_dict(), checkpoint_path)
            print(f"  Ensemble: top {config.TOP_SEEDS} of {config.NUM_SEEDS} seeds")

        for m in top_models:
            m.eval()

        for ticker, df in test_feat_scaled.items():
            if ticker not in tickers_2_id or len(df) < config.SEQUENCE_LENGTH:
                continue
            rows = _predict_ticker_ensemble(
                top_models, df, feat_cols,
                tickers_2_id[ticker],
                config.SEQUENCE_LENGTH,
                config.BATCH_SIZE,
                device,
                pd.Timestamp(test_start),
            )
            for row in rows:
                row["ticker"] = ticker
            all_results.extend(rows)

    result = pd.DataFrame(all_results, columns=["date", "ticker", "position", "fwd_return", "vs_factor"])
    result = result.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Clamp raw signals to [-1, 1] — tanh already guarantees this but guards
    # against any floating-point edge cases from ensemble averaging
    result["position"] = result["position"].clip(-1.0, 1.0)

    return result


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
        WEIGHT_DECAY = 0.01
        BATCH_SIZE = 8
        EPOCHS = 2
        PATIENCE = 2
        TARGET_VOL = 0.1
        SEQUENCE_LENGTH = 20
        START_DATE = "2019-01-01"
        END_DATE = "2021-12-31"
        TRAIN_YEARS = 1
        NUM_SEEDS = 2
        TOP_SEEDS = 1
        VAL_FRAC = 0.20

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
    print(f"Output: {len(out)} rows, {out['date'].nunique()} dates, {out['ticker'].nunique()} tickers  PASS")
    print(f"Columns: {out.columns.tolist()}  PASS")
    print(f"Position range: [{out['position'].min():.4f}, {out['position'].max():.4f}]  PASS")
    print(f"Checkpoints saved: {[p.name for p in saved]}  PASS")
