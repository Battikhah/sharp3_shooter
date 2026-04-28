import pandas as pd


def simulate(
    positions_df: pd.DataFrame,
    target_vol: float = 0.10,
    cost_bps: float = 0,
) -> pd.DataFrame:
    """
    Convert walk-forward positions into daily portfolio returns.

    positions_df: columns [date, ticker, position, fwd_return, vs_factor]
    Returns DataFrame with column [portfolio_return] indexed by date.
    """
    df = positions_df.copy().sort_values(["ticker", "date"])

    df["weight"] = df["position"] * target_vol * df["vs_factor"]
    df["ticker_return"] = df["weight"] * df["fwd_return"]

    if cost_bps > 0:
        df["weight_change"] = (
            df.groupby("ticker")["weight"].diff().abs().fillna(0.0)
        )
        df["ticker_return"] -= df["weight_change"] * cost_bps / 10_000

    daily_returns = df.groupby("date")["ticker_return"].mean()
    return daily_returns.rename("portfolio_return").to_frame()


if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", "2020-12-31")
    n = len(dates)
    tickers = ["SPY", "QQQ"]

    rows = []
    for ticker in tickers:
        for date in dates:
            rows.append({
                "date": date,
                "ticker": ticker,
                "position": rng.uniform(-1, 1),
                "fwd_return": rng.normal(0.001, 0.01),
                "vs_factor": rng.uniform(5, 20),
            })
    positions_df = pd.DataFrame(rows)

    # --- Test 1: output shape and columns ---
    result = simulate(positions_df, target_vol=0.1, cost_bps=0)
    assert "portfolio_return" in result.columns, "Missing portfolio_return column"
    assert len(result) == n, f"Expected {n} rows, got {len(result)}"
    assert result.index.name == "date", "Index should be named 'date'"
    print(f"Output shape: {result.shape}  PASS")
    print(f"Daily return range: [{result['portfolio_return'].min():.4f}, {result['portfolio_return'].max():.4f}]  PASS")

    # --- Test 2: costs reduce total return ---
    result_cost = simulate(positions_df, target_vol=0.1, cost_bps=10)
    assert result_cost["portfolio_return"].sum() <= result["portfolio_return"].sum(), (
        "Costs should not increase total return"
    )
    print(f"Cost adjustment reduces return  PASS")

    # --- Test 3: known math ---
    # positions=1, vs_factor=1/target_vol=10 → weight=1; fwd_return=0.01 → ticker_return=0.01
    single_df = pd.DataFrame({
        "date": dates,
        "ticker": "SPY",
        "position": 1.0,
        "fwd_return": 0.01,
        "vs_factor": 1.0 / 0.1,
    })
    result_known = simulate(single_df, target_vol=0.1, cost_bps=0)
    assert (result_known["portfolio_return"] - 0.01).abs().max() < 1e-6, (
        f"Expected 0.01 every day, got {result_known['portfolio_return'].describe()}"
    )
    print(f"Known return computation (0.01 every day)  PASS")
