import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def compute_metrics(daily_returns: pd.Series) -> dict:
    ann_return = float(daily_returns.mean() * 252)
    ann_vol = float(daily_returns.std() * (252 ** 0.5))
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    cumulative = (1 + daily_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
    hit_rate = float((daily_returns > 0).mean())
    avg_turnover = float(daily_returns.abs().mean())

    return {
        "annualized_return": ann_return,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "hit_rate": hit_rate,
        "avg_abs_daily_return": avg_turnover,
    }


def plot_pnl(daily_returns: pd.Series, out_path: str = "results/pnl.png") -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cumulative = (1 + daily_returns).cumprod()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(cumulative.index, cumulative.values, linewidth=1.2)
    ax.set_title("Cumulative Portfolio PnL")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"PnL chart saved → {out_path}")


if __name__ == "__main__":
    import numpy as np

    # --- Test 1: Sharpe ratio with known mean/std ---
    rng = np.random.default_rng(0)
    n = 10_000
    daily_mu, daily_sig = 0.001, 0.01
    returns = pd.Series(rng.normal(daily_mu, daily_sig, n))
    metrics = compute_metrics(returns)

    expected_sharpe = daily_mu / daily_sig * (252 ** 0.5)
    assert abs(metrics["sharpe"] - expected_sharpe) < 0.15, (
        f"Sharpe mismatch: expected ~{expected_sharpe:.2f}, got {metrics['sharpe']:.2f}"
    )
    print(f"Sharpe ≈ {metrics['sharpe']:.4f} (expected ~{expected_sharpe:.4f})  PASS")

    # --- Test 2: max drawdown is negative and plausible ---
    # Up for 200 days then down for 100 days
    drawdown_returns = pd.Series([0.001] * 200 + [-0.003] * 100)
    metrics_dd = compute_metrics(drawdown_returns)
    assert metrics_dd["max_drawdown"] < 0, "Max drawdown must be negative"
    assert metrics_dd["max_drawdown"] > -1.0, "Max drawdown cannot exceed -100%"
    print(f"Max drawdown: {metrics_dd['max_drawdown']:.4f}  PASS")

    # --- Test 3: hit rate ---
    pos_returns = pd.Series([0.01] * 60 + [-0.01] * 40)
    metrics_hr = compute_metrics(pos_returns)
    assert abs(metrics_hr["hit_rate"] - 0.6) < 1e-6, (
        f"Hit rate mismatch: {metrics_hr['hit_rate']}"
    )
    print(f"Hit rate: {metrics_hr['hit_rate']:.2f}  PASS")

    # --- Test 4: PnL plot saves ---
    plot_returns = pd.Series(
        rng.normal(0.0005, 0.01, 252),
        index=pd.bdate_range("2020-01-01", periods=252),
    )
    plot_pnl(plot_returns, out_path="results/test_pnl.png")
    assert Path("results/test_pnl.png").exists(), "Plot file not found"
    print(f"PnL plot saved  PASS")

    print("\nAll metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
