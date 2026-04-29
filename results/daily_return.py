"""
Daily Returns Analysis Script

Reads daily_returns.parquet and displays comprehensive analysis in multiple formats:
- Summary statistics
- Performance metrics
- Time series visualizations
- Distribution analysis
- Rolling statistics
- Monthly/yearly breakdown
- Risk metrics
- Cumulative performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Set plotting style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def load_daily_returns():
    """Load daily_returns.parquet from results directory."""
    parquet_path = Path(__file__).parent / "daily_returns.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"File not found: {parquet_path}")
    return pd.read_parquet(parquet_path)


def print_summary_statistics(returns_df):
    """Print summary statistics of daily returns."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"\nDate Range: {returns_df.index.min()} to {returns_df.index.max()}")
    print(f"Number of Days: {len(returns_df)}")
    print(f"\n{returns_df.describe().to_string()}")


def print_performance_metrics(returns_df):
    """Print key performance metrics."""
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)

    # Assuming the column is 'portfolio_return' or the first numeric column
    col = returns_df.columns[0] if len(returns_df.columns) > 0 else None
    if col is None:
        print("No return data found")
        return

    returns = returns_df[col]

    # Total return
    total_return = (1 + returns).prod() - 1
    print(f"\nTotal Return: {total_return:.2%}")

    # Annualized return
    n_years = len(returns) / 252
    annualized_return = (1 + total_return) ** (1 / n_years) - 1
    print(f"Annualized Return: {annualized_return:.2%}")

    # Volatility
    volatility = returns.std() * np.sqrt(252)
    print(f"Annualized Volatility: {volatility:.2%}")

    # Sharpe ratio (assuming 0% risk-free rate)
    sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")

    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    print(f"Maximum Drawdown: {max_drawdown:.2%}")

    # Win rate
    win_rate = (returns > 0).sum() / len(returns)
    print(f"Win Rate: {win_rate:.2%}")

    # Average win / loss
    avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
    avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0
    print(f"Average Win: {avg_win:.4%}")
    print(f"Average Loss: {avg_loss:.4%}")

    # Profit factor
    if avg_loss != 0:
        profit_factor = (returns[returns > 0].sum()) / abs((returns[returns < 0].sum()))
        print(f"Profit Factor: {profit_factor:.4f}")

    # Max consecutive wins/losses
    win_streak = (returns > 0).astype(int)
    consecutive_wins = win_streak.groupby((win_streak != win_streak.shift()).cumsum()).sum().max()
    loss_streak = (returns < 0).astype(int)
    consecutive_losses = loss_streak.groupby((loss_streak != loss_streak.shift()).cumsum()).sum().max()
    print(f"Max Consecutive Wins: {consecutive_wins}")
    print(f"Max Consecutive Losses: {consecutive_losses}")


def print_monthly_breakdown(returns_df):
    """Print monthly and yearly return breakdown."""
    print("\n" + "=" * 80)
    print("MONTHLY AND YEARLY BREAKDOWN")
    print("=" * 80)

    col = returns_df.columns[0]
    returns = returns_df[col]

    # Monthly returns
    monthly_returns = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    print(f"\nMonthly Returns:\n{monthly_returns.to_string()}")

    # Yearly returns
    yearly_returns = returns.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    print(f"\n\nYearly Returns:\n{yearly_returns.to_string()}")

    # Monthly statistics
    print("\n\nMonthly Statistics:")
    print(f"Average Monthly Return: {monthly_returns.mean():.4%}")
    print(f"Monthly Volatility: {monthly_returns.std():.4%}")
    print(f"Best Month: {monthly_returns.max():.4%} ({monthly_returns.idxmax().strftime('%Y-%m')})")
    print(f"Worst Month: {monthly_returns.min():.4%} ({monthly_returns.idxmin().strftime('%Y-%m')})")


def plot_cumulative_returns(returns_df, output_dir):
    """Plot cumulative returns over time."""
    col = returns_df.columns[0]
    returns = returns_df[col]
    cumulative_returns = (1 + returns).cumprod()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(cumulative_returns.index, cumulative_returns.values, linewidth=2)
    ax.fill_between(cumulative_returns.index, cumulative_returns.values, alpha=0.3)
    ax.set_title("Cumulative Returns Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (Factor)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "cumulative_returns.png", dpi=150, bbox_inches="tight")
    print("\n✓ Saved: cumulative_returns.png")
    plt.close()


def plot_daily_returns_distribution(returns_df, output_dir):
    """Plot histogram and KDE of daily returns."""
    col = returns_df.columns[0]
    returns = returns_df[col]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram with KDE
    axes[0].hist(returns, bins=50, alpha=0.7, edgecolor="black")
    returns.plot(kind="kde", ax=axes[0], secondary_y=False, linewidth=2, color="red")
    axes[0].set_title("Distribution of Daily Returns", fontweight="bold")
    axes[0].set_xlabel("Daily Return")
    axes[0].set_ylabel("Frequency")

    # Q-Q plot
    from scipy import stats
    stats.probplot(returns, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q Plot (vs Normal Distribution)", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "returns_distribution.png", dpi=150, bbox_inches="tight")
    print("✓ Saved: returns_distribution.png")
    plt.close()


def plot_drawdown(returns_df, output_dir):
    """Plot cumulative returns and drawdown."""
    col = returns_df.columns[0]
    returns = returns_df[col]
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [2, 1]})

    # Cumulative returns
    axes[0].plot(cumulative.index, cumulative.values, linewidth=2, label="Cumulative Return")
    axes[0].fill_between(cumulative.index, cumulative.values, alpha=0.3)
    axes[0].set_title("Cumulative Returns with Drawdown", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Cumulative Return (Factor)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Drawdown
    axes[1].fill_between(drawdown.index, drawdown.values, alpha=0.5, color="red")
    axes[1].plot(drawdown.index, drawdown.values, linewidth=1, color="darkred")
    axes[1].set_title("Drawdown", fontweight="bold")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Drawdown %")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "drawdown.png", dpi=150, bbox_inches="tight")
    print("✓ Saved: drawdown.png")
    plt.close()


def plot_rolling_metrics(returns_df, output_dir, window=60):
    """Plot rolling volatility and Sharpe ratio."""
    col = returns_df.columns[0]
    returns = returns_df[col]

    rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
    rolling_sharpe = (returns.rolling(window=window).mean() * 252) / (returns.rolling(window=window).std() * np.sqrt(252))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Rolling volatility
    axes[0].plot(rolling_vol.index, rolling_vol.values, linewidth=2, color="blue")
    axes[0].fill_between(rolling_vol.index, rolling_vol.values, alpha=0.3, color="blue")
    axes[0].set_title(f"Rolling Annualized Volatility ({window}-day window)", fontweight="bold")
    axes[0].set_ylabel("Volatility")
    axes[0].grid(True, alpha=0.3)

    # Rolling Sharpe
    axes[1].plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2, color="green")
    axes[1].axhline(y=0, color="black", linestyle="--", linewidth=1)
    axes[1].fill_between(rolling_sharpe.index, rolling_sharpe.values, 0, alpha=0.3, color="green")
    axes[1].set_title(f"Rolling Sharpe Ratio ({window}-day window)", fontweight="bold")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Sharpe Ratio")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "rolling_metrics.png", dpi=150, bbox_inches="tight")
    print("✓ Saved: rolling_metrics.png")
    plt.close()


def plot_monthly_heatmap(returns_df, output_dir):
    """Plot monthly returns as a heatmap."""
    col = returns_df.columns[0]
    returns = returns_df[col]

    monthly_returns = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly_returns.index = pd.to_datetime(monthly_returns.index)

    # Create pivot table (year x month)
    pivot = pd.DataFrame(monthly_returns)
    pivot["Year"] = pivot.index.year
    pivot["Month"] = pivot.index.month
    pivot = pivot.set_index(["Year", "Month"])[col].unstack()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt=".2%", cmap="RdYlGn", center=0, cbar_kws={"label": "Monthly Return"}, ax=ax)
    ax.set_title("Monthly Returns Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "monthly_heatmap.png", dpi=150, bbox_inches="tight")
    print("✓ Saved: monthly_heatmap.png")
    plt.close()


def plot_returns_vs_time(returns_df, output_dir):
    """Plot daily returns with trend."""
    col = returns_df.columns[0]
    returns = returns_df[col]

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["green" if x > 0 else "red" for x in returns.values]
    ax.bar(returns.index, returns.values, color=colors, alpha=0.6, width=1)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_title("Daily Returns", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Return")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "daily_returns_bars.png", dpi=150, bbox_inches="tight")
    print("✓ Saved: daily_returns_bars.png")
    plt.close()


def main():
    """Main execution function."""
    # Load data
    returns_df = load_daily_returns()
    print(f"\nLoaded daily_returns.parquet: {len(returns_df)} rows")

    # Create output directory
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)

    # Print statistics
    print_summary_statistics(returns_df)
    print_performance_metrics(returns_df)
    print_monthly_breakdown(returns_df)

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    plot_cumulative_returns(returns_df, output_dir)
    plot_daily_returns_distribution(returns_df, output_dir)
    plot_drawdown(returns_df, output_dir)
    plot_rolling_metrics(returns_df, output_dir)
    plot_monthly_heatmap(returns_df, output_dir)
    plot_returns_vs_time(returns_df, output_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()

    print("\n" + "=" * 80)
    print("ADDITIONAL ANALYSIS: SHARPE RATIO BY PERIOD")
    print("=" * 80)
    returns = pd.read_parquet("results/daily_returns.parquet")["portfolio_return"]

    periods = {
        "2010-2025": ("2010-01-01", "2025-12-31"),
        "2015-2025": ("2015-01-01", "2025-12-31"),
        "2010-2015": ("2010-01-01", "2014-12-31"),
        "2015-2020": ("2015-01-01", "2019-12-31"),
        "2020-2025": ("2020-01-01", "2025-12-31"),
    }

    for label, (start, end) in periods.items():
        r = returns.loc[start:end]
        sharpe = r.mean() / r.std() * np.sqrt(252)
        print(f"{label:12s}  Sharpe: {sharpe:.4f}")
