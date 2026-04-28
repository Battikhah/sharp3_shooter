import torch


def sharpe_loss(
    positions: torch.Tensor,
    fwd_returns: torch.Tensor,
    vs_factors: torch.Tensor,
    target_vol: float = 0.10,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Differentiable negative annualized Sharpe ratio computed on a batch.

    positions:   (batch,) model output in [-1, 1]
    fwd_returns: (batch,) next-day asset returns
    vs_factors:  (batch,) 1/σ_t for vol scaling
    """
    weights = positions * target_vol * vs_factors
    portfolio_returns = weights * fwd_returns
    mean = portfolio_returns.mean()
    std = portfolio_returns.std()
    sharpe = mean / (std + eps) * (252 ** 0.5)
    return -sharpe


if __name__ == "__main__":
    # --- Test 1: scalar, requires_grad, backward ---
    n = 256
    torch.manual_seed(0)
    positions = torch.randn(n).clamp(-1, 1).requires_grad_(True)
    fwd_returns = torch.randn(n) * 0.01
    vs_factors = torch.ones(n) * 10.0

    loss = sharpe_loss(positions, fwd_returns, vs_factors, target_vol=0.1)

    assert loss.shape == (), f"Expected scalar, got {loss.shape}"
    assert loss.requires_grad, "Loss must require grad"
    loss.backward()
    assert positions.grad is not None, "No gradient on positions"
    print(f"Scalar + requires_grad + backward  PASS  (loss={loss.item():.4f})")

    # --- Test 2: known Sharpe ---
    # positions=1, vs_factors=1/target_vol → weight=1, portfolio_returns=fwd_returns
    # Construct fwd_returns with exact mean/std via standardization
    n = 500
    daily_std = 0.01
    daily_mean = 2.0 / (252 ** 0.5) * daily_std  # Sharpe=2 with this std
    base = torch.linspace(-1.0, 1.0, n)
    fwd_returns = (base - base.mean()) / base.std() * daily_std + daily_mean

    positions = torch.ones(n, requires_grad=True)
    vs_factors = torch.ones(n) * (1.0 / 0.1)  # weights = 1.0 exactly

    loss = sharpe_loss(positions, fwd_returns, vs_factors, target_vol=0.1)
    expected_sharpe = (fwd_returns.mean() / fwd_returns.std() * (252 ** 0.5)).item()

    assert abs(loss.item() + expected_sharpe) < 1e-3, (
        f"Expected -sharpe ≈ {-expected_sharpe:.4f}, got {loss.item():.4f}"
    )
    print(f"Known Sharpe match  PASS  (expected {expected_sharpe:.4f}, got {-loss.item():.4f})")
