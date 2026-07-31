import numpy as np


def annualized_return(portfolio_returns, trading_days_per_year):
    """Computes the compound annual growth rate of portfolio returns."""

    total_growth = (1 + portfolio_returns).prod()
    n_years = len(portfolio_returns) / trading_days_per_year
    cagr = total_growth ** (1 / n_years) - 1
    return cagr


def annualized_vol(portfolio_returns, trading_days_per_year):
    """Computes annualized portfolio return volatility."""

    return portfolio_returns.std() * np.sqrt(trading_days_per_year)


def sharpe(portfolio_returns, trading_days_per_year):
    """Computes the annualized Sharpe ratio with a zero cash rate."""
    daily_vol = portfolio_returns.std()
    if daily_vol <= 0:
        return np.nan
    return (
        portfolio_returns.mean()
        / daily_vol
        * np.sqrt(trading_days_per_year)
    )


def max_drawdown(equity_curve):
    """Computes the largest peak to trough decline in an equity curve."""
    cumulative_max  = equity_curve.cummax()
    return min(equity_curve / cumulative_max - 1)


def hit_rate(portfolio_returns):
    """Computes the fraction of portfolio returns above zero."""
    return (portfolio_returns > 0).mean()


def compute_performance_metrics(
    portfolio_returns,
    equity_curve,
    trading_days_per_year,
):
    """Computes the reported return, risk, and drawdown statistics."""

    metrics_dict = {}
    metrics_dict["return"] = annualized_return(
        portfolio_returns,
        trading_days_per_year,
    )
    metrics_dict["vol"] = annualized_vol(
        portfolio_returns,
        trading_days_per_year,
    )
    metrics_dict["sharpe"] = sharpe(
        portfolio_returns,
        trading_days_per_year,
    )
    metrics_dict["max_drawdown"] = max_drawdown(equity_curve)
    metrics_dict["hit_rate"] = hit_rate(portfolio_returns)

    return metrics_dict
