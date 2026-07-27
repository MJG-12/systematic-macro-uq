import math
from statistics import NormalDist

import numpy as np


CENTRAL_RISK_COVERAGE = 0.95


def annualized_return(portfolio_returns, trading_days_per_year):

    total_growth = (1 + portfolio_returns).prod()
    n_years = len(portfolio_returns) / trading_days_per_year
    cagr = total_growth ** (1 / n_years) - 1
    return cagr


def annualized_vol(portfolio_returns, trading_days_per_year):

    return portfolio_returns.std() * np.sqrt(trading_days_per_year)


def sharpe(portfolio_returns, trading_days_per_year):
    daily_vol = portfolio_returns.std()
    if daily_vol <= 0:
        return np.nan
    return (
        portfolio_returns.mean()
        / daily_vol
        * np.sqrt(trading_days_per_year)
    )


def max_drawdown(equity_curve):
    cumulative_max  = equity_curve.cummax()
    return min(equity_curve / cumulative_max - 1)


def hit_rate(portfolio_returns):
    return (portfolio_returns > 0).mean()


def central_quantile_risk(
    portfolio_returns,
    trading_days_per_year,
    coverage,
):
    """Express a realized central absolute-return quantile as annualized risk."""
    returns = np.asarray(portfolio_returns)
    returns = returns[np.isfinite(returns)]
    rank = math.ceil((len(returns) + 1) * coverage)
    if not len(returns) or rank > len(returns):
        return np.nan

    radius = np.partition(np.abs(returns), rank - 1)[rank - 1]
    reference_radius = NormalDist().inv_cdf((1 + coverage) / 2)
    return (
        np.sqrt(trading_days_per_year)
        * radius
        / reference_radius
    )


def central_quantile_return_ratio(
    portfolio_returns,
    trading_days_per_year,
    coverage,
):
    """Annualized arithmetic return per unit of central-quantile risk."""
    returns = np.asarray(portfolio_returns)
    returns = returns[np.isfinite(returns)]
    risk = central_quantile_risk(
        returns,
        trading_days_per_year,
        coverage,
    )
    if not np.isfinite(risk) or risk <= 0:
        return np.nan
    annualized_mean = np.mean(returns) * trading_days_per_year
    return annualized_mean / risk


def compute_performance_metrics(
    portfolio_returns,
    equity_curve,
    trading_days_per_year,
):

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
    metrics_dict["central_95_risk"] = central_quantile_risk(
        portfolio_returns,
        trading_days_per_year,
        CENTRAL_RISK_COVERAGE,
    )
    metrics_dict["central_95_return_ratio"] = (
        central_quantile_return_ratio(
            portfolio_returns,
            trading_days_per_year,
            CENTRAL_RISK_COVERAGE,
        )
    )

    return metrics_dict
