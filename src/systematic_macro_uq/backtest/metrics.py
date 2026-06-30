import numpy as np


def annualized_return(portfolio_returns):

    total_growth = (1 + portfolio_returns).prod()
    n_years = len(portfolio_returns) / 252
    cagr = total_growth ** (1 / n_years) - 1
    return cagr


def annualized_vol(portfolio_returns):

    return portfolio_returns.std() * np.sqrt(252)


def sharpe(annualized_ret, annualized_vol, rf_rate = 0.03):
    
    return (annualized_ret - rf_rate) / annualized_vol if annualized_vol > 0 else np.nan


def max_drawdown(equity_curve):
    cumulative_max  = equity_curve.cummax()
    return min(equity_curve / cumulative_max - 1)


def hit_rate(portfolio_returns):
    return (portfolio_returns > 0).mean()


def compute_performance_metrics(portfolio_returns, equity_curve):

    metrics_dict = {}
    metrics_dict["return"] = annualized_return(portfolio_returns)
    metrics_dict["vol"] = annualized_vol(portfolio_returns)
    # metrics_dict["sharpe"] = sharpe(portfolio_returns)
    metrics_dict["sharpe"] = sharpe(metrics_dict["return"], metrics_dict["vol"])
    metrics_dict["max_drawdown"] = max_drawdown(equity_curve)
    metrics_dict["hit_rate"] = hit_rate(portfolio_returns)

    return metrics_dict
