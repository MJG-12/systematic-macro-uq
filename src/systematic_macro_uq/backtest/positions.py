import pandas as pd
import numpy as np


def estimate_instrument_diversification_multiplier(
    df,
    instrument_weights,
):
    """Estimates the fixed IDM from the initial estimation period.

    The omitted target volatility and average forecast constants are common
    across markets and therefore do not affect the correlation estimate.
    """

    df = df.sort_values(["market", "date"]).copy()
    by_market = df.groupby("market")
    forecast = by_market["forecast_combined"].shift(1)
    volatility = by_market["annualized_pct_vol"].shift(1)
    df["subsystem_return"] = (
        forecast
        / volatility
        * df["held_daily_return"]
    )

    returns = df.pivot(
        index="date",
        columns="market",
        values="subsystem_return",
    )
    weights = pd.Series(instrument_weights, dtype=float)
    missing_markets = weights.index.difference(returns.columns)
    if not missing_markets.empty:
        raise ValueError(
            f"Missing IDM returns for: {', '.join(missing_markets)}"
        )
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Instrument weights must sum to one.")

    correlation = returns.corr().reindex(
        index=weights.index,
        columns=weights.index,
    )
    if correlation.isna().any().any():
        raise ValueError("Complete IDM correlations are required.")

    correlation_values = correlation.to_numpy(copy=True)
    # Do not let unstable negative estimates create artificial diversification.
    correlation_values = np.maximum(correlation_values, 0)
    np.fill_diagonal(correlation_values, 1.0)

    portfolio_variance = (
        weights.to_numpy()
        @ correlation_values
        @ weights.to_numpy()
    )
    # Carver's fixed ceiling prevents an implausibly large IDM.
    return min(1 / np.sqrt(portfolio_variance), 2.5)


def construct_forecast_positions(
    df,
    instrument_weights,
    instrument_diversification_multiplier,
    volatility_target,
    average_forecast,
    forecast_column="forecast_combined",
):
    """Maps a forecast into lagged, volatility targeted positions."""

    df = df.sort_values(["market", "date"]).copy()
    by_market = df.groupby("market")

    df["instrument_weight"] = df["market"].map(instrument_weights)
    df["forecast_for_position"] = by_market[forecast_column].shift(1)
    df["volatility_for_position"] = by_market["annualized_pct_vol"].shift(1)

    valid = (
        df["instrument_weight"].notna()
        & df["forecast_for_position"].notna()
        & df["volatility_for_position"].gt(0)
    )
    df["position"] = 0.0
    df.loc[valid, "position"] = (
        df.loc[valid, "instrument_weight"]
        * instrument_diversification_multiplier
        * volatility_target
        / df.loc[valid, "volatility_for_position"]
        * df.loc[valid, "forecast_for_position"]
        / average_forecast
    )

    df["return_contribution"] = (
        df["position"] * df["held_daily_return"].fillna(0)
    )

    return df[
        [
            "date",
            "market",
            "held_instrument_id",
            "position",
            "instrument_weight",
            "forecast_for_position",
            "volatility_for_position",
            "held_daily_return",
            "return_contribution",
        ]
    ]


def aggregate_portfolio(position_df, cost_bps):
    """Aggregates market contributions and costs into a portfolio path."""

    cost_col = f"cost_{cost_bps}bps"
    portfolio = position_df.groupby("date")[["return_contribution", cost_col]].sum()
    portfolio = portfolio.rename(columns={
        "return_contribution": "gross_return",
        cost_col: "cost",
    })
    
    portfolio["net_return"] = portfolio["gross_return"] - portfolio["cost"]
    portfolio["equity_curve"] = (1 + portfolio["net_return"]).cumprod()

    return portfolio
