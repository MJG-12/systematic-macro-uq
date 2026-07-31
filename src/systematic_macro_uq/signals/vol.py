import numpy as np


def add_ewma_volatility(
    df,
    span,
    trading_days_per_year,
    long_run_weight,
    long_run_window_days,
):
    """Estimates volatility using Carver's short and long run blend.

    The long run average expands until a full window is available because the
    price history starts too late for a ten year window early in the sample.
    """
    df = df.sort_values(["market", "date"]).copy()

    squared_returns = df["held_daily_return"].pow(2)
    ewma_variance = squared_returns.groupby(df["market"]).transform(
        lambda x: x.ewm(span=span, adjust=False).mean()
    )
    short_run_vol = np.sqrt(ewma_variance)
    long_run_vol = short_run_vol.groupby(df["market"]).transform(
        lambda x: x.rolling(long_run_window_days, min_periods=1).mean()
    )

    df["daily_pct_vol"] = (
        long_run_weight * long_run_vol
        + (1 - long_run_weight) * short_run_vol
    )
    df["annualized_pct_vol"] = (
        df["daily_pct_vol"] * np.sqrt(trading_days_per_year)
    )
    df["daily_price_vol"] = (
        df["daily_pct_vol"] * df["selected_settlement_price"]
    )
    df["annualized_price_vol"] = (
        df["daily_price_vol"] * np.sqrt(trading_days_per_year)
    )

    return df
