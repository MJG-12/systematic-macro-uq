import numpy as np


def add_ewma_volatility(
    df,
    span,
    trading_days_per_year,
):
    df = df.sort_values(["market", "date"]).copy()

    squared_returns = df["held_daily_return"].pow(2)
    ewma_variance = squared_returns.groupby(df["market"]).transform(
        lambda x: x.ewm(span=span, adjust=False).mean()
    )

    df["daily_pct_vol"] = np.sqrt(ewma_variance)
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
