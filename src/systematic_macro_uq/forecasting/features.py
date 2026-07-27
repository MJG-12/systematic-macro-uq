import numpy as np


def add_forward_return_target(
    df,
    horizon_days,
    trading_days_per_year,
    target_column,
):
    df = df.sort_values(["market", "date"]).copy()

    df["target_end_date"] = (
        df.groupby("market")["date"]
        .shift(-horizon_days)
    )

    return_index = (
        df["held_daily_return"]
        .fillna(0)
        .add(1)
        .groupby(df["market"])
        .cumprod()
    )
    future_return_index = (
        return_index.groupby(df["market"])
        .shift(-horizon_days)
    )

    forward_return = (
        future_return_index / return_index - 1
    )

    horizon_vol = (
        df["annualized_pct_vol"]
        * np.sqrt(horizon_days / trading_days_per_year)
    )

    df[target_column] = (
        forward_return / horizon_vol
    )

    return df
