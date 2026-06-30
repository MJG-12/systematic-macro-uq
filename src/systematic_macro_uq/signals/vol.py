import numpy as np


def add_trailing_vol(df, lookback):
    df = df.sort_values(["market", "date"]).copy()
    df[f"vol_{lookback}d"] = (df.groupby("market")["held_daily_return"]
                              .transform(
        lambda x: x.rolling(lookback).std().shift(1) * np.sqrt(252))
    )
    return df


def create_vol_signals(df, lookback_list =[21, 63]):
    for lookback in lookback_list:
        df = add_trailing_vol(df, lookback)
    return df