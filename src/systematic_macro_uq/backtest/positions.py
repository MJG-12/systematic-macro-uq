import pandas as pd
import numpy as np


def construct_equal_weight_positions(df, signal_col, n_markets):

    df = df.sort_values(["date", "market"]).copy()
    df["position"] = df[signal_col].fillna(0) * (1/n_markets)
    df["return_contribution"] = df["position"] * df["held_daily_return"].fillna(0)

    return df[["date", "market", "position", "held_daily_return", "return_contribution"]]


def construct_vol_scaled_positions(df, signal_col, vol_lookback=63):

    df = df.sort_values(["date", "market"]).copy()

    vol_col = f"vol_{vol_lookback}d"
    df["raw_position"] = np.where((df[vol_col].notna()) & (df[vol_col] > 0),
                                  df[signal_col].fillna(0) / df[vol_col],
                                  0
    )

    df["total_position"] =(
        df.groupby("date")["raw_position"]
        .transform(lambda x: x.abs().sum())
    )
    df["position"] = np.where(df["total_position"] > 0,
        df["raw_position"] / df["total_position"],
        0
    )
    
    df["return_contribution"] = df["position"].fillna(0) * df["held_daily_return"].fillna(0)

    return df[["date", "market", "position", "held_daily_return", "return_contribution"]]


def aggregate_portfolio(position_df, cost_bps):

    cost_col = f"cost_{cost_bps}bps"
    portfolio = position_df.groupby("date")[["return_contribution", cost_col]].sum()
    portfolio = portfolio.rename(columns={
        "return_contribution": "gross_return",
        cost_col: "cost",
    })
    
    portfolio["net_return"] = portfolio["gross_return"] - portfolio["cost"]
    portfolio["equity_curve"] = (1 + portfolio["net_return"]).cumprod()

    return portfolio
