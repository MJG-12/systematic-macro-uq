
def add_drift_aware_turnover(position_df):

    position_df = position_df.copy()
    position_df["gross_portfolio_return"] = (
        position_df.groupby("date")["return_contribution"]
        .transform("sum")
    )

    asset_return = position_df["held_daily_return"].fillna(0)
    post_return_position = (position_df["position"] *
                            (1 + asset_return) /
                            (1 + position_df["gross_portfolio_return"])
    )

    position_df["post_return_position"] = post_return_position

    position_df["pre_trade_position"] = (
        position_df.groupby("market")["post_return_position"]
        .shift(1)
        .fillna(0)
    )
    
    position_df["turnover"] = abs(position_df["position"] - position_df["pre_trade_position"])

    return position_df


def add_linear_costs(position_df, cost_bps):
    
    position_df = position_df.copy()
    position_df[f"cost_{cost_bps}bps"] = position_df["turnover"] * cost_bps /10000
    return position_df

