# 

def add_drift_aware_turnover(position_df):

    position_df = position_df.copy()
    position_df["gross_portfolio_return"] = (
        position_df.groupby("date")["return_contribution"]
        .transform("sum")
    )

    position_df["pre_trade_position"] = (position_df["position"].shift(1) *
                                         (1 + position_df["return_contribution"]) /
                                         (1 + position_df["gross_portfolio_return"])
    )

    position_df["turnover"] = abs(position_df["position"] - position_df["pre_trade_position"])


    return position_df


def add_linear_costs(position_df, cost_bps):
    
    position_df = position_df.copy()
    position_df["cost"] = position_df["turnover"] * cost_bps
    return position_df


