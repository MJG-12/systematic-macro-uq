
def add_drift_aware_turnover(position_df):
    """Computes turnover after allowing existing positions to drift.

    Each previous position is updated for its market return relative to the
    portfolio return before it is compared with the next target position.
    """
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


def add_roll_aware_turnover(position_df):
    """Charges both roll legs when the held contract changes."""

    position_df = position_df.sort_values(["market", "date"]).copy()
    position_df = add_drift_aware_turnover(position_df)

    previous_contract = (
        position_df.groupby("market")["held_instrument_id"].shift(1)
    )
    position_df["roll_trade"] = (
        position_df["held_instrument_id"].notna()
        & previous_contract.notna()
        & position_df["held_instrument_id"].ne(previous_contract)
    )

    position_df["ordinary_turnover"] = position_df["turnover"].where(
        ~position_df["roll_trade"],
        0.0,
    )
    position_df["roll_turnover"] = (
        position_df["pre_trade_position"].abs()
        + position_df["position"].abs()
    ).where(position_df["roll_trade"], 0.0)
    position_df["turnover"] = (
        position_df["ordinary_turnover"]
        + position_df["roll_turnover"]
    )

    return position_df


def add_linear_costs(position_df, cost_bps):
    """Applies one way costs to total turnover."""
    
    position_df = position_df.copy()
    position_df[f"cost_{cost_bps}bps"] = position_df["turnover"] * cost_bps /10000
    return position_df
