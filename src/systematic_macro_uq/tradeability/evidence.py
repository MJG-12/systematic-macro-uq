import numpy as np


def compute_evidence(portfolio_results_df):

    df = portfolio_results_df.copy()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    df["gross_growth_factor"] = df["gross_return"] + 1
    df["net_growth_factor"] = df["net_return"] + 1

    monthly_df = (df.groupby(["month", "strategy", "cost_bps"])
    .agg(
        monthly_gross_growth_factor=("gross_growth_factor", "prod"),
        monthly_net_growth_factor=("net_growth_factor", "prod")
    )
    .reset_index()
    .sort_values(["strategy", "cost_bps", "month"])
    )

    monthly_df["monthly_gross_payoff"] = monthly_df["monthly_gross_growth_factor"] - 1
    monthly_df["monthly_net_payoff"] = monthly_df["monthly_net_growth_factor"] - 1
    monthly_df["monthly_cost"] = monthly_df["monthly_gross_payoff"] - monthly_df["monthly_net_payoff"] 
    
    monthly_df["trailing_vol"] = (
    monthly_df.groupby(["strategy", "cost_bps"])["monthly_net_payoff"]
    .transform(lambda x: x.rolling(36, min_periods=24).std())
    )

    monthly_df["delta_Y"] = (
    monthly_df["monthly_net_payoff"] / monthly_df["trailing_vol"] * np.sqrt(1 / 12)
    )

    monthly_df["Y_cumulative"] = (
        monthly_df.groupby(["strategy", "cost_bps"])["delta_Y"]
        .cumsum()
    )
    
    return monthly_df[
        [
        "strategy", 
        "cost_bps", 
        "month",
        "monthly_gross_payoff",
        "monthly_net_payoff",
        "monthly_cost",
        "trailing_vol",
        "delta_Y",
        "Y_cumulative"
    ]
    ]