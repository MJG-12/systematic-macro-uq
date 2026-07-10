import numpy as np
import pandas as pd

def compute_placebo_evidence(monthly_evidence_df, tradeability_config):

    base_df = monthly_evidence_df[
        [
            "strategy",
            "cost_bps",
            "month",
            "monthly_gross_payoff",
            "monthly_cost",
        ]
    ].copy()

    rng = np.random.default_rng(tradeability_config.placebo_seed)
    placebo_draws = []

    for i in range(tradeability_config.placebo_draws):
        random_sign_vector = rng.choice([-1, 1], size=len(base_df))
        df = base_df.copy()
        df["placebo_draw"] = i
        df["monthly_gross_payoff"] = random_sign_vector * df["monthly_gross_payoff"]
        df["monthly_net_payoff"] = df["monthly_gross_payoff"] - df["monthly_cost"]

        df["trailing_vol"] = (
            df.groupby(["strategy", "cost_bps"])["monthly_net_payoff"]
            .transform(lambda x: x.rolling(36, min_periods=24).std())
        ) 

        df["delta_Y"] = (
            df["monthly_net_payoff"] 
            / df["trailing_vol"] 
            * np.sqrt(1 / 12)
        )
        
        df["Y_cumulative"] =  (
             df.groupby(["strategy", "cost_bps"])["delta_Y"]
             .cumsum()
        )

        placebo_draws.append(df)

    return pd.concat(placebo_draws, ignore_index=True)



def summarize_placebo_activations(placebo_rules_df, config):

    group_cols = ["strategy", "cost_bps", "placebo_draw"]
    df = placebo_rules_df.sort_values(by=group_cols + ["month"]).copy()
    df = df[df["month"] < pd.Timestamp(config.HOLDOUT_START)].copy()

    first_activation = (
        df.loc[df["active"]]
        .groupby(group_cols)["month"]
        .min()
        .rename("first_activation_month")
    )

    summary = (
        df.groupby(group_cols)
        .agg(
            ever_activated=("active", "any"),
            active_months=("active", "sum"),
        )
        .join(first_activation)
        .reset_index()
    )

    cols = group_cols + [
        "ever_activated",
        "first_activation_month",
        "active_months",
    ]

    return summary[cols]
