import numpy as np
import pandas as pd


def _rule_group_cols(df):
    group_cols = ["strategy", "cost_bps"]
    if "placebo_draw" in df.columns:
        group_cols.append("placebo_draw")
    return group_cols


def add_tradeability_score(
    posterior_df,
    tradeability_config,
):

    df = posterior_df.copy()
    a = tradeability_config.evidence_strength

    df["tradeability_score"] = (a * df["prob_positive"] 
                                - tradeability_config.score_hurdle 
                                - tradeability_config.penalty_dead * df["prob_dead"]
                                - tradeability_config.penalty_negative * df["prob_negative"]
    )

    return df


def apply_activate_hold_rule(
    scored_posterior_df,
    config,
):

    group_cols = _rule_group_cols(scored_posterior_df)

    df = scored_posterior_df.sort_values(group_cols + ["month"]).copy()
    df["evidence_months"] = df["evidence_time"] * 12
    tradeability_config = config.tradeability

    condition = (
        (df["evidence_months"] >= tradeability_config.min_validation_months)
        & (df["prob_positive"] >= tradeability_config.activate_prob_threshold)
        & (df["tradeability_score"] > 0)
    ) 
    
    df["decision"] = np.where(
        condition,
        "activate",
        "hold"
    )


    df["active"] = df["decision"] == "activate"
    df["active_next_month"] = (
        df.groupby(group_cols)["active"]
        .shift(1)
        .fillna(False)
    )
    df["holdout"] = df["month"] >= pd.Timestamp(config.HOLDOUT_START)

    return df
