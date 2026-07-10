import numpy as np


def compute_three_state_posterior(evidence_df, tradeability_config):

    df = evidence_df.copy()

    placebo_col = ["placebo_draw"] if "placebo_draw" in df.columns else []
    groupby_cols = placebo_col + ["strategy", "cost_bps"]
    sort_cols = groupby_cols + ["month"]

    df = df.sort_values(by = sort_cols)

    a = tradeability_config.evidence_strength
    prior_dead = tradeability_config.prior_dead
    prior_positive = (1 - prior_dead) / 2
    prior_negative = (1 - prior_dead) / 2
    
    df["valid_evidence"] = df["delta_Y"].notna()

    df["evidence_count"] = (
        df.groupby(groupby_cols)["valid_evidence"]
        .cumsum()
    )

    df["evidence_time"] = df["evidence_count"] / 12
    
    df["likelihood_dead"] = 1
    df["likelihood_negative"] = np.exp(-a * df["Y_cumulative"] - 0.5 * a**2 * df["evidence_time"])
    df["likelihood_positive"] = np.exp(a * df["Y_cumulative"] - 0.5 * a**2 * df["evidence_time"])

    df["weight_negative"] = prior_negative * df["likelihood_negative"]
    df["weight_dead"] = prior_dead * df["likelihood_dead"]
    df["weight_positive"] = prior_positive * df["likelihood_positive"]

    df["total_weight"] = (
        df["weight_negative"]
        + df["weight_dead"]
        + df["weight_positive"]
    )

    df["prob_negative"] = df["weight_negative"] / df["total_weight"]
    df["prob_dead"] = df["weight_dead"] / df["total_weight"]
    df["prob_positive"] = df["weight_positive"] / df["total_weight"]

    df["posterior_mean"] = (
        df["prob_negative"] * (-a)
        + df["prob_positive"] * a
    )

    df["posterior_second_moment"] = (
        (df["prob_negative"] + df["prob_positive"]) * a**2
    )

    df["posterior_variance"] = (
        df["posterior_second_moment"] - df["posterior_mean"]**2
    )


    cols = placebo_col + [
        "strategy",
        "cost_bps",
        "month",
        "monthly_net_payoff",
        "trailing_vol",
        "Y_cumulative",
        "delta_Y",
        "evidence_time",
        "prob_negative",
        "prob_dead",
        "prob_positive",
        "posterior_mean",
        "posterior_variance",
    ]

    return df[cols]
