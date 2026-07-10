import pandas as pd


def summarize_posterior_paths(posterior_df):

    df = posterior_df.copy()
    df = df.sort_values(["strategy", "cost_bps", "month"])

    grouped = df.groupby(["strategy", "cost_bps"])

    final_values = grouped.tail(1)[["strategy", 
                                    "cost_bps", 
                                    "prob_negative", 
                                    "prob_dead", 
                                    "prob_positive", 
                                    "posterior_mean"]].rename(
        columns = {"prob_negative": "final_prob_negative", 
                   "prob_dead": "final_prob_dead", 
                   "prob_positive": "final_prob_positive", 
                   "posterior_mean": "final_posterior_mean"
                   }
    )      

    max_min_summary = grouped.agg(
        max_prob_positive = ("prob_positive", "max"),
        max_prob_negative = ("prob_negative", "max"),
        min_prob_dead = ("prob_dead", "min"),
        months_prob_dead_below_50 = ("prob_dead", lambda x: (x < 0.5).sum()),
        months_prob_positive_above_50 = ("prob_positive", lambda x: (x > 0.5).sum())
    ).reset_index()

    summary = pd.merge(final_values,
                            max_min_summary,
                            on=["strategy", "cost_bps"],
                            how="left")
   
    return summary