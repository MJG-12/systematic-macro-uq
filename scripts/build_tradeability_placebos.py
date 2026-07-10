import pandas as pd
from systematic_macro_uq.config import RunConfig
from systematic_macro_uq.tradeability.placebos import (compute_placebo_evidence,
                                                       summarize_placebo_activations)
from systematic_macro_uq.tradeability.posterior import compute_three_state_posterior
from systematic_macro_uq.tradeability.rules import (add_tradeability_score,
                                                    apply_activate_hold_rule)


def build_tradeability_placebos(config):
    monthly_tradeability_evidence = pd.read_parquet(
        config.backtest_dir / "monthly_tradeability_evidence.parquet"
    )

    placebo_evidence = compute_placebo_evidence(monthly_tradeability_evidence, config.tradeability)
    posterior = compute_three_state_posterior(placebo_evidence, config.tradeability)
    
    scored_posterior = add_tradeability_score(
        posterior,
        config.tradeability,
    )
    tradeability_rules = apply_activate_hold_rule(scored_posterior, config)

    tradeability_summary = summarize_placebo_activations(
        tradeability_rules,
        config,
    )

    tradeability_rules.to_parquet(
        config.backtest_dir / "tradeability_rules_placebo.parquet",
        index=False,
    )

    tradeability_summary.to_parquet(
        config.backtest_dir / "tradeability_summary_placebo.parquet",
        index=False,
    )

    tradeability_summary.to_csv(
        config.backtest_dir / "tradeability_summary_placebo.csv",
        index=False,
    )

    return tradeability_rules


if __name__ == "__main__":
    build_tradeability_placebos(RunConfig())
