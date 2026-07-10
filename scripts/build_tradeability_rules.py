import pandas as pd
from systematic_macro_uq.config import RunConfig
from systematic_macro_uq.tradeability.rules import (add_tradeability_score,
                                                    apply_activate_hold_rule)

    
def build_tradeability_rules(config):
    posterior = pd.read_parquet(config.backtest_dir / "tradeability_posterior.parquet")

    scored_posterior = add_tradeability_score(
        posterior,
        config.tradeability,
    )
    tradeability_rules = apply_activate_hold_rule(scored_posterior, config)
    
    tradeability_rules.to_parquet(
        config.backtest_dir / "tradeability_rules.parquet",
        index=False,
    )

    return tradeability_rules
 

if __name__ == "__main__":
    build_tradeability_rules(RunConfig())
