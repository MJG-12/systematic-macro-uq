import pandas as pd
from systematic_macro_uq.config import RunConfig
from systematic_macro_uq.tradeability.posterior import compute_three_state_posterior
from systematic_macro_uq.tradeability.diagnostics import summarize_posterior_paths


def build_tradeability_posterior(config):


    monthly_evidence = pd.read_parquet(config.backtest_dir / "monthly_tradeability_evidence.parquet")

    posterior = compute_three_state_posterior(
        monthly_evidence,
        config.tradeability,
    )
    diagnostics = summarize_posterior_paths(posterior)

    posterior.to_parquet(
        config.backtest_dir / "tradeability_posterior.parquet",
        index=False,
    )

    posterior.to_csv(
        config.backtest_dir / "tradeability_posterior.csv",
        index=False,
    )

    diagnostics.to_csv(
        config.backtest_dir / "posterior_diagnostics.csv",
        index=False,
    )

    return posterior


if __name__ == "__main__":
    build_tradeability_posterior(RunConfig())
