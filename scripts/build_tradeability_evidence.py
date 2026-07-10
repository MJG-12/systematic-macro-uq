import pandas as pd
from systematic_macro_uq.config import RunConfig
from systematic_macro_uq.tradeability.evidence import compute_evidence

def build_tradeability_evidence(config):
    portfolio_results = pd.read_parquet(
        config.backtest_dir / "portfolio_results.parquet"
    )

    evidence = compute_evidence(portfolio_results)

    evidence.to_parquet(
        config.backtest_dir / "monthly_tradeability_evidence.parquet",
        index=False,
    )

    evidence.to_csv(
        config.backtest_dir / "monthly_tradeability_evidence.csv",
        index=False,
    )

    return evidence


if __name__ == "__main__":
    build_tradeability_evidence(RunConfig())