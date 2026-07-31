from pathlib import Path

import pandas as pd

from systematic_macro_uq.backtest.evaluation import (
    evaluate_position_strategies,
)
from systematic_macro_uq.config import RunConfig
from systematic_macro_uq.conformal.strategies import (
    build_conformal_strategies,
)
from systematic_macro_uq.forecasting.pipeline import calculate_forecasts


def run_development_conformal_sizing(config):
    """Evaluates conformal sizing across development coverage levels."""
    holdout_start = pd.Timestamp(config.HOLDOUT_START, tz="UTC")
    evaluation_start = pd.Timestamp(config.DEVELOPMENT_START, tz="UTC")
    daily = pd.read_parquet(
        config.futures_daily_path,
        filters=[("date", "<", holdout_start)],
    )
    forecasts = calculate_forecasts(daily, config)
    strategies, bounds, application = build_conformal_strategies(
        forecasts,
        coverages=config.conformal.development_coverages,
        config=config,
        evaluation_start=evaluation_start,
        evaluation_end=holdout_start,
    )

    output_dir = (
        Path(config.conformal_backtest_dir) / "development_sizing"
    )
    result = evaluate_position_strategies(
        config=config,
        strategies=strategies,
        evaluation_start=evaluation_start,
        evaluation_end=holdout_start,
        output_dir=output_dir,
    )
    bounds.to_parquet(output_dir / "bounds.parquet", index=False)
    application.to_csv(
        output_dir / "application_diagnostics.csv",
        index=False,
    )

    print(result.summary.to_string(index=False))
    print()
    print(application.to_string(index=False))
    return result.summary


if __name__ == "__main__":
    run_development_conformal_sizing(RunConfig())
