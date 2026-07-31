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


HOLDOUT_FROZEN = True


def run_holdout_conformal_sizing(config, *, frozen):
    """Evaluates the selected conformal sizing rules on the frozen holdout."""
    if not frozen:
        raise RuntimeError(
            "Conformal holdout is locked. Set HOLDOUT_FROZEN=True only "
            "after the application and coverage are frozen."
        )

    daily = pd.read_parquet(config.futures_daily_path)
    forecasts = calculate_forecasts(daily, config)
    evaluation_start = pd.Timestamp(config.HOLDOUT_START, tz="UTC")
    evaluation_end = pd.Timestamp(config.END, tz="UTC")
    strategies, bounds, application = build_conformal_strategies(
        forecasts,
        coverages=(config.conformal.selected_coverage,),
        config=config,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
    )

    output_dir = Path(config.conformal_backtest_dir) / "holdout_sizing"
    result = evaluate_position_strategies(
        config=config,
        strategies=strategies,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
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
    run_holdout_conformal_sizing(
        RunConfig(),
        frozen=HOLDOUT_FROZEN,
    )
