import pandas as pd

from systematic_macro_uq.backtest.forecast_evaluation import (
    evaluate_forecast_strategies,
)
from systematic_macro_uq.config import RunConfig
from systematic_macro_uq.forecasting.pipeline import calculate_forecasts


HOLDOUT_FROZEN = True


def run_holdout_backtest(config, *, frozen):
    """Evaluates the fixed forecast mappings on the frozen holdout."""
    if not frozen:
        raise RuntimeError(
            "Holdout is locked. Set HOLDOUT_FROZEN=True only after the "
            "forecast pipeline and comparison protocol are frozen."
        )

    daily_df = pd.read_parquet(config.futures_daily_path)
    forecasts_df = calculate_forecasts(daily_df, config)
    holdout_start = pd.Timestamp(config.HOLDOUT_START, tz="UTC")
    result = evaluate_forecast_strategies(
        config=config,
        forecasts_df=forecasts_df,
        evaluation_start=holdout_start,
        evaluation_end=None,
        output_dir=config.holdout_backtest_dir,
    )
    return result.summary


if __name__ == "__main__":
    summary = run_holdout_backtest(
        RunConfig(),
        frozen=HOLDOUT_FROZEN,
    )
    print(summary.to_string(index=False))
