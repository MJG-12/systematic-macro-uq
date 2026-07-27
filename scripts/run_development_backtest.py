import pandas as pd

from systematic_macro_uq.backtest.forecast_evaluation import (
    evaluate_forecast_strategies,
)
from systematic_macro_uq.config import RunConfig
from systematic_macro_uq.forecasting.pipeline import calculate_forecasts


def run_development_backtest(config):
    holdout_start = pd.Timestamp(config.HOLDOUT_START, tz="UTC")
    daily_df = pd.read_parquet(
        config.futures_daily_path,
        filters=[("date", "<", holdout_start)],
    )
    forecasts_df = calculate_forecasts(daily_df, config)
    evaluation_start = pd.Timestamp(config.DEVELOPMENT_START, tz="UTC")

    result = evaluate_forecast_strategies(
        config=config,
        forecasts_df=forecasts_df,
        evaluation_start=evaluation_start,
        evaluation_end=holdout_start,
        output_dir=config.development_backtest_dir,
    )
    return result.summary


if __name__ == "__main__":
    result = run_development_backtest(RunConfig())
    print(result.to_string(index=False))
