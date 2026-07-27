import pandas as pd

from systematic_macro_uq.config import RunConfig
from systematic_macro_uq.forecasting.diagnostics import (
    compute_forecast_bins,
)
from systematic_macro_uq.forecasting.pipeline import calculate_forecasts


def run_holdout_forecast_bins(config):
    """Apply the fixed development bin diagnostic to holdout observations."""
    daily_df = pd.read_parquet(config.futures_daily_path)
    df = calculate_forecasts(daily_df, config)
    holdout_start = pd.Timestamp(config.HOLDOUT_START, tz="UTC")
    holdout_df = df.loc[df["date"].ge(holdout_start)].copy()
    bins = compute_forecast_bins(holdout_df, config)

    config.holdout_backtest_dir.mkdir(parents=True, exist_ok=True)
    bins.to_csv(
        config.holdout_backtest_dir / "forecast_bins.csv",
        index=False,
    )
    return bins


if __name__ == "__main__":
    result = run_holdout_forecast_bins(RunConfig())
    print(result.to_string(index=False))
