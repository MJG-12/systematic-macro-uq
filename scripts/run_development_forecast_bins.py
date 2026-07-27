import pandas as pd

from systematic_macro_uq.config import RunConfig
from systematic_macro_uq.forecasting.diagnostics import (
    compute_forecast_bins,
)
from systematic_macro_uq.forecasting.pipeline import calculate_forecasts


def run_development_forecast_bins(config):
    """Tabulate development returns by forecast strength."""
    holdout_start = pd.Timestamp(config.HOLDOUT_START, tz="UTC")
    daily_df = pd.read_parquet(
        config.futures_daily_path,
        filters=[("date", "<", holdout_start)],
    )
    df = calculate_forecasts(daily_df, config)
    evaluation_start = pd.Timestamp(config.DEVELOPMENT_START, tz="UTC")
    development_df = df.loc[df["date"].ge(evaluation_start)].copy()
    bins = compute_forecast_bins(development_df, config)

    config.development_backtest_dir.mkdir(parents=True, exist_ok=True)
    bins.to_csv(
        config.development_backtest_dir / "forecast_bins.csv",
        index=False,
    )
    return bins


if __name__ == "__main__":
    result = run_development_forecast_bins(RunConfig())
    print(result.to_string(index=False))
