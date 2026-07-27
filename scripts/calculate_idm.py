import pandas as pd

from systematic_macro_uq.backtest.positions import (
    estimate_instrument_diversification_multiplier,
)
from systematic_macro_uq.config import RunConfig
from systematic_macro_uq.forecasting.pipeline import calculate_forecasts


def calculate_idm(config):
    estimation_end = pd.Timestamp(config.DEVELOPMENT_START, tz="UTC")
    daily_df = pd.read_parquet(
        config.futures_daily_path,
        filters=[("date", "<", estimation_end)],
    )
    forecasts_df = calculate_forecasts(daily_df, config)
    return estimate_instrument_diversification_multiplier(
        forecasts_df,
        config.portfolio.instrument_weights,
    )


if __name__ == "__main__":
    print(calculate_idm(RunConfig()))
