import pandas as pd

from systematic_macro_uq.config import RunConfig
from systematic_macro_uq.forecasting.pipeline import calculate_forecasts


def build_forecasts(config):
    df = pd.read_parquet(config.futures_daily_path)
    df = calculate_forecasts(df, config)
    df.to_parquet(config.futures_forecasts_path, index=False)
    return df


if __name__ == "__main__":
    build_forecasts(RunConfig())
