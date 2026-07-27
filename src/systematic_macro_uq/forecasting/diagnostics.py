import numpy as np
import pandas as pd

from systematic_macro_uq.forecasting.features import add_forward_return_target


FORECAST_BIN_EDGES = [0, 5, 10, 15, 20]


def compute_forecast_bins(df, config):
    """Tabulate signed subsequent returns in fixed forecast-strength bins."""
    df = add_forward_return_target(
        df,
        horizon_days=config.forecast.target_horizon,
        trading_days_per_year=config.forecast.trading_days_per_year,
        target_column=config.forecast.target_column,
    )
    df = df.dropna(
        subset=["forecast_combined", config.forecast.target_column]
    )
    df["signed_target"] = (
        np.sign(df["forecast_combined"])
        * df[config.forecast.target_column]
    )
    df["forecast_bin"] = pd.cut(
        df["forecast_combined"].abs(),
        bins=FORECAST_BIN_EDGES,
        include_lowest=True,
    )
    return (
        df.groupby("forecast_bin", observed=True)
        .agg(
            n_observations=("signed_target", "size"),
            mean_signed_target=("signed_target", "mean"),
            median_signed_target=("signed_target", "median"),
        )
        .reset_index()
    )
