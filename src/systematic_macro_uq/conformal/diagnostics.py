import numpy as np

from systematic_macro_uq.conformal.application import standard_normal_radius


def compute_application_diagnostics(
    bounds,
    method,
    strategy,
    coverage,
    evaluation_start,
    evaluation_end,
    trading_days_per_year,
    local_coverage_window,
):
    """Summarizes calibration and sizing behavior over one sample."""
    sample = bounds.loc[
        bounds["date"].ge(evaluation_start)
        & bounds["date"].lt(evaluation_end)
    ]
    error_column = f"{method}_error"
    rolling_miscoverage = sample.groupby("market")[error_column].transform(
        lambda x: x.rolling(
            local_coverage_window,
            min_periods=local_coverage_window,
        ).mean()
    )
    radius = sample[f"{method}_radius"]
    valid = radius.gt(0) & np.isfinite(radius)
    multiplier = (
        standard_normal_radius(coverage)
        * sample.loc[valid, "volatility_for_position"]
        / (np.sqrt(trading_days_per_year) * radius.loc[valid])
    )
    return {
        "strategy": strategy,
        "target_coverage": coverage,
        "realized_coverage": 1 - sample[error_column].mean(),
        "rolling_coverage_mae": (
            rolling_miscoverage - (1 - coverage)
        ).abs().mean(),
        "mean_multiplier_vs_vol": multiplier.mean(),
        "median_multiplier_vs_vol": multiplier.median(),
        "infinite_radius_share": np.isinf(radius).mean(),
        "minimum_alpha": sample[f"{method}_alpha"].min(),
        "maximum_alpha": sample[f"{method}_alpha"].max(),
    }
