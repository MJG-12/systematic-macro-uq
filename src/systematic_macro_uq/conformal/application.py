from statistics import NormalDist

import numpy as np


def standard_normal_radius(target_coverage):
    """Returns the standard normal radius for a central coverage level."""
    quantile = NormalDist().inv_cdf((1 + target_coverage) / 2)
    if quantile <= 0:
        raise ValueError("Target coverage must be positive.")
    return quantile


def construct_conformal_positions(
    baseline_positions,
    radius_column,
    volatility_target,
    average_forecast,
    instrument_diversification_multiplier,
    trading_days_per_year,
    target_coverage,
):
    """Replaces forecast volatility with a conformal return scale."""
    positions = baseline_positions.copy()
    reference_quantile = standard_normal_radius(
        target_coverage
    )
    positions["conformal_annualized_risk"] = (
        np.sqrt(trading_days_per_year)
        * positions[radius_column]
        / reference_quantile
    )

    valid = (
        positions["instrument_weight"].notna()
        & positions["forecast_for_position"].notna()
        & positions["conformal_annualized_risk"].gt(0)
        & np.isfinite(positions["conformal_annualized_risk"])
    )
    positions["position"] = 0.0
    positions.loc[valid, "position"] = (
        positions.loc[valid, "instrument_weight"]
        * instrument_diversification_multiplier
        * volatility_target
        / positions.loc[valid, "conformal_annualized_risk"]
        * positions.loc[valid, "forecast_for_position"]
        / average_forecast
    )
    positions["return_contribution"] = (
        positions["position"]
        * positions["held_daily_return"].fillna(0)
    )
    return positions
