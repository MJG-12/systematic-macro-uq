import math

import numpy as np


def _conformal_radius(scores, miscoverage):
    """Returns the finite sample quantile for a conformal radius."""
    rank = math.ceil((len(scores) + 1) * (1 - miscoverage))
    if rank > len(scores):
        return np.inf
    if rank <= 0:
        return 0.0
    return np.partition(scores, rank - 1)[rank - 1]


def add_per_market_conformal_radii(
    positions,
    target_coverage,
    calibration_window,
    learning_rate,
    prefix,
):
    """Calibrates symmetric conformal intervals for next day returns.

    Each row's radius is formed from earlier absolute returns. The return on
    that row is observed only after the bound has been recorded and therefore
    can update the ACI level only for subsequent positions.
    """
    result = (
        positions.sort_values(["market", "date"])
        .reset_index(drop=True)
        .copy()
    )
    alpha_column = f"{prefix}_alpha"
    radius_column = f"{prefix}_radius"
    error_column = f"{prefix}_error"
    result[alpha_column] = np.nan
    result[radius_column] = np.nan
    result[error_column] = np.nan

    target_miscoverage = 1 - target_coverage
    for _, indexes in result.groupby("market", sort=False).groups.items():
        alpha = target_miscoverage
        completed_scores = []

        for index in indexes:
            if len(completed_scores) >= calibration_window:
                radius = _conformal_radius(
                    np.asarray(completed_scores[-calibration_window:]),
                    alpha,
                )
                result.at[index, alpha_column] = alpha
                result.at[index, radius_column] = radius

            realized_return = result.at[index, "held_daily_return"]
            if np.isfinite(realized_return):
                radius = result.at[index, radius_column]
                if not np.isnan(radius):
                    error = float(abs(realized_return) > radius)
                    result.at[index, error_column] = error
                    alpha = alpha + learning_rate * (
                        target_miscoverage - error
                    )
                completed_scores.append(abs(realized_return))

    return result
