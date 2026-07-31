import pandas as pd

from systematic_macro_uq.backtest.positions import (
    construct_forecast_positions,
)
from systematic_macro_uq.conformal.aci import (
    add_per_market_conformal_radii,
)
from systematic_macro_uq.conformal.application import (
    construct_conformal_positions,
)
from systematic_macro_uq.conformal.diagnostics import (
    compute_application_diagnostics,
)


CONFORMAL_METHODS = ("fixed", "aci")


def coverage_label(coverage):
    """Formats a coverage level for use in strategy names."""
    return f"{100 * coverage:g}".replace(".", "p")


def validate_coverages(coverages):
    """Validates and deduplicates requested coverage levels."""
    values = tuple(dict.fromkeys(float(value) for value in coverages))
    if not values or any(value <= 0.5 or value >= 1 for value in values):
        raise ValueError("Coverages must be unique values between 0.5 and 1.")
    return values


def add_conformal_radii(baseline_positions, coverage, config):
    """Adds fixed conformal and ACI radii at one coverage level."""
    result = add_per_market_conformal_radii(
        baseline_positions,
        target_coverage=coverage,
        calibration_window=config.conformal.calibration_window,
        learning_rate=0.0,
        prefix="fixed",
    )
    return add_per_market_conformal_radii(
        result,
        target_coverage=coverage,
        calibration_window=config.conformal.calibration_window,
        learning_rate=config.conformal.aci_learning_rate,
        prefix="aci",
    )


def build_conformal_strategies(
    forecasts,
    coverages,
    config,
    evaluation_start,
    evaluation_end,
):
    """Constructs baseline, fixed conformal, and ACI position panels."""
    coverages = validate_coverages(coverages)
    baseline = construct_forecast_positions(
        forecasts,
        instrument_weights=config.portfolio.instrument_weights,
        instrument_diversification_multiplier=(
            config.portfolio.instrument_diversification_multiplier
        ),
        volatility_target=config.portfolio.volatility_target,
        average_forecast=config.portfolio.average_forecast,
    )
    strategies = {"baseline": baseline}
    bound_frames = []
    application_rows = []

    for coverage in coverages:
        bounds = add_conformal_radii(baseline, coverage, config)
        bounds["target_coverage"] = coverage
        bound_frames.append(
            bounds[
                [
                    "date",
                    "market",
                    "target_coverage",
                    "held_daily_return",
                    "fixed_alpha",
                    "fixed_radius",
                    "fixed_error",
                    "aci_alpha",
                    "aci_radius",
                    "aci_error",
                ]
            ]
        )

        for method in CONFORMAL_METHODS:
            strategy = f"{method}_{coverage_label(coverage)}"
            strategies[strategy] = construct_conformal_positions(
                bounds,
                radius_column=f"{method}_radius",
                volatility_target=config.portfolio.volatility_target,
                average_forecast=config.portfolio.average_forecast,
                instrument_diversification_multiplier=(
                    config.portfolio.instrument_diversification_multiplier
                ),
                trading_days_per_year=(
                    config.forecast.trading_days_per_year
                ),
                target_coverage=coverage,
            )
            application_rows.append(
                compute_application_diagnostics(
                    bounds,
                    method=method,
                    strategy=strategy,
                    coverage=coverage,
                    evaluation_start=evaluation_start,
                    evaluation_end=evaluation_end,
                    trading_days_per_year=(
                        config.forecast.trading_days_per_year
                    ),
                    local_coverage_window=(
                        config.conformal.calibration_window
                    ),
                )
            )

    return (
        strategies,
        pd.concat(bound_frames, ignore_index=True),
        pd.DataFrame(application_rows),
    )
