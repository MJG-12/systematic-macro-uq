"""Exploratory development-only directional conformal sizing experiment.

The score is strategy loss: the return oriented by the sign of the forecast
that set that day's position. An empirically unresolvable ACI bound is infinite
and therefore sets that market's position to zero. The experiment is retained
as a possible abstention hypothesis but is not evaluated after 2021. The
selected symmetric implementation remains in ``src/`` and the main
development runner.
"""

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from systematic_macro_uq.backtest.evaluation import (
    evaluate_position_strategies,
)
from systematic_macro_uq.backtest.positions import construct_forecast_positions
from systematic_macro_uq.config import RunConfig
from systematic_macro_uq.forecasting.pipeline import calculate_forecasts


def _conformal_quantile(scores, miscoverage):
    rank = math.ceil((len(scores) + 1) * (1 - miscoverage))
    if rank > len(scores):
        return np.inf
    return np.partition(scores, rank - 1)[rank - 1]


def _add_directional_bounds(
    baseline,
    coverage,
    calibration_window,
    learning_rate,
):
    """Use only completed, directionally oriented returns in each bound."""
    result = baseline.sort_values(["market", "date"]).reset_index(drop=True).copy()
    result[["alpha", "bound", "error"]] = np.nan
    target_alpha = 1 - coverage

    for _, indexes in result.groupby("market", sort=False).groups.items():
        alpha = target_alpha
        completed_scores = []
        for index in indexes:
            if len(completed_scores) >= calibration_window:
                bound = _conformal_quantile(
                    np.asarray(completed_scores[-calibration_window:]),
                    alpha,
                )
                result.at[index, "alpha"] = alpha
                result.at[index, "bound"] = bound

            realized_return = result.at[index, "held_daily_return"]
            forecast = result.at[index, "forecast_for_position"]
            if np.isfinite(realized_return) and np.isfinite(forecast):
                score = -np.sign(forecast) * realized_return
                bound = result.at[index, "bound"]
                if not np.isnan(bound):
                    error = float(score > bound)
                    result.at[index, "error"] = error
                    alpha += learning_rate * (target_alpha - error)
                completed_scores.append(score)
    return result


def _construct_positions(bounds, bound_column, coverage, config):
    positions = bounds.copy()
    reference_quantile = NormalDist().inv_cdf(coverage)
    positions["conformal_annualized_risk"] = (
        np.sqrt(config.forecast.trading_days_per_year)
        * positions[bound_column]
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
        * config.portfolio.instrument_diversification_multiplier
        * config.portfolio.volatility_target
        / positions.loc[valid, "conformal_annualized_risk"]
        * positions.loc[valid, "forecast_for_position"]
        / config.portfolio.average_forecast
    )
    positions["return_contribution"] = (
        positions["position"] * positions["held_daily_return"].fillna(0)
    )
    return positions


def _application_diagnostics(bounds, strategy, coverage, start, end):
    sample = bounds.loc[bounds["date"].ge(start) & bounds["date"].lt(end)]
    bound = sample["bound"]
    infinite = np.isinf(bound)
    daily_count = infinite.groupby(sample["date"]).sum()
    return {
        "strategy": strategy,
        "target_coverage": coverage,
        "realized_coverage": 1 - sample["error"].mean(),
        "minimum_alpha": sample["alpha"].min(),
        "maximum_alpha": sample["alpha"].max(),
        "infinite_market_day_share": infinite.mean(),
        "days_with_any_abstention_share": daily_count.gt(0).mean(),
        "average_markets_abstained": daily_count.mean(),
    }


def main():
    config = RunConfig()
    holdout_start = pd.Timestamp(config.HOLDOUT_START, tz="UTC")
    evaluation_start = pd.Timestamp(config.DEVELOPMENT_START, tz="UTC")
    daily = pd.read_parquet(
        config.futures_daily_path,
        filters=[("date", "<", holdout_start)],
    )
    forecasts = calculate_forecasts(daily, config)
    baseline = construct_forecast_positions(
        forecasts,
        instrument_weights=config.portfolio.instrument_weights,
        instrument_diversification_multiplier=(
            config.portfolio.instrument_diversification_multiplier
        ),
        volatility_target=config.portfolio.volatility_target,
        average_forecast=config.portfolio.average_forecast,
    )
    calibration_window = config.conformal.calibration_window
    strategies = {"baseline": baseline}
    diagnostics = []

    for coverage in config.conformal.development_coverages:
        label = f"{100 * coverage:g}"
        methods = (
            ("fixed", 0.0),
            ("aci", config.conformal.aci_learning_rate),
        )
        for method, learning_rate in methods:
            prefix = f"directional_{method}_{label}"
            bounds = _add_directional_bounds(
                baseline,
                coverage=coverage,
                calibration_window=calibration_window,
                learning_rate=learning_rate,
            )
            strategies[prefix] = _construct_positions(
                bounds,
                bound_column="bound",
                coverage=coverage,
                config=config,
            )
            diagnostics.append(
                _application_diagnostics(
                    bounds,
                    strategy=prefix,
                    coverage=coverage,
                    start=evaluation_start,
                    end=holdout_start,
                )
            )

    result = evaluate_position_strategies(
        config=config,
        strategies=strategies,
        evaluation_start=evaluation_start,
        evaluation_end=holdout_start,
    )
    output_dir = (
        config.conformal_backtest_dir
        / "development_directional_abstention"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result.summary.to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(
        output_dir / "application_diagnostics.csv",
        index=False,
    )

    print(
        result.summary.loc[
            result.summary["cost_bps"].eq(2)
        ].to_string(index=False)
    )
    print()
    print(pd.DataFrame(diagnostics).to_string(index=False))


if __name__ == "__main__":
    main()
