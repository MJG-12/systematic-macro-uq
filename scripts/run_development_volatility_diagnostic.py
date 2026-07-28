from dataclasses import replace

import pandas as pd

from systematic_macro_uq.backtest.forecast_evaluation import (
    evaluate_forecast_strategies,
)
from systematic_macro_uq.config import RunConfig
from systematic_macro_uq.forecasting.pipeline import calculate_forecasts


CANONICAL_COST_BPS = 2
DIAGNOSTIC_STRATEGIES = {
    "linear": "forecast_combined",
    "inverse_quadratic": "forecast_inverse_quadratic",
}


def run_development_volatility_diagnostic(config):
    """Compare short-only and blended volatility on development data."""
    holdout_start = pd.Timestamp(config.HOLDOUT_START, tz="UTC")
    daily_df = pd.read_parquet(
        config.futures_daily_path,
        filters=[("date", "<", holdout_start)],
    )

    blended_forecasts = calculate_forecasts(daily_df.copy(), config)
    naive_config = replace(
        config,
        forecast=replace(
            config.forecast,
            volatility_span=36,
            volatility_long_run_weight=0.0,
        ),
    )
    naive_forecasts = calculate_forecasts(daily_df.copy(), naive_config)

    fixed_idm = config.portfolio.instrument_diversification_multiplier
    evaluation_config = replace(
        config,
        transaction_costs=[CANONICAL_COST_BPS],
    )
    evaluation_start = pd.Timestamp(config.DEVELOPMENT_START, tz="UTC")

    results = []
    for volatility_model, forecasts_df in (
        ("span36_short_only", naive_forecasts),
        ("span32_70_30_blend", blended_forecasts),
    ):
        evaluation = evaluate_forecast_strategies(
            config=evaluation_config,
            forecasts_df=forecasts_df,
            evaluation_start=evaluation_start,
            evaluation_end=holdout_start,
            strategy_forecast_columns=DIAGNOSTIC_STRATEGIES,
            instrument_diversification_multiplier=fixed_idm,
        )
        summary = evaluation.summary
        summary.insert(0, "volatility_model", volatility_model)
        results.append(summary)

    strategy_results = pd.concat(results, ignore_index=True)
    comparison = (
        strategy_results.pivot(
            index=["volatility_model", "cost_bps"],
            columns="strategy",
            values="sharpe",
        )
        .reset_index()
        .rename_axis(columns=None)
        .rename(
            columns={
                "linear": "linear_sharpe",
                "inverse_quadratic": "inverse_quadratic_sharpe",
            }
        )
    )
    comparison["inverse_quadratic_minus_linear"] = (
        comparison["inverse_quadratic_sharpe"]
        - comparison["linear_sharpe"]
    )

    output_dir = config.development_backtest_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    strategy_results.to_csv(
        output_dir / "volatility_diagnostic_strategy_results.csv",
        index=False,
    )
    comparison.to_csv(
        output_dir / "volatility_diagnostic.csv",
        index=False,
    )
    return comparison


if __name__ == "__main__":
    result = run_development_volatility_diagnostic(RunConfig())
    print(result.to_string(index=False))
