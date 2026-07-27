from systematic_macro_uq.backtest.evaluation import (
    evaluate_position_strategies,
)
from systematic_macro_uq.backtest.positions import (
    construct_forecast_positions,
)


DEFAULT_FORECAST_STRATEGIES = {
    "linear": "forecast_combined",
    "ferson_siegel": "forecast_ferson_siegel",
    "clip10": "forecast_clip10",
}


def evaluate_forecast_strategies(
    config,
    forecasts_df,
    evaluation_start,
    evaluation_end,
    output_dir=None,
    strategy_forecast_columns=None,
    instrument_diversification_multiplier=None,
):
    """Construct and evaluate fixed forecast-to-position mappings."""
    if strategy_forecast_columns is None:
        strategy_forecast_columns = DEFAULT_FORECAST_STRATEGIES
    if instrument_diversification_multiplier is None:
        instrument_diversification_multiplier = (
            config.portfolio.instrument_diversification_multiplier
        )

    strategies = {
        strategy: construct_forecast_positions(
            forecasts_df,
            instrument_weights=config.portfolio.instrument_weights,
            instrument_diversification_multiplier=(
                instrument_diversification_multiplier
            ),
            volatility_target=config.portfolio.volatility_target,
            average_forecast=config.portfolio.average_forecast,
            forecast_column=forecast_column,
        )
        for strategy, forecast_column in strategy_forecast_columns.items()
    }
    return evaluate_position_strategies(
        config=config,
        strategies=strategies,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        output_dir=output_dir,
    )
