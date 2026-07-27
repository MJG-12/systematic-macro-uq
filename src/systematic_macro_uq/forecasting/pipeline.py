from systematic_macro_uq.signals.carry import add_carry_forecast
from systematic_macro_uq.signals.composite import (
    add_combined_forecast,
    add_mapped_forecasts,
)
from systematic_macro_uq.signals.trend import create_ewmac_forecasts
from systematic_macro_uq.signals.vol import add_ewma_volatility


def calculate_forecasts(df, config):
    df = add_ewma_volatility(
        df,
        span=config.forecast.volatility_span,
        trading_days_per_year=config.forecast.trading_days_per_year,
        long_run_weight=config.forecast.volatility_long_run_weight,
        long_run_window_days=config.forecast.volatility_long_run_window_days,
    )
    df = create_ewmac_forecasts(
        df,
        ewmac_rules=config.forecast.ewmac_rules,
        forecast_cap=config.forecast.forecast_cap,
    )
    df = add_carry_forecast(
        df,
        scalar=config.forecast.carry_scalar,
        forecast_cap=config.forecast.forecast_cap,
        smoothing_com=config.forecast.carry_smoothing_com,
    )
    df = add_combined_forecast(
        df,
        ewmac_rules=config.forecast.ewmac_rules,
        carry_weight=config.forecast.carry_weight,
        forecast_diversification_multiplier=(
            config.forecast.forecast_diversification_multiplier
        ),
        forecast_cap=config.forecast.forecast_cap,
    )
    return add_mapped_forecasts(
        df,
        average_forecast=config.portfolio.average_forecast,
        ferson_siegel_scale=config.forecast.ferson_siegel_scale,
        clipped_scale=config.forecast.clipped_scale,
    )
