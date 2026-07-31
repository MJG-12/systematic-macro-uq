def add_combined_forecast(
    df,
    ewmac_rules,
    carry_weight,
    forecast_diversification_multiplier,
    forecast_cap,
):
    """Combines fixed weight component forecasts and applies the FDM and cap."""
    df = df.copy()

    combined_forecast = df["forecast_carry"] * carry_weight
    for fast_span, slow_span, _, weight in ewmac_rules:
        combined_forecast = (
            combined_forecast
            + df[f"forecast_ewmac_{fast_span}_{slow_span}"] * weight
        )

    df["forecast_combined"] = (
        combined_forecast
        * forecast_diversification_multiplier
    ).clip(-forecast_cap, forecast_cap)

    return df


def add_mapped_forecasts(
    df,
    average_forecast,
    inverse_quadratic_scale,
    inverse_quadratic_reference_variance,
    clipped_scale,
):
    """Adds the fixed inverse quadratic and clipped forecast mappings."""
    df = df.copy()

    scaled = df["forecast_combined"] / average_forecast
    df["forecast_inverse_quadratic"] = (
        average_forecast
        * inverse_quadratic_scale
        * scaled
        / (
            inverse_quadratic_reference_variance
            + scaled.pow(2)
        )
    )
    turning_point = inverse_quadratic_reference_variance**0.5
    df["forecast_clipped"] = (
        average_forecast
        * clipped_scale
        * scaled.clip(-turning_point, turning_point)
    )

    return df
