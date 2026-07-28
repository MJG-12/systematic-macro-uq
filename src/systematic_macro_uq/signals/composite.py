def add_combined_forecast(
    df,
    ewmac_rules,
    carry_weight,
    forecast_diversification_multiplier,
    forecast_cap,
):
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
    clipped_scale,
):
    """Add fixed, variance-matched nonlinear forecast mappings.

    The constants preserve variance under a standard-normal input signal.
    They are fixed independently of this dataset, so no sample normalization
    or future information enters either mapping.
    """
    df = df.copy()

    scaled = df["forecast_combined"] / average_forecast
    df["forecast_inverse_quadratic"] = (
        average_forecast
        * inverse_quadratic_scale
        * scaled
        / (1 + scaled.pow(2))
    )
    df["forecast_clip10"] = (
        average_forecast
        * clipped_scale
        * scaled.clip(-1, 1)
    )

    return df
