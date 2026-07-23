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
