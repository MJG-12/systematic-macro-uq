def add_carry_forecast(
    df,
    scalar,
    forecast_cap,
    smoothing_com,
):
    """Add the daily smoothed-carry forecast from the companion system."""
    df = df.sort_values(["market", "date"]).copy()

    raw_carry = (
        df["carry_annualized_price"]
        / df["annualized_price_vol"]
    )
    smoothed_carry = raw_carry.groupby(df["market"]).transform(
        lambda x: x.ewm(com=smoothing_com).mean()
    )
    df["forecast_carry"] = (
        (smoothed_carry * scalar).clip(-forecast_cap, forecast_cap)
    )

    return df
