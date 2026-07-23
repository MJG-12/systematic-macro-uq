def add_carry_forecast(
    df,
    scalar,
    forecast_cap,
    update_weekday,
):
    df = df.sort_values(["market", "date"]).copy()

    scaled_forecast = (
        df["carry_annualized_price"]
        / df["annualized_price_vol"]
        * scalar
    )
    weekly_forecast = scaled_forecast.where(
        df["date"].dt.weekday.eq(update_weekday)
    )
    df["forecast_carry"] = (
        weekly_forecast.groupby(df["market"])
        .ffill()
        .clip(-forecast_cap, forecast_cap)
    )

    return df
