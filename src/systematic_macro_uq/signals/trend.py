def add_stitched_price(df):
    df = df.sort_values(["market", "date"]).copy()

    previous_contract_price = (
        df["held_settlement_price"] / (1 + df["held_daily_return"])
    )
    held_price_change = (
        df["held_settlement_price"] - previous_contract_price
    ).fillna(0)

    initial_price = df.groupby("market")["selected_settlement_price"].transform(
        "first"
    )
    df["stitched_price"] = (
        initial_price
        + held_price_change.groupby(df["market"]).cumsum()
    )

    return df


def add_ewmac_forecast(
    df,
    fast_span,
    slow_span,
    scalar,
    forecast_cap,
):
    df = df.sort_values(["market", "date"]).copy()
    prices = df.groupby("market")["stitched_price"]

    fast_ewma = prices.transform(
        lambda x: x.ewm(
            span=fast_span,
            adjust=False,
            min_periods=slow_span,
        ).mean()
    )
    slow_ewma = prices.transform(
        lambda x: x.ewm(
            span=slow_span,
            adjust=False,
            min_periods=slow_span,
        ).mean()
    )

    scaled_forecast = (
        (fast_ewma - slow_ewma)
        / df["daily_price_vol"]
        * scalar
    )
    df[f"forecast_ewmac_{fast_span}_{slow_span}"] = scaled_forecast.clip(
        -forecast_cap,
        forecast_cap,
    )

    return df


def create_ewmac_forecasts(
    df,
    ewmac_rules,
    forecast_cap,
):
    df = add_stitched_price(df)

    for fast_span, slow_span, scalar, _ in ewmac_rules:
        df = add_ewmac_forecast(
            df,
            fast_span=fast_span,
            slow_span=slow_span,
            scalar=scalar,
            forecast_cap=forecast_cap,
        )

    return df
