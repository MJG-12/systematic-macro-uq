import pandas as pd


def compute_backtest_diagnostics(positions_df, portfolio_results_df):
    """Summarizes turnover, costs, returns, and portfolio exposure."""

    positions_df = positions_df.copy()
    positions_df["abs_position"] = positions_df["position"].abs()

    daily_position_diagnostics = (
        positions_df
        .groupby(["date", "strategy"])
        .agg(
            turnover=("turnover", "sum"),
            ordinary_turnover=("ordinary_turnover", "sum"),
            roll_turnover=("roll_turnover", "sum"),
            roll_trades=("roll_trade", "sum"),
            gross_exposure=("abs_position", "sum"),
            max_market_exposure=("abs_position", "max"),
        )
        .reset_index()
    )

    strategy_diagnostics = (
        daily_position_diagnostics
        .groupby("strategy")
        .agg(
            average_turnover=("turnover", "mean"),
            average_ordinary_turnover=("ordinary_turnover", "mean"),
            average_roll_turnover=("roll_turnover", "mean"),
            total_roll_trades=("roll_trades", "sum"),
            average_gross_exposure=("gross_exposure", "mean"),
            median_gross_exposure=("gross_exposure", "median"),
            p95_gross_exposure=("gross_exposure", lambda x: x.quantile(0.95)),
            max_gross_exposure=("gross_exposure", "max"),
            max_market_exposure=("max_market_exposure", "max"),
        )
        .reset_index()
    )

    cost_diagnostics = (
        portfolio_results_df.groupby(["strategy", "cost_bps"])
        .agg(
            average_cost=("cost", "mean"),
            average_gross_return=("gross_return", "mean"),
            average_net_return=("net_return", "mean"),
        )
        .reset_index()
    )

    diagnostics = pd.merge(
        cost_diagnostics,
        strategy_diagnostics,
        on="strategy",
        how="left",
    )

    return diagnostics[
        [
            "strategy",
            "cost_bps",
            "average_turnover",
            "average_ordinary_turnover",
            "average_roll_turnover",
            "total_roll_trades",
            "average_cost",
            "average_gross_exposure",
            "median_gross_exposure",
            "p95_gross_exposure",
            "max_gross_exposure",
            "max_market_exposure",
            "average_gross_return",
            "average_net_return",
        ]
    ]


def compute_return_attribution(
    positions_df,
    asset_classes,
    trading_days_per_year,
):
    """Computes annualized gross return attribution by market and year."""
    df = positions_df.copy()
    df["asset_class"] = df["market"].map(asset_classes)
    df["year"] = df["date"].dt.year

    def _annualized(group_columns):
        """Annualizes return contributions for the requested groups."""
        grouped = df.groupby(group_columns)
        n_dates = grouped["date"].nunique()
        contribution = grouped["return_contribution"].sum()
        return (
            (contribution / n_dates * trading_days_per_year)
            .rename("annualized_contribution")
            .reset_index()
        )

    by_market = _annualized(["strategy", "asset_class", "market"])
    by_year = _annualized(["strategy", "asset_class", "year"])
    return by_market, by_year


def compute_asset_class_exposures(positions_df, asset_classes):
    """Summarizes signed and gross exposure by strategy and asset class."""
    positions_df = positions_df.copy()
    positions_df["asset_class"] = positions_df["market"].map(asset_classes)
    if positions_df["asset_class"].isna().any():
        missing = positions_df.loc[
            positions_df["asset_class"].isna(), "market"
        ].unique()
        raise ValueError(
            f"Missing asset classes for: {', '.join(sorted(missing))}"
        )

    daily = (
        positions_df.assign(
            gross_exposure=positions_df["position"].abs(),
            net_exposure=positions_df["position"],
        )
        .groupby(["date", "strategy", "asset_class"])
        [["gross_exposure", "net_exposure"]]
        .sum()
        .reset_index()
    )
    return (
        daily.groupby(["strategy", "asset_class"])
        .agg(
            average_gross_exposure=("gross_exposure", "mean"),
            max_gross_exposure=("gross_exposure", "max"),
            average_net_exposure=("net_exposure", "mean"),
            minimum_net_exposure=("net_exposure", "min"),
            maximum_net_exposure=("net_exposure", "max"),
        )
        .reset_index()
    )
