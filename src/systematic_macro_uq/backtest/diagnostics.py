import pandas as pd


def compute_backtest_diagnostics(positions_df, portfolio_results_df):

    positions_df = positions_df.copy()
    positions_df["abs_position"] = positions_df["position"].abs()
    positions_df["active_market"] = positions_df["abs_position"]> 0

    daily_position_diagnostics = (
        positions_df
        .groupby(["date", "strategy"])
        .agg(
            turnover=("turnover", "sum"),
            gross_exposure=("abs_position", "sum"),
            active_markets=("active_market", "sum"),
        )
        .reset_index()
    )

    strategy_diagnostics = (
        daily_position_diagnostics
        .groupby("strategy")
        .agg(
            average_turnover=("turnover", "mean"),
            average_gross_exposure=("gross_exposure", "mean"),
            max_gross_exposure=("gross_exposure", "max"),
            average_active_markets=("active_markets", "mean"),
            min_active_markets=("active_markets", "min"),
            max_active_markets=("active_markets", "max"),
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
            "average_cost",
            "average_gross_exposure",
            "max_gross_exposure",
            "average_active_markets",
            "min_active_markets",
            "max_active_markets",
            "average_gross_return",
            "average_net_return",
        ]
    ]