import pandas as pd


def compute_backtest_diagnostics(positions_df, portfolio_results_df):

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
