import pandas as pd
from systematic_macro_uq.backtest.positions import (
    aggregate_portfolio,
    construct_forecast_positions,
    estimate_instrument_diversification_multiplier,
)
from systematic_macro_uq.backtest.metrics import compute_performance_metrics
from systematic_macro_uq.backtest.costs import (
    add_linear_costs,
    add_roll_aware_turnover,
)
from systematic_macro_uq.backtest.diagnostics import compute_backtest_diagnostics
from systematic_macro_uq.config import RunConfig


def run_backtest(config):

    forecasts_df = pd.read_parquet(config.futures_forecasts_path)
    holdout_start = pd.Timestamp(config.HOLDOUT_START, tz="UTC")
    forecasts_df = forecasts_df.loc[
        forecasts_df["date"] < holdout_start
    ].copy()

    instrument_diversification_multiplier = (
        estimate_instrument_diversification_multiplier(
            forecasts_df,
            config.portfolio.instrument_weights,
        )
    )

    strategy_name = "combined_forecast"
    position_df = construct_forecast_positions(
        forecasts_df,
        instrument_weights=config.portfolio.instrument_weights,
        instrument_diversification_multiplier=(
            instrument_diversification_multiplier
        ),
        volatility_target=config.portfolio.volatility_target,
        average_forecast=config.portfolio.average_forecast,
    )
    position_df["strategy"] = strategy_name
    position_df = add_roll_aware_turnover(position_df)

    portfolio_results = []
    rows = []

    for cost_bps in config.transaction_costs:
        position_df = add_linear_costs(position_df, cost_bps)
        backtest_result = aggregate_portfolio(position_df, cost_bps)
        backtest_result["strategy"] = strategy_name
        backtest_result["cost_bps"] = cost_bps

        portfolio_return = backtest_result["net_return"]
        equity_curve = backtest_result["equity_curve"]
        metrics = compute_performance_metrics(
            portfolio_return,
            equity_curve,
            trading_days_per_year=config.forecast.trading_days_per_year,
        )
        metrics["strategy"] = strategy_name
        metrics["cost_bps"] = cost_bps

        portfolio_result = backtest_result.reset_index()
        portfolio_result = portfolio_result[
            [
                "date",
                "strategy",
                "cost_bps",
                "gross_return",
                "cost",
                "net_return",
                "equity_curve",
            ]
        ]

        portfolio_results.append(portfolio_result)
        rows.append(metrics)

    portfolio_results_df = pd.concat(portfolio_results, ignore_index=True)
    diagnostics = compute_backtest_diagnostics(
        position_df,
        portfolio_results_df,
    )
    
    summary = pd.DataFrame(rows)
    summary = summary[["strategy", "cost_bps", "return", "vol", "sharpe", "max_drawdown", "hit_rate"]]
        
    config.backtest_dir.mkdir(parents=True, exist_ok=True)
    
    diagnostics.to_csv(config.backtest_dir / "diagnostics.csv", index=False)
    summary.to_csv(config.backtest_summary_path, index=False)

    portfolio_results_df.to_parquet(config.backtest_dir / "portfolio_results.parquet", index=False)
    position_df.to_parquet(config.backtest_dir / "positions.parquet", index=False)
    return summary
    

if __name__ == "__main__":
    run_backtest(RunConfig())
