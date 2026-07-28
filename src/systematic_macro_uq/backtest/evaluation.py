from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from systematic_macro_uq.backtest.costs import (
    add_linear_costs,
    add_roll_aware_turnover,
)
from systematic_macro_uq.backtest.diagnostics import (
    compute_asset_class_exposures,
    compute_backtest_diagnostics,
    compute_return_attribution,
)
from systematic_macro_uq.backtest.metrics import compute_performance_metrics
from systematic_macro_uq.backtest.positions import aggregate_portfolio


@dataclass
class BacktestEvaluation:
    summary: pd.DataFrame
    diagnostics: pd.DataFrame
    asset_class_exposures: pd.DataFrame
    attribution_by_market: pd.DataFrame
    attribution_by_year: pd.DataFrame
    positions: pd.DataFrame
    portfolio: pd.DataFrame


def select_evaluation_period(df, start, end=None):
    """Select an inclusive start and optional exclusive end date."""
    mask = df["date"].ge(start)
    if end is not None:
        mask &= df["date"].lt(end)
    return df.loc[mask].copy()


def _write_evaluation(result, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.summary.to_csv(output_dir / "summary.csv", index=False)
    result.diagnostics.to_csv(output_dir / "diagnostics.csv", index=False)
    result.asset_class_exposures.to_csv(
        output_dir / "asset_class_exposures.csv",
        index=False,
    )
    result.attribution_by_market.to_csv(
        output_dir / "attribution_by_market.csv",
        index=False,
    )
    result.attribution_by_year.to_csv(
        output_dir / "attribution_by_year.csv",
        index=False,
    )
    result.positions.to_parquet(
        output_dir / "positions.parquet",
        index=False,
    )
    result.portfolio.to_parquet(
        output_dir / "portfolio_results.parquet",
        index=False,
    )


def evaluate_position_strategies(
    config,
    strategies,
    evaluation_start,
    evaluation_end,
    output_dir=None,
):
    """Evaluate named, preconstructed position panels over one sample."""
    position_frames = []
    portfolio_frames = []
    metric_rows = []

    for strategy, raw_positions in strategies.items():
        positions = add_roll_aware_turnover(raw_positions)
        positions = select_evaluation_period(
            positions,
            start=evaluation_start,
            end=evaluation_end,
        )
        positions["strategy"] = strategy

        for cost_bps in config.transaction_costs:
            positions = add_linear_costs(positions, cost_bps)
            portfolio = aggregate_portfolio(positions, cost_bps)
            portfolio["strategy"] = strategy
            portfolio["cost_bps"] = cost_bps

            metrics = compute_performance_metrics(
                portfolio["net_return"],
                portfolio["equity_curve"],
                trading_days_per_year=(
                    config.forecast.trading_days_per_year
                ),
            )
            metrics["strategy"] = strategy
            metrics["cost_bps"] = cost_bps
            metric_rows.append(metrics)
            portfolio_frames.append(portfolio.reset_index())

        position_frames.append(positions)

    positions = pd.concat(position_frames, ignore_index=True)
    portfolio = pd.concat(portfolio_frames, ignore_index=True)
    summary = pd.DataFrame(metric_rows)[
        [
            "strategy",
            "cost_bps",
            "return",
            "vol",
            "sharpe",
            "max_drawdown",
            "hit_rate",
        ]
    ]
    attribution_by_market, attribution_by_year = compute_return_attribution(
        positions,
        config.portfolio.asset_classes,
        config.forecast.trading_days_per_year,
    )
    result = BacktestEvaluation(
        summary=summary,
        diagnostics=compute_backtest_diagnostics(positions, portfolio),
        asset_class_exposures=compute_asset_class_exposures(
            positions,
            config.portfolio.asset_classes,
        ),
        attribution_by_market=attribution_by_market,
        attribution_by_year=attribution_by_year,
        positions=positions,
        portfolio=portfolio,
    )
    if output_dir is not None:
        _write_evaluation(result, output_dir)
    return result
