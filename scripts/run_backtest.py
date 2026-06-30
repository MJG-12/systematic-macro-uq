import pandas as pd
from systematic_macro_uq.backtest.positions import (construct_equal_weight_positions,
                                                    construct_vol_scaled_positions,
                                                    aggregate_portfolio)
from systematic_macro_uq.backtest.metrics import compute_performance_metrics
from systematic_macro_uq.config import RunConfig


def run_backtest(config, signals, vol_scaled=True):

    signals_df = pd.read_parquet(config.futures_signals_path)
    # portfolio_returns = []
    positions = []
    equity_curves = []
    rows = []
    
    for signal in signals:
        signal_col = f"signal_{signal}"
        
        if vol_scaled:
            strategy_name = f"{signal}_vol_scaled"
            position_df =  construct_vol_scaled_positions(signals_df, 
                                                           signal_col, 
                                                           vol_lookback=63)
        else:
            strategy_name = f"{signal}_equal_weight"
            position_df = construct_equal_weight_positions(signals_df, 
                                                            signal_col, 
                                                            len(config.markets))

        portfolio_return_col = f"portfolio_return_{strategy_name}"
        equity_curve_col = f"equity_curve_{strategy_name}"
    
        backtest_result = aggregate_portfolio(position_df, 
                        strategy_name
                        )
        portfolio_return = backtest_result[portfolio_return_col]
        equity_curve = backtest_result[equity_curve_col]
        metrics = compute_performance_metrics(portfolio_return, equity_curve)
        metrics["strategy"] = strategy_name

        position_df = position_df.copy()
        position_df["strategy"] = strategy_name
        positions.append(position_df)
        equity_curves.append(equity_curve)
        rows.append(metrics)
    
    positions_df = pd.concat(positions, ignore_index=True)
    equity_curves_df = pd.concat(equity_curves, axis=1)
    summary = pd.DataFrame(rows)
    summary = summary[["strategy", "return", "vol", "sharpe", "max_drawdown", "hit_rate"]]
        
    config.backtest_dir.mkdir(parents=True, exist_ok=True)
    positions_df.to_csv(config.backtest_dir / "positions.csv", index=False)
    equity_curves_df.to_csv(config.backtest_dir / "equity_curves.csv", index=False)
    summary.to_csv(config.backtest_summary_path, index=False)

    positions_df.to_parquet(config.backtest_dir / "positions.parquet")
    equity_curves_df.to_parquet(config.backtest_dir / "equity_curves.parquet")

    return 

if __name__ == "__main__":
    signals = ["trend_63d", "trend_126d", "trend_252d", "carry"]
    run_backtest(RunConfig(), signals, vol_scaled=True)
