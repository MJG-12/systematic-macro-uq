from pathlib import Path
import pandas as pd
from systematic_macro_uq.signals.trend import create_trend_signals
from systematic_macro_uq.signals.carry import create_carry_signals
from systematic_macro_uq.signals.vol import create_vol_signals
from systematic_macro_uq.config  import RunConfig


def add_signals(config):

    df = pd.read_parquet(config.futures_daily_path)
    df = create_trend_signals(df, lookback_list=[21, 63, 126, 252])
    df = create_carry_signals(df)
    df = create_vol_signals(df, lookback_list =[21, 63])
    df.to_parquet(config.futures_signals_path, index=False)
    return df


if __name__ == "__main__":
    add_signals(RunConfig())
