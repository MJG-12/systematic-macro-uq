import numpy as np

def add_return_index(df):
    df = df.sort_values(["market", "date"]).copy()
    df["return_index"] = df["held_daily_return"].fillna(0) + 1
    df["return_index"] = (df.groupby("market")["return_index"]
                          .transform(lambda x: x.cumprod()))
    
    return df


def add_trailing_return(df, lookback):
    df = df.sort_values(["market", "date"]).copy()
    df[f"trend_{lookback}d"] = ((df["return_index"] /
                                df.groupby("market")["return_index"].transform(lambda x: x.shift(lookback)))
                                - 1)
    return df


def add_trend_signal(df, lookback):
    df = df.sort_values(["market", "date"]).copy()
    df[f"signal_trend_{lookback}d"] = np.sign(df[f"trend_{lookback}d"])
    df[f"signal_trend_{lookback}d"] = df.groupby("market")[f"signal_trend_{lookback}d"].shift(1)
    return df


def create_trend_signals(df, lookback_list=[63, 126, 252]):
    
    df = add_return_index(df)
    for lookback in lookback_list:
        df = add_trailing_return(df, lookback)
        df = add_trend_signal(df, lookback)

    df = add_trend_composite_signal(df, lookback_list)
    return df


def add_trend_composite_signal(df, lookback_list):

    df = df.sort_values(["market", "date"]).copy()
    trend_signals = [f"signal_trend_{lookback}d" for lookback in lookback_list]
    df["signal_trend_composite"] = df[trend_signals].mean(axis=1)
  
    return df
