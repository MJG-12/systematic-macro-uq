# import pandas as pd
import numpy as np


def create_carry_signals(df):
    df = df.sort_values(["market", "date"]).copy()
    df["signal_carry"] = np.sign(df["held_carry_annualized"])
    df["signal_carry"] = df.groupby("market")["signal_carry"].shift(1)
    return df