import os
import pandas as pd
import numpy as np
import databento as db
from dotenv import load_dotenv
from functools import cache


@cache
def get_databento_client():
    load_dotenv()
    api_key = os.environ["DATABENTO_API_KEY"]
    return db.Historical(api_key)


def fetch_futures_data(schema, symbol, start_date, end_date):
    """Returns Databento raw futures data. 
    """
    client = get_databento_client()

    df = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        schema=f"{schema}",
        symbols=f"{symbol}.FUT",
        stype_in="parent",
        start=start_date,
        end=end_date,
    ).to_df()
    df["market"] = symbol
    return df


def build_futures_metadata(raw_df):
    """Cleans Databento raw futures metadata by removing unused columns
    contracts.
    """
    meta_cols = [
        "market",
        "ts_event",
        "instrument_id",
        "raw_symbol",
        "security_update_action",
        "instrument_class",
        "activation",
        "expiration",
        "maturity_year",
        "maturity_month"
    ]
    df = raw_df[meta_cols].copy().reset_index()
    df = df.rename(columns = {"ts_recv" : "ts_recv_def"})
    return df.sort_values(["ts_recv_def"])


def build_daily_statistic_data(raw_df, statistic="settlement_price"):
    
    daily_statistics = {
        "settlement_price": (3, "price"),
        "volume": (6, "quantity"),
    }

    if statistic not in daily_statistics:
        raise ValueError("statistic must be settlement_price or volume.")

    stat_int, old_col = daily_statistics[statistic]
        
    cols = [
        "market",
        "symbol", 
        "price",
        "quantity",
        "stat_type",
        "ts_ref",
        "instrument_id",
        "update_action"
    ]
    df = raw_df[cols].copy().reset_index()
    renamed_columns = {"ts_recv" : "ts_recv_stat",
                        "symbol" : "symbol_stat",
                        old_col : statistic}
    df = df.rename(columns=renamed_columns)

    df = df[df["stat_type"] == stat_int]
    df = df.dropna(subset=["ts_ref", statistic])
    df["date"] = df["ts_ref"].dt.normalize()

    df = (
    df.sort_values("ts_recv_stat")
      .drop_duplicates(["date", "instrument_id"], keep="last")
      .sort_values(["ts_ref", "instrument_id"])
    )

    return df


def compute_contract_returns(df):
    df = df.sort_values(["instrument_id", "date"]).copy()
    df["contract_return"] = (
        df.groupby(["instrument_id"])["settlement_price"]
          .pct_change()
    )
    return df[["date", "instrument_id", "contract_return"]]


def build_roll_schedule(universe_df, prices_df, volumes_df, min_days_to_expiry):

    prices_and_volumes = pd.merge(prices_df, 
                                  volumes_df[["date", "instrument_id", "volume"]],
                                  on = ["date", "instrument_id"],
                                  how = "left",
    ).sort_values("ts_ref")

    df  =  pd.merge_asof(
        prices_and_volumes,
        universe_df,
        left_on="ts_ref",
        right_on="ts_recv_def",
        by=["market", "instrument_id"],
        direction="backward"
    )

    df = df[df["instrument_class"] == "F"]
    assert df["market"].nunique() == 1

    df["live"] = df["date"].between(df["activation"].dt.normalize(), df["expiration"].dt.normalize(), inclusive="left")
    df = df.loc[df["live"]]
    df["days_to_expiry"] = (df["expiration"].dt.normalize() - df["date"]).dt.days

    eligible = df.loc[(df["days_to_expiry"] > min_days_to_expiry)].copy()
    eligible = eligible.sort_values(["ts_ref", "expiration", "instrument_id"])
    eligible ["rank"] = eligible.groupby("date").cumcount() + 1
    eligible["volume"] = eligible["volume"].fillna(0)
    
    front = eligible.loc[eligible["rank"]==1, ["date", "volume"]].rename(
    columns={"volume": "front_volume"})
    next_ = eligible.loc[eligible["rank"]==2, ["date", "volume"]].rename(
    columns={"volume": "next_volume"})

    decision = pd.merge(front, next_, on="date", how="left")
    decision["selected_rank"] = np.where(
        decision["next_volume"] > decision["front_volume"],
        2,
        1,
    )

    eligible = pd.merge(
        eligible,
        decision[["date", "selected_rank"]],
        on="date",
        how="left",
    )

    schedule = eligible.loc[eligible["rank"] == eligible["selected_rank"]].drop(columns=["selected_rank"]).sort_values("date")
    schedule = schedule.rename(columns={"instrument_id": "scheduled_instrument_id"})
    schedule["held_instrument_id"] = schedule["scheduled_instrument_id"].shift(1)

    return schedule 


def fetch_and_build_market_return_ts(symbol, price_start, end, definition_start, min_days_to_expiry):
    universe_raw = fetch_futures_data("definition", symbol, definition_start, end)
    stats_raw = fetch_futures_data("statistics", symbol, price_start, end)

    universe_df = build_futures_metadata(universe_raw)
    prices_df = build_daily_statistic_data(stats_raw, "settlement_price")
    volumes_df = build_daily_statistic_data(stats_raw, "volume")

    schedule = build_roll_schedule(universe_df, prices_df, volumes_df, min_days_to_expiry)
    contract_returns_df = compute_contract_returns(prices_df).rename(
        columns={"instrument_id": "held_instrument_id"}
    )

    market_returns_df = pd.merge(schedule, 
                                 contract_returns_df,
                                 on = ["date", "held_instrument_id"],
                                 how = "left"
                                 )
    
    final_cols = [
        "date",
        "market",
        "return",
        "scheduled_instrument_id",
        "held_instrument_id",
        "raw_symbol",
        "settlement_price",
        "volume",
        "rank",
        "days_to_expiry",
        "expiration",
        "maturity_year",
        "maturity_month",
        "ts_ref",
        "ts_recv_stat",
        "ts_recv_def",
    ]

    return market_returns_df.rename(columns={"contract_return" : "return"})[final_cols]


def build_futures_return_dataset(markets, price_start, end, definition_start, min_days_to_expiry):
    schedules = [
        fetch_and_build_market_return_ts(m, price_start, end, definition_start, min_days_to_expiry)
        for m in markets
    ]
    return pd.concat(schedules, ignore_index=True)