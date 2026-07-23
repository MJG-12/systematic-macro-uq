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


def fetch_futures_data(schema, market, start_date, end_date,  symbols=None, stype_in="parent"):
    """Returns Databento raw futures data. 
    """
    client = get_databento_client()

    if symbols is None:
        symbols = f"{market}.FUT"

    df = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        schema=schema,
        symbols=symbols,
        stype_in=stype_in,
        start=start_date,
        end=end_date
    ).to_df()

    df["market"] = market
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
    df = df[df["instrument_class"] == "F"].copy()

    return df.sort_values(["ts_recv_def"])


def build_daily_statistic_data(raw_df, statistic):
    
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

    mask = (
    (raw_df["stat_type"] == stat_int)
    & raw_df["ts_ref"].notna()
    & raw_df[old_col].notna()
    )
    df = raw_df.loc[mask, cols].copy().reset_index()
    
    renamed_columns = {"ts_recv" : "ts_recv_stat",
                        "symbol" : "symbol_stat",
                        old_col : statistic}
    df = df.rename(columns=renamed_columns)

    df["date"] = df["ts_ref"].dt.normalize()
    df = df.loc[df["date"].dt.weekday < 5].copy()

    df = (
    df.sort_values("ts_recv_stat")
      .drop_duplicates(["date", "instrument_id"], keep="last")
      .sort_values(["ts_ref", "instrument_id"])
    )

    return df


def build_contract_panel(universe_df, prices_df, volumes_df):

    panel_cols = [
        "date",
        "market",
        "instrument_id",
        "raw_symbol",
        "settlement_price",
        "volume",
        "activation",
        "expiration",
        "days_to_expiry",
        "ts_ref",
        "ts_recv_stat",
        "ts_recv_def",
    ]

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

    assert df["market"].nunique() == 1
    assert universe_df["instrument_class"].eq("F").all()

    df["live"] = df["date"].between(df["activation"].dt.normalize(), df["expiration"].dt.normalize(), inclusive="left")
    df["volume"] = df["volume"].fillna(0)
    df["days_to_expiry"] = (
        df["expiration"].dt.normalize() - df["date"]
    ).dt.days
    contract_panel = df.loc[df["live"], panel_cols]
    
    return contract_panel


def build_contract_returns_data(contract_panel):

    df = contract_panel.sort_values(["instrument_id", "date"]).copy()
    df["contract_return"] = (
        df.groupby(["instrument_id"])["settlement_price"]
          .pct_change()
    )
    df = df.rename(columns = {"instrument_id" : "held_instrument_id",
                              "contract_return" : "held_daily_return",
                              "settlement_price" : "held_settlement_price"
                              })
    return df[[
        "date",
        "held_instrument_id",
        "held_daily_return",
        "held_settlement_price",
    ]]


def build_roll_schedule(contract_panel, min_days_to_expiry):

    schedule_cols = [
        "date",
        "market",
        "selected_instrument_id",
        "selected_raw_symbol",
        "selected_settlement_price",
        "selected_volume",
        "selected_days_to_expiry",
        "selected_expiration",
        "held_instrument_id",
        "held_raw_symbol",
        "rolled_today",
    ]
     
    df = contract_panel.copy()
   
    eligible = df.loc[(df["days_to_expiry"] > min_days_to_expiry)].copy()
    eligible = eligible.sort_values(["ts_ref", "expiration", "instrument_id"])
    eligible ["rank"] = eligible.groupby("date").cumcount() + 1
    
    front = eligible.loc[eligible["rank"]==1, ["date", "volume"]].rename(
    columns={"volume": "front_volume"})
    next_ = eligible.loc[eligible["rank"]==2, ["date", "volume"]].rename(
    columns={"volume": "next_volume"})

    decision = pd.merge(front, next_, on="date", how="left").sort_values("date")
    decision["liquidity_roll"] = decision["next_volume"] > decision["front_volume"]
    decision["confirmed_liquidity_roll"] = (
        decision["liquidity_roll"]
        & decision["liquidity_roll"].shift(1).fillna(False)
    )
    decision["selected_rank"] = np.where(decision["confirmed_liquidity_roll"],
                                         2,
                                         1)
    eligible = pd.merge(
        eligible,
        decision[["date", "selected_rank"]],
        on="date",
        how="left",
    )

    candidate = (
        eligible.loc[eligible["rank"] == eligible["selected_rank"]]
        .sort_values("date")
        .copy()
    )
    candidate["min_selected_expiration"] = candidate["expiration"].cummax()

    eligible = pd.merge(eligible,
                        candidate[["date", "min_selected_expiration"]],
                        on="date",
                        how="left")

    eligible = eligible.loc[eligible["expiration"] >= eligible["min_selected_expiration"]]

    schedule = (eligible
                .sort_values(["date", "expiration", "instrument_id"])
                .drop_duplicates("date", keep="first")
    )
    
    schedule = schedule.rename(columns={"instrument_id": "selected_instrument_id",
                                        "raw_symbol" : "selected_raw_symbol",
                                        "settlement_price" : "selected_settlement_price",
                                        "volume" : "selected_volume",
                                        "expiration" : "selected_expiration",
                                        "days_to_expiry": "selected_days_to_expiry"})
                                  
    schedule["held_instrument_id"] = schedule["selected_instrument_id"].shift(1)
    schedule["held_raw_symbol"] = schedule["selected_raw_symbol"].shift(1)
    schedule["rolled_today"] = (schedule["selected_instrument_id"] != schedule["held_instrument_id"]) & (~schedule["held_instrument_id"].isna())
    
    return schedule[schedule_cols]


def build_carry_data(panel, roll_schedule):

    carry_cols = [
        "date",
        "market",
        "selected_instrument_id",
        "carry_price_difference",
        "carry_day_gap",
        "carry_annualized_price",
    ]
    curve = panel.sort_values(["market", "date", "expiration", "instrument_id"]).copy()

    g = curve.groupby(["market", "date"], sort=False)
    curve["nearer_settlement_price"] = g["settlement_price"].shift(1)
    curve["nearer_expiration"] = g["expiration"].shift(1)
    curve["next_settlement_price"] = g["settlement_price"].shift(-1)
    curve["next_expiration"] = g["expiration"].shift(-1)

    schedule = roll_schedule[
        ["date", "market", "selected_instrument_id"]
    ].copy()

    carry_data = pd.merge(
        schedule,
        curve,
        left_on=["date", "market", "selected_instrument_id"],
        right_on=["date", "market", "instrument_id"],
        how="left",
    )

    selected_is_nearest = carry_data["nearer_expiration"].isna()
    near_price = carry_data["nearer_settlement_price"].where(
        ~selected_is_nearest,
        carry_data["settlement_price"],
    )
    far_price = carry_data["settlement_price"].where(
        ~selected_is_nearest,
        carry_data["next_settlement_price"],
    )
    near_expiration = carry_data["nearer_expiration"].where(
        ~selected_is_nearest,
        carry_data["expiration"],
    )
    far_expiration = carry_data["expiration"].where(
        ~selected_is_nearest,
        carry_data["next_expiration"],
    )

    carry_data["carry_price_difference"] = near_price - far_price
    carry_data["carry_day_gap"] = (
        far_expiration - near_expiration
    ).dt.days

    valid = (
        carry_data["carry_price_difference"].notna()
        & carry_data["carry_day_gap"].gt(0)
    )

    carry_data["carry_annualized_price"] = (
        carry_data["carry_price_difference"]
        * 365
        / carry_data["carry_day_gap"]
    ).where(valid)

    return carry_data[carry_cols]
