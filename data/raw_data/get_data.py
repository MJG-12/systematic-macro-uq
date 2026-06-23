import os
import pandas as pd
import databento as db
from dotenv import load_dotenv


load_dotenv()
API_KEY = os.environ["DATABENTO_API_KEY"]
client = db.Historical(API_KEY)

def return_futures_metadata(symbol, start_date, end_date):
    """Returns Databento raw futures metadata. 
    """
    df = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        schema="definition",
        symbols=f"{symbol}.FUT",
        stype_in="parent",
        start=start_date,
        end=end_date,
    ).to_df()

    return df[df["instrument_class"] == "F"]

def build_daily_futures_universe(raw_df):
    """Cleans Databento raw futures metadata by removing unused columns and dropping duplicate 
    contracts.
    """
    meta_cols = [
        "instrument_id",
        "raw_symbol",
        "security_update_action",
        "instrument_class",
        "exchange",
        "asset",
        "group",
        "activation",
        "expiration",
        "maturity_year",
        "maturity_month",
    ]
    df = raw_df[meta_cols].copy().sort_index()
    df["date"] = df.index.normalize()
    # df["ts_recv"] = df.index
    df = df.reset_index()
   
    df = (
    df.sort_values("ts_recv")
      .drop_duplicates(["date", "raw_symbol", "instrument_id"], keep="last")
      .sort_values(["date", "expiration", "raw_symbol", "instrument_id"])
      .reset_index(drop=True)
    )
    
    return df



if __name__ == "__main__":
    r = return_futures_metadata("ES", "2026-02-01", "2026-02-02")
    print(r.index)
    print(build_daily_futures_universe(r))
