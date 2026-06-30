import pandas as pd
from pathlib import Path
from systematic_macro_uq.data.futures import (fetch_futures_data,
                                              build_futures_metadata,
                                              build_daily_statistic_data,
                                              build_contract_panel,
                                              build_contract_returns_data,
                                              build_roll_schedule, 
                                              build_carry_data)

from systematic_macro_uq.config  import RunConfig


def load_or_fetch_universe(config, market):

    INPUT_PATH = config.raw_data_run_dir / f"definition_{market}.parquet"
    if config.REUSE_DATA and INPUT_PATH.exists():
        return pd.read_parquet(INPUT_PATH)
    
    universe = fetch_futures_data("definition", market, config.DEFINITION_START, config.END)
    INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    universe.to_parquet(INPUT_PATH, index=False)
    return universe


def load_or_fetch_stats(config, market):

    INPUT_PATH = config.raw_data_run_dir /  f"stats_{market}.parquet"
    if config.REUSE_DATA and INPUT_PATH.exists():
        return pd.read_parquet(INPUT_PATH)
    
    stats = fetch_futures_data("statistics", market, config.PRICE_START, config.END)
    INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    stats.to_parquet(INPUT_PATH, index=False)
    return stats


def fetch_and_build_market_dataset(config, market):
    
    OUTPUT_PATH = config.by_market_dir / f"futures_dataset_{market}.parquet"
    if config.REUSE_DATA and OUTPUT_PATH.exists():
        return pd.read_parquet(OUTPUT_PATH)

    universe_raw = load_or_fetch_universe(config, market)
    stats_raw = load_or_fetch_stats(config, market)

    universe_df = build_futures_metadata(universe_raw)
    prices_df = build_daily_statistic_data(stats_raw, "settlement_price")
    volumes_df = build_daily_statistic_data(stats_raw, "volume")

    panel = build_contract_panel(universe_df, prices_df, volumes_df)

    schedule = build_roll_schedule(panel, config.min_days_to_expiry_by_market[market])
    contract_returns = build_contract_returns_data(panel)
    carry = build_carry_data(panel, schedule)

    dataset = pd.merge(
        schedule, 
        contract_returns,
        on = ["date", "held_instrument_id"],
        how = "left"
    )
    
    dataset = pd.merge(
        dataset, 
        carry,
        on = ["date", "held_instrument_id"],
        how = "left"
    )
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(OUTPUT_PATH, index=False)

    return dataset


def build_futures_dataset(config):

    OUTPUT_PATH = config.futures_daily_path
    if config.REUSE_DATA and OUTPUT_PATH.exists():
        return pd.read_parquet(OUTPUT_PATH)

    market_datasets = [
        fetch_and_build_market_dataset(config, market)
        for market in config.markets
    ]
    data = pd.concat(market_datasets, ignore_index=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(OUTPUT_PATH, index=False)

    return data


if __name__ == "__main__":
    build_futures_dataset(RunConfig())
