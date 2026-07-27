import pandas as pd

from systematic_macro_uq.config import RunConfig
from systematic_macro_uq.data.futures import (
    build_carry_data,
    build_contract_panel,
    build_contract_returns_data,
    build_daily_statistic_data,
    build_futures_metadata,
    build_roll_schedule,
    fetch_futures_data,
)


def load_or_fetch_universe(config, market):
    input_path = config.raw_data_run_dir / f"definition_{market}.parquet"
    if config.REUSE_DATA and input_path.exists():
        return pd.read_parquet(input_path)

    universe = fetch_futures_data(
        "definition",
        market,
        config.DEFINITION_START,
        config.END,
    )
    input_path.parent.mkdir(parents=True, exist_ok=True)
    universe.to_parquet(input_path)
    return universe


def load_or_fetch_stats(config, market, universe_df):
    input_path = config.raw_data_run_dir / f"stats_{market}.parquet"
    if config.REUSE_DATA and input_path.exists():
        return pd.read_parquet(input_path)

    raw_symbols = (
        universe_df["raw_symbol"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    stats = fetch_futures_data(
        "statistics",
        market,
        config.PRICE_START,
        config.END,
        symbols=raw_symbols,
        stype_in="raw_symbol",
    )

    stats = stats[stats["stat_type"].isin([3, 6])].copy()

    input_path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_parquet(input_path)
    return stats


def fetch_and_build_market_dataset(config, market):
    output_path = config.by_market_dir / f"futures_dataset_{market}.parquet"
    if config.REUSE_DATA and output_path.exists():
        return pd.read_parquet(output_path)

    universe_raw = load_or_fetch_universe(config, market)
    universe_df = build_futures_metadata(universe_raw)

    stats_raw = load_or_fetch_stats(config, market, universe_df)

    prices_df = build_daily_statistic_data(stats_raw, "settlement_price")
    volumes_df = build_daily_statistic_data(stats_raw, "volume")

    panel = build_contract_panel(universe_df, prices_df, volumes_df)

    schedule = build_roll_schedule(
        panel,
        config.min_days_to_expiry_by_market[market],
    )
    contract_returns = build_contract_returns_data(panel)
    carry = build_carry_data(panel, schedule)

    dataset = pd.merge(
        schedule,
        contract_returns,
        on=["date", "held_instrument_id"],
        how="left",
    )

    dataset = pd.merge(
        dataset,
        carry,
        on=["date", "market", "selected_instrument_id"],
        how="left",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)

    return dataset


def build_futures_dataset(config):
    output_path = config.futures_daily_path
    if config.REUSE_DATA and output_path.exists():
        return pd.read_parquet(output_path)

    market_datasets = [
        fetch_and_build_market_dataset(config, market)
        for market in config.markets
    ]
    data = pd.concat(market_datasets, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output_path, index=False)

    return data


if __name__ == "__main__":
    build_futures_dataset(RunConfig())
