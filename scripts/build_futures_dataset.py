from pathlib import Path
from systematic_macro_uq.data.futures import build_futures_return_dataset


MARKETS = [
    "ES",  # S&P 500 E-mini
    "NQ",  # Nasdaq 100 E-mini
    "ZN",  # 10Y Treasury Note
    "ZB",  # 30Y Treasury Bond
    "6E",  # Euro FX
    "6J",  # Japanese Yen
    "CL",  # WTI Crude Oil
    "NG",  # Natural Gas
    "GC",  # Gold
    "HG",  # Copper
    "ZC",  # Corn
    "ZS",  # Soybeans
]

OUTPUT_PATH = Path("data/processed/futures_daily_returns.parquet")

def build_futures_dataset(
    markets=MARKETS,
    price_start="2014-01-01",
    end="2026-05-01",
    definition_start="2013-01-01",
    min_days_to_expiry=5,
    output_path=OUTPUT_PATH,
):
    data = build_futures_return_dataset(
        markets,
        price_start,
        end,
        definition_start,
        min_days_to_expiry,
    )
    data.to_csv(Path("/mnt/c/Users/MiroGhanem/Desktop/files/futures_returns.csv"), index=False)
    data.to_parquet(output_path, index=False)
    return data


if __name__ == "__main__":
    build_futures_dataset()