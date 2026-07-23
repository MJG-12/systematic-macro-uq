from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ForecastConfig:
    trading_days_per_year: int = 256
    volatility_span: int = 36

    ewmac_rules: tuple[tuple[int, int, float, float], ...] = (
        (16, 64, 3.75, 0.21),
        (32, 128, 2.65, 0.08),
        (64, 256, 1.87, 0.21),
    )

    carry_scalar: float = 30.0
    carry_weight: float = 0.50
    forecast_diversification_multiplier: float = 1.31
    forecast_cap: float = 20.0
    carry_update_weekday: int = 4  # Friday
    target_horizon: int = 20

    @property
    def target_column(self):
        return f"target_return_{self.target_horizon}d_vol_scaled"


@dataclass(frozen=True)
class PortfolioConfig:
    volatility_target: float = 0.10
    average_forecast: float = 10.0
    instrument_weights: dict[str, float] = field(default_factory=lambda: {
        # Equal 25% asset-class allocations, then equal within each class.
        "ES": 0.125,
        "NQ": 0.125,
        "ZN": 0.125,
        "ZB": 0.125,
        "6E": 0.125,
        "6J": 0.125,
        "CL": 1 / 24,
        "NG": 1 / 24,
        "GC": 1 / 24,
        "HG": 1 / 24,
        "ZC": 1 / 24,
        "ZS": 1 / 24,
    })


@dataclass(frozen=True)
class ConformalConfig:
    initial_training_years: int = 1
    target_coverage: float = 0.80
    aci_calibration_years: int = 1
    aci_learning_rate: float = 0.005
    application_history_years: int = 1
    application_minimum_days: int = 64
    abstention_quantile: float = 0.90


@dataclass(frozen=True)
class RunConfig:
    RUN_ID: str = "full_run_extended"
    PRICE_START: str = "2011-01-01"
    END: str = "2026-05-01"
    DEFINITION_START: str = "2010-06-06"
    HOLDOUT_START = "2022-01-01"
    REUSE_DATA: bool= True
    RAW_DATA_DIR: Path = Path("data/raw")
    PROCESSED_DATA_DIR: Path = Path("data/processed/")
    min_days_to_expiry_by_market: dict[str, int] = field(default_factory=lambda: {
        "ES": 7,
        "NQ": 7,
        "ZN": 7,
        "ZB": 7,
        "6E": 7,
        "6J": 7,
        "CL": 10,
        "NG": 10,
        "GC": 14,
        "HG": 14,
        "ZC": 14,
        "ZS": 14,
    })
    transaction_costs: list[int] = field(default_factory=lambda: [1, 2, 5])
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    conformal: ConformalConfig = field(default_factory=ConformalConfig)

    @property
    def markets(self):
        return list(self.min_days_to_expiry_by_market)
    
    @property
    def raw_data_run_dir(self):
        return self.RAW_DATA_DIR / self.RUN_ID
    
    @property
    def processed_data_run_dir(self):
        return self.PROCESSED_DATA_DIR / self.RUN_ID

    @property
    def by_market_dir(self):
        return self.processed_data_run_dir / "by_market"

    @property
    def futures_daily_path(self):
        return self.processed_data_run_dir / "futures_daily.parquet"

    @property
    def futures_forecasts_path(self):
        return self.processed_data_run_dir / "futures_daily_forecasts.parquet"

    @property
    def conformal_dataset_path(self):
        return self.processed_data_run_dir / "conformal_dataset.parquet"

    @property
    def conformal_backtest_dir(self):
        return self.backtest_dir / "conformal"

    @property
    def backtest_dir(self):
        return self.processed_data_run_dir / "backtests"

    @property
    def backtest_summary_path(self):
        return self.backtest_dir / "summary.csv"
