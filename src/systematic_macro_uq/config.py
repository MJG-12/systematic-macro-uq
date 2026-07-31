from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ForecastConfig:
    trading_days_per_year: int = 256
    # AFTS strategy three: EWMA span 32, blended 30/70 with a ten-year
    # average of the estimate (expanding until a full window exists).
    volatility_span: int = 32
    volatility_long_run_weight: float = 0.30
    volatility_long_run_window_days: int = 2560

    ewmac_rules: tuple[tuple[int, int, float, float], ...] = (
        (16, 64, 3.75, 0.21),
        (32, 128, 2.65, 0.08),
        (64, 256, 1.87, 0.21),
    )

    carry_scalar: float = 30.0
    # Systematic Trading chapter 15 companion configuration: smooth the
    # daily raw carry forecast with pandas ewm(com=90).
    carry_smoothing_com: int = 90
    carry_weight: float = 0.50
    forecast_diversification_multiplier: float = 1.31
    forecast_cap: float = 20.0
    target_horizon: int = 1
    # Fixed mapping constants under the capped Gaussian Carver reference.
    inverse_quadratic_reference_variance: float = 1.279475
    inverse_quadratic_scale: float = 3.170506
    clipped_scale: float = 1.341270

    @property
    def target_column(self):
        return f"target_return_{self.target_horizon}d_vol_scaled"


@dataclass(frozen=True)
class PortfolioConfig:
    volatility_target: float = 0.10
    average_forecast: float = 10.0
    instrument_diversification_multiplier: float = 2.13605733015194
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
    asset_classes: dict[str, str] = field(default_factory=lambda: {
        "ES": "equities",
        "NQ": "equities",
        "ZN": "rates",
        "ZB": "rates",
        "6E": "fx",
        "6J": "fx",
        "CL": "commodities",
        "NG": "commodities",
        "GC": "commodities",
        "HG": "commodities",
        "ZC": "commodities",
        "ZS": "commodities",
    })


@dataclass(frozen=True)
class ConformalConfig:
    calibration_window: int = 256
    aci_learning_rate: float = 0.005
    development_coverages: tuple[float, ...] = (
        0.95,
        0.96,
        0.97,
        0.98,
        0.99,
    )
    selected_coverage: float = 0.95


@dataclass(frozen=True)
class RunConfig:
    RUN_ID: str = "dev_v1"
    SOURCE_DATA_RUN_ID: str = "full_run_extended"
    PRICE_START: str = "2011-01-01"
    END: str = "2026-05-01"
    DEFINITION_START: str = "2010-06-06"
    DEVELOPMENT_START: str = "2014-01-06"
    HOLDOUT_START: str = "2022-01-01"
    REUSE_DATA: bool = True
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
        return self.RAW_DATA_DIR / self.SOURCE_DATA_RUN_ID

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
    def conformal_backtest_dir(self):
        return self.backtest_dir / "conformal"

    @property
    def backtest_dir(self):
        return self.processed_data_run_dir / "backtests"

    @property
    def development_backtest_dir(self):
        return self.backtest_dir / "development"

    @property
    def holdout_backtest_dir(self):
        return self.backtest_dir / "holdout"
