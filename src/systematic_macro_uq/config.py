from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TradeabilityConfig:
    evidence_strength: float = 0.5  # annualized Sharpe-like state size, +/- a
    prior_dead: float = 0.70

    score_hurdle: float = 0.05
    penalty_dead: float = 0.05
    penalty_negative: float = 0.10

    min_validation_months: int = 24
    activate_prob_threshold: float = 0.50
    # deactivate_prob_threshold: float = 0.35

    placebo_draws: int = 200
    placebo_seed: int = 12345


@dataclass(frozen=True)
class RunConfig:
    RUN_ID: str = "full_run_1"
    PRICE_START: str = "2014-01-01"
    END: str = "2026-05-01"
    DEFINITION_START: str = "2013-01-01"
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
    transaction_costs: list[int] = field(default_factory=lambda: [5, 10, 15])
    tradeability: TradeabilityConfig = field(default_factory=TradeabilityConfig)

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
    def futures_signals_path(self):
        return self.processed_data_run_dir / "futures_daily_signals.parquet"

    @property
    def backtest_dir(self):
        return self.processed_data_run_dir / "backtests"

    @property
    def backtest_summary_path(self):
        return self.backtest_dir / "summary.csv"

