import numpy as np
import pandas as pd
import pytest

from systematic_macro_uq.backtest.evaluation import select_evaluation_period
from systematic_macro_uq.backtest.costs import add_roll_aware_turnover
from systematic_macro_uq.backtest.positions import (
    estimate_instrument_diversification_multiplier,
)
from systematic_macro_uq.conformal.aci import (
    add_per_market_conformal_radii,
)
from systematic_macro_uq.conformal.application import (
    construct_conformal_positions,
)
from systematic_macro_uq.config import (
    ForecastConfig,
    PortfolioConfig,
    RunConfig,
)
from systematic_macro_uq.data.futures import build_carry_data
from systematic_macro_uq.forecasting.features import add_forward_return_target
from systematic_macro_uq.signals.carry import add_carry_forecast
from systematic_macro_uq.signals.composite import add_mapped_forecasts
from systematic_macro_uq.signals.vol import add_ewma_volatility


def utc_dates(values):
    return pd.to_datetime(values, utc=True)


def test_companion_forecast_specification_is_locked():
    config = ForecastConfig()

    assert config.volatility_span == 32
    assert config.volatility_long_run_weight == 0.30
    assert config.volatility_long_run_window_days == 2560
    assert config.ewmac_rules == (
        (16, 64, 3.75, 0.21),
        (32, 128, 2.65, 0.08),
        (64, 256, 1.87, 0.21),
    )
    assert config.carry_smoothing_com == 90
    assert config.carry_scalar == 30.0
    assert config.carry_weight == 0.50
    assert config.forecast_diversification_multiplier == 1.31
    assert config.inverse_quadratic_reference_variance == 1.279475
    assert config.inverse_quadratic_scale == 3.170506
    assert config.clipped_scale == 1.341270
    assert (
        sum(rule[3] for rule in config.ewmac_rules)
        + config.carry_weight
    ) == pytest.approx(1.0)


def test_one_day_target_uses_the_next_return_and_current_volatility():
    df = pd.DataFrame(
        {
            "market": "ES",
            "date": utc_dates(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "held_daily_return": [0.0, 0.01, -0.02],
            "annualized_pct_vol": [0.16, 0.32, 0.32],
        }
    )

    result = add_forward_return_target(
        df,
        horizon_days=1,
        trading_days_per_year=256,
        target_column="target",
    )

    assert result.loc[0, "target_end_date"] == result.loc[1, "date"]
    assert result.loc[0, "target"] == pytest.approx(1.0)
    assert result.loc[1, "target"] == pytest.approx(-1.0)


def test_central_radius_uses_only_prior_absolute_returns():
    df = pd.DataFrame(
        {
            "market": "ES",
            "date": utc_dates(
                ["2020-01-01", "2020-01-02", "2020-01-03"]
            ),
            "held_daily_return": [-1.0, 2.0, -100.0],
        }
    )

    result = add_per_market_conformal_radii(
        df,
        target_coverage=0.5,
        calibration_window=2,
        learning_rate=0.0,
        prefix="fixed",
    )

    assert np.isnan(result.loc[1, "fixed_radius"])
    assert result.loc[2, "fixed_radius"] == pytest.approx(2.0)
    assert result.loc[2, "fixed_error"] == pytest.approx(1.0)


def test_central_rule_uses_two_sided_gaussian_reference_scale():
    positions = pd.DataFrame(
        {
            "date": utc_dates(["2020-01-01"]),
            "market": "ES",
            "held_instrument_id": "ESH0",
            "instrument_weight": 1.0,
            "forecast_for_position": 10.0,
            "volatility_for_position": 0.20,
            "held_daily_return": 0.0,
            "return_contribution": 0.0,
            # A central 68.27% Gaussian set has radius one sigma.
            "radius": 0.20 / 16,
        }
    )

    result = construct_conformal_positions(
        positions,
        radius_column="radius",
        volatility_target=0.10,
        average_forecast=10.0,
        instrument_diversification_multiplier=1.0,
        trading_days_per_year=256,
        target_coverage=0.6826894921370859,
    )

    assert result.loc[0, "conformal_annualized_risk"] == pytest.approx(0.20)
    assert result.loc[0, "position"] == pytest.approx(0.5)


def test_aci_error_updates_only_the_next_alpha():
    df = pd.DataFrame(
        {
            "market": "ES",
            "date": pd.date_range("2020-01-01", periods=4, tz="UTC"),
            "held_daily_return": [1.0, 2.0, 3.0, 0.0],
        }
    )

    result = add_per_market_conformal_radii(
        df,
        target_coverage=0.5,
        calibration_window=2,
        learning_rate=0.1,
        prefix="aci",
    )

    assert result.loc[2, "aci_alpha"] == pytest.approx(0.5)
    assert result.loc[2, "aci_error"] == pytest.approx(1.0)
    assert result.loc[3, "aci_alpha"] == pytest.approx(0.45)


def test_conformal_state_is_continuous_at_an_evaluation_boundary():
    df = pd.DataFrame(
        {
            "market": "ES",
            "date": pd.date_range("2020-01-01", periods=5, tz="UTC"),
            "held_daily_return": [1.0, 2.0, 3.0, 0.0, 1.0],
        }
    )
    kwargs = {
        "target_coverage": 0.5,
        "calibration_window": 2,
        "learning_rate": 0.1,
        "prefix": "aci",
    }

    continuous = add_per_market_conformal_radii(df, **kwargs)
    truncated = add_per_market_conformal_radii(df.iloc[:4], **kwargs)
    columns = ["aci_alpha", "aci_radius", "aci_error"]

    pd.testing.assert_series_equal(
        continuous.loc[3, columns],
        truncated.loc[3, columns],
    )
    reset = add_per_market_conformal_radii(df.iloc[3:], **kwargs)
    assert np.isnan(reset.loc[0, "aci_radius"])


def test_infinite_conformal_radius_sets_position_to_zero():
    positions = pd.DataFrame(
        {
            "date": utc_dates(["2020-01-01"]),
            "market": "ES",
            "held_instrument_id": "ESH0",
            "instrument_weight": 1.0,
            "forecast_for_position": 10.0,
            "volatility_for_position": 0.20,
            "held_daily_return": 0.01,
            "return_contribution": 0.0,
            "radius": np.inf,
        }
    )

    result = construct_conformal_positions(
        positions,
        radius_column="radius",
        volatility_target=0.10,
        average_forecast=10.0,
        instrument_diversification_multiplier=1.0,
        trading_days_per_year=256,
        target_coverage=0.95,
    )

    assert result.loc[0, "position"] == pytest.approx(0.0)
    assert result.loc[0, "return_contribution"] == pytest.approx(0.0)


def test_roll_turnover_charges_close_and_open_legs():
    df = pd.DataFrame(
        {
            "market": "ES",
            "date": utc_dates(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "held_instrument_id": ["ESH0", "ESH0", "ESM0"],
            "position": [1.0, 1.0, 0.5],
            "held_daily_return": 0.0,
            "return_contribution": 0.0,
        }
    )

    result = add_roll_aware_turnover(df)

    assert not result.loc[1, "roll_trade"]
    assert result.loc[1, "turnover"] == pytest.approx(0.0)
    assert result.loc[2, "roll_trade"]
    assert result.loc[2, "roll_turnover"] == pytest.approx(1.5)


def test_idm_floors_negative_correlations_at_zero():
    dates = pd.date_range("2020-01-01", periods=5, tz="UTC")
    df = pd.DataFrame(
        {
            "market": ["A"] * 5 + ["B"] * 5,
            "date": list(dates) * 2,
            "forecast_combined": 1.0,
            "annualized_pct_vol": 1.0,
            "held_daily_return": [0, 1, -1, 1, -1, 0, -1, 1, -1, 1],
        }
    )

    result = estimate_instrument_diversification_multiplier(
        df,
        {"A": 0.5, "B": 0.5},
    )

    assert result == pytest.approx(np.sqrt(2))


def test_mapped_forecasts_use_fixed_variance_matching():
    turning_point = 10 * np.sqrt(1.279475)
    dates = pd.date_range("2020-01-01", periods=9, tz="UTC")
    df = pd.DataFrame(
        {
            "market": "ES",
            "date": dates,
            "forecast_combined": [
                -20.0,
                -10.0,
                -2.0,
                0.0,
                2.0,
                5.0,
                10.0,
                turning_point,
                20.0,
            ],
        }
    )

    result = add_mapped_forecasts(
        df,
        average_forecast=10.0,
        inverse_quadratic_scale=3.170506,
        inverse_quadratic_reference_variance=1.279475,
        clipped_scale=1.341270,
    )

    for column in ["forecast_inverse_quadratic", "forecast_clipped"]:
        assert np.sign(result[column]).eq(np.sign(df["forecast_combined"])).all()

    inverse_quadratic = result.set_index(
        df["forecast_combined"]
    )["forecast_inverse_quadratic"]
    assert inverse_quadratic.loc[10.0] == pytest.approx(13.9089307845)
    assert inverse_quadratic.loc[20.0] == pytest.approx(12.0106866687)
    assert inverse_quadratic.loc[20.0] < inverse_quadratic.loc[10.0]

    clipped = result.set_index(df["forecast_combined"])[
        "forecast_clipped"
    ]
    assert clipped.loc[10.0] == pytest.approx(13.41270)
    assert clipped.loc[20.0] == pytest.approx(
        clipped.loc[turning_point]
    )


def test_fixed_mappings_do_not_depend_on_other_rows_or_dates():
    base = pd.DataFrame(
        {
            "market": "ES",
            "date": utc_dates(["2020-01-01", "2020-01-02"]),
            "forecast_combined": [5.0, 10.0],
        }
    )
    extended = pd.concat(
        [
            base,
            pd.DataFrame(
                {
                    "market": "ES",
                    "date": utc_dates(["2030-01-01"]),
                    "forecast_combined": [20.0],
                }
            ),
        ],
        ignore_index=True,
    )

    kwargs = {
        "average_forecast": 10.0,
        "inverse_quadratic_scale": 3.170506,
        "inverse_quadratic_reference_variance": 1.279475,
        "clipped_scale": 1.341270,
    }
    base_result = add_mapped_forecasts(base, **kwargs)
    extended_result = add_mapped_forecasts(extended, **kwargs)

    columns = ["forecast_inverse_quadratic", "forecast_clipped"]
    pd.testing.assert_frame_equal(
        base_result[columns],
        extended_result.loc[:1, columns],
    )


def test_blended_volatility_damps_a_short_run_spike():
    dates = pd.date_range("2020-01-01", periods=101, tz="UTC")
    df = pd.DataFrame(
        {
            "market": "ES",
            "date": dates,
            "held_daily_return": [0.001] * 100 + [0.05],
            "selected_settlement_price": 100.0,
        }
    )

    short_run_only = add_ewma_volatility(
        df,
        span=32,
        trading_days_per_year=256,
        long_run_weight=0.0,
        long_run_window_days=2560,
    )
    blended = add_ewma_volatility(
        df,
        span=32,
        trading_days_per_year=256,
        long_run_weight=0.3,
        long_run_window_days=2560,
    )

    spike_short = short_run_only["daily_pct_vol"].iloc[-1]
    spike_blended = blended["daily_pct_vol"].iloc[-1]
    long_run_mean = short_run_only["daily_pct_vol"].rolling(
        2560, min_periods=1
    ).mean().iloc[-1]

    assert spike_blended < spike_short
    assert spike_blended == pytest.approx(
        0.3 * long_run_mean + 0.7 * spike_short
    )


def test_smoothed_carry_uses_companion_com_and_respects_cap():
    dates = pd.date_range("2021-04-01", periods=2, tz="UTC")
    df = pd.DataFrame(
        {
            "market": "ES",
            "date": dates,
            "carry_annualized_price": [0.0, 2.0],
            "annualized_price_vol": 1.0,
        }
    )

    result = add_carry_forecast(
        df,
        scalar=1.0,
        forecast_cap=20.0,
        smoothing_com=90,
    )

    # The companion call is ewm(com=90), whose default adjust=True gives
    # normalized weights of 90/91 and 1 for these two observations.
    assert result.loc[1, "forecast_carry"] == pytest.approx(182 / 181)

    capped = add_carry_forecast(
        df,
        scalar=100.0,
        forecast_cap=20.0,
        smoothing_com=90,
    )
    assert capped["forecast_carry"].max() == pytest.approx(20.0)


def test_nearest_contract_carry_uses_current_minus_next_price():
    date = pd.Timestamp("2020-01-02", tz="UTC")
    panel = pd.DataFrame(
        {
            "market": ["ES", "ES"],
            "date": [date, date],
            "expiration": utc_dates(["2020-03-20", "2020-06-19"]),
            "instrument_id": ["ESH0", "ESM0"],
            "settlement_price": [100.0, 95.0],
        }
    )
    schedule = pd.DataFrame(
        {
            "date": [date],
            "market": ["ES"],
            "selected_instrument_id": ["ESH0"],
        }
    )

    result = build_carry_data(panel, schedule)

    assert result.loc[0, "carry_price_difference"] == pytest.approx(5.0)
    assert result.loc[0, "carry_annualized_price"] > 0


def test_development_and_holdout_windows_are_disjoint():
    config = RunConfig()
    dates = pd.DataFrame(
        {
            "date": utc_dates(
                [
                    "2014-01-05",
                    "2014-01-06",
                    "2021-12-31",
                    "2022-01-01",
                ]
            )
        }
    )
    development = select_evaluation_period(
        dates,
        start=pd.Timestamp(config.DEVELOPMENT_START, tz="UTC"),
        end=pd.Timestamp(config.HOLDOUT_START, tz="UTC"),
    )
    holdout = select_evaluation_period(
        dates,
        start=pd.Timestamp(config.HOLDOUT_START, tz="UTC"),
    )

    assert set(development["date"]).isdisjoint(holdout["date"])
    assert development["date"].min() == pd.Timestamp(
        config.DEVELOPMENT_START,
        tz="UTC",
    )
    assert holdout["date"].min() == pd.Timestamp(
        config.HOLDOUT_START,
        tz="UTC",
    )
