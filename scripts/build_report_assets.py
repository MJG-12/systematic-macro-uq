from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from systematic_macro_uq.config import RunConfig

CANONICAL_COST_BPS = 2
REPORT_DIR = Path("reports/report")
RESULTS_DIR = REPORT_DIR / "results"
FIGURES_DIR = REPORT_DIR / "figures"
STRATEGY_LABELS = {
    "linear": "Linear",
    "clipped": "Clipped control",
    "inverse_quadratic": "Inverse quadratic",
}
STRATEGY_COLOURS = {
    "linear": "#4C78A8",
    "clipped": "#72B7B2",
    "inverse_quadratic": "#E45756",
}


def _save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, bbox_inches="tight", dpi=200)
    plt.close(fig)


def _write_combined_csv(sample_dirs, filename):
    frames = []
    for sample, directory in sample_dirs.items():
        frame = pd.read_csv(directory / filename)
        frame.insert(0, "sample", sample)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(RESULTS_DIR / filename, index=False)
    return combined


def _load_net_market_attribution(
    sample_dirs,
    cost_bps,
    asset_classes,
    trading_days_per_year,
):
    frames = []
    for sample, directory in sample_dirs.items():
        positions = pd.read_parquet(directory / "positions.parquet")
        positions["asset_class"] = positions["market"].map(asset_classes)
        positions["net_contribution"] = (
            positions["return_contribution"]
            - positions[f"cost_{cost_bps}bps"]
        )
        grouped = positions.groupby(["strategy", "asset_class", "market"])
        attribution = (
            (
                grouped["net_contribution"].sum()
                / grouped["date"].nunique()
                * trading_days_per_year
            )
            .rename("annualized_contribution")
            .reset_index()
        )
        attribution.insert(0, "cost_bps", cost_bps)
        attribution.insert(0, "sample", sample)
        frames.append(attribution)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(
        RESULTS_DIR / "attribution_by_market.csv",
        index=False,
    )
    return combined


def _plot_timeline():
    periods = [
        ("Warm-up", "2011-01-01", "2012-01-04", "#BAB0AC"),
        ("IDM estimation", "2012-01-05", "2014-01-03", "#F2CF5B"),
        ("Development", "2014-01-06", "2021-12-31", "#4C78A8"),
        ("Holdout", "2022-01-03", "2026-04-30", "#E45756"),
    ]
    fig, ax = plt.subplots(figsize=(8.0, 1.8))
    for row, (label, start, end, colour) in enumerate(periods):
        start_date = pd.Timestamp(start)
        end_date = pd.Timestamp(end)
        ax.barh(
            row,
            end_date - start_date,
            left=start_date,
            height=0.55,
            color=colour,
        )
        ax.text(
            start_date + (end_date - start_date) / 2,
            row,
            label,
            ha="center",
            va="center",
            fontsize=9,
        )
    ax.set_yticks([])
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2)
    _save_figure(fig, "sample_timeline.png")


def _plot_mappings(config):
    raw_forecast = np.linspace(-25, 25, 501)
    forecast = np.clip(
        raw_forecast,
        -config.forecast.forecast_cap,
        config.forecast.forecast_cap,
    )
    scaled = forecast / config.portfolio.average_forecast
    inverse_quadratic = (
        config.portfolio.average_forecast
        * config.forecast.inverse_quadratic_scale
        * scaled
        / (
            config.forecast.inverse_quadratic_reference_variance
            + scaled**2
        )
    )
    clipped = (
        config.portfolio.average_forecast
        * config.forecast.clipped_scale
        * np.clip(
            scaled,
            -np.sqrt(
                config.forecast.inverse_quadratic_reference_variance
            ),
            np.sqrt(
                config.forecast.inverse_quadratic_reference_variance
            ),
        )
    )

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.plot(
        raw_forecast,
        forecast,
        label="Linear",
        color=STRATEGY_COLOURS["linear"],
        linewidth=2,
    )
    ax.plot(
        raw_forecast,
        inverse_quadratic,
        label="Inverse quadratic",
        color=STRATEGY_COLOURS["inverse_quadratic"],
        linewidth=2,
    )
    ax.plot(
        raw_forecast,
        clipped,
        label="Clipped control",
        color=STRATEGY_COLOURS["clipped"],
        linewidth=2,
    )
    for cap in (-config.forecast.forecast_cap, config.forecast.forecast_cap):
        ax.axvline(
            cap,
            color="#777777",
            linestyle="--",
            linewidth=1,
        )
    ax.axhline(0, color="#555555", linewidth=0.7)
    ax.axvline(0, color="#555555", linewidth=0.7)
    ax.set(
        xlabel="Combined forecast before final cap",
        ylabel="Mapped forecast",
        xlim=(-25, 25),
    )
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.grid(alpha=0.2)
    _save_figure(fig, "forecast_mappings.png")


def _plot_bins(forecast_bins):
    centres = np.array([2.5, 7.5, 12.5, 17.5])
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for sample, colour in (
        ("development", "#4C78A8"),
        ("holdout", "#E45756"),
    ):
        sample_df = forecast_bins.loc[
            forecast_bins["sample"].eq(sample)
        ].reset_index(drop=True)
        ax.plot(
            centres,
            sample_df["mean_signed_target"],
            marker="o",
            linewidth=2,
            color=colour,
            label=sample.title(),
        )
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set(
        xlabel="Absolute combined-forecast bin",
        ylabel="Mean signed next-day target",
        xticks=centres,
        xticklabels=["0–5", "5–10", "10–15", "15–20"],
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    _save_figure(fig, "forecast_bins.png")


def _plot_cumulative_returns(portfolio_returns):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    for ax, sample in zip(axes, ("development", "holdout")):
        sample_df = portfolio_returns.loc[
            portfolio_returns["sample"].eq(sample)
        ]
        for strategy in ("linear", "clipped", "inverse_quadratic"):
            strategy_df = sample_df.loc[
                sample_df["strategy"].eq(strategy)
            ].sort_values("date")
            ax.plot(
                pd.to_datetime(strategy_df["date"]),
                (strategy_df["equity_curve"] - 1) * 100,
                label=STRATEGY_LABELS[strategy],
                color=STRATEGY_COLOURS[strategy],
                linewidth=1.7,
            )
        ax.axhline(0, color="#555555", linewidth=0.7)
        ax.set_title(sample.title())
        ax.set_xlabel("")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Cumulative net return (%)")
    axes[0].legend(frameon=False, fontsize=8)
    _save_figure(fig, "cumulative_returns.png")


def _plot_market_attribution(market_attribution, config):
    differences = (
        market_attribution.pivot_table(
            index=["sample", "market"],
            columns="strategy",
            values="annualized_contribution",
        )
        .assign(
            difference=lambda x: (
                x["inverse_quadratic"] - x["linear"]
            ) * 100
        )
        .reset_index()
    )
    markets = config.markets
    y = np.arange(len(markets))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for offset, sample, colour in (
        (-width / 2, "development", "#4C78A8"),
        (width / 2, "holdout", "#E45756"),
    ):
        values = (
            differences.loc[differences["sample"].eq(sample)]
            .set_index("market")
            .reindex(markets)["difference"]
        )
        ax.barh(
            y + offset,
            values,
            height=width,
            label=sample.title(),
            color=colour,
        )
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set(
        yticks=y,
        yticklabels=markets,
        xlabel="Inverse quadratic minus linear annual net contribution (pp)",
    )
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.2)
    _save_figure(fig, "market_attribution.png")


def build_report_assets(config):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sample_dirs = {
        "development": config.development_backtest_dir,
        "holdout": config.holdout_backtest_dir,
    }

    performance = _write_combined_csv(sample_dirs, "summary.csv")
    performance.to_csv(
        RESULTS_DIR / "strategy_performance.csv",
        index=False,
    )
    (RESULTS_DIR / "summary.csv").unlink()

    diagnostics = _write_combined_csv(sample_dirs, "diagnostics.csv")
    diagnostics.loc[
        diagnostics["cost_bps"].eq(CANONICAL_COST_BPS)
    ].to_csv(
        RESULTS_DIR / "strategy_diagnostics.csv",
        index=False,
    )
    (RESULTS_DIR / "diagnostics.csv").unlink()

    forecast_bins = _write_combined_csv(
        sample_dirs,
        "forecast_bins.csv",
    )
    market_attribution = _load_net_market_attribution(
        sample_dirs,
        cost_bps=CANONICAL_COST_BPS,
        asset_classes=config.portfolio.asset_classes,
        trading_days_per_year=config.forecast.trading_days_per_year,
    )

    conformal_dirs = {
        "development": (
            config.conformal_backtest_dir / "development_sizing"
        ),
        "holdout": config.conformal_backtest_dir / "holdout_sizing",
    }
    conformal_performance = _write_combined_csv(
        conformal_dirs,
        "summary.csv",
    )
    conformal_performance.to_csv(
        RESULTS_DIR / "conformal_performance.csv",
        index=False,
    )
    (RESULTS_DIR / "summary.csv").unlink()

    conformal_application = _write_combined_csv(
        conformal_dirs,
        "application_diagnostics.csv",
    )
    conformal_application.to_csv(
        RESULTS_DIR / "conformal_application_diagnostics.csv",
        index=False,
    )
    (RESULTS_DIR / "application_diagnostics.csv").unlink()
    pd.read_csv(
        config.development_backtest_dir / "volatility_diagnostic.csv"
    ).to_csv(
        RESULTS_DIR / "volatility_diagnostic.csv",
        index=False,
    )

    portfolio_frames = []
    for sample, directory in sample_dirs.items():
        portfolio = pd.read_parquet(
            directory / "portfolio_results.parquet"
        )
        portfolio = portfolio.loc[
            portfolio["cost_bps"].eq(CANONICAL_COST_BPS)
        ].copy()
        portfolio.insert(0, "sample", sample)
        portfolio_frames.append(portfolio)
    portfolio_returns = pd.concat(portfolio_frames, ignore_index=True)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
        }
    )
    _plot_timeline()
    _plot_mappings(config)
    _plot_bins(forecast_bins)
    _plot_cumulative_returns(portfolio_returns)
    _plot_market_attribution(market_attribution, config)


if __name__ == "__main__":
    build_report_assets(RunConfig())
