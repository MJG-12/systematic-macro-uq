# Systematic Futures Sizing and Uncertainty

Can conformal risk scaling or a fixed nonlinear forecast response improve the
performance of a published Carver trend and carry futures baseline?

We apply both alternatives at the final sizing stage only, leaving the
underlying alpha forecasts and all other portfolio conventions unchanged. The
first replaces the baseline's linear response to forecast strength with a
fixed inverse quadratic response adapted from recent work on nonlinear time
series momentum. The Adaptive Conformal Inference variant replaces volatility
in the denominator of the sizing rule with the conformal interval half width.
Fixed conformal serves as the corresponding nonadaptive benchmark.

Both changes improved development performance, but neither improvement was
corroborated in the 2022–2026 holdout. The appropriate practical default
remains linear volatility sizing. The nonlinear result, however, is especially
worth further testing because its response and normalization contain no
parameters fitted to strategy returns, and its development advantage was
broad.

See the [full report](reports/report/report.pdf) for the methodology,
diagnostics and detailed results. Compact results in machine readable form are
available under [`reports/report/results/`](reports/report/results/).

## Main results

The canonical comparison charges 2 bps on each side of a trade:

| Sizing method | Development Sharpe, 2014–21 | Holdout Sharpe |
|---|---:|---:|
| Linear volatility | 0.697 | 0.073 |
| Clipped control | 0.859 | 0.052 |
| Inverse quadratic | 1.039 | -0.035 |
| Fixed conformal | 0.739 | -0.006 |
| Adaptive conformal inference | 0.810 | -0.007 |

The inverse quadratic improvement appeared at every tested cost, in both
halves of the development period and in 11 of 12 markets. Because its form and
scaling were fixed independently of our data, this is notable development
evidence, but the holdout result does not corroborate the improvement. A
period of about 4.3 years is too short to establish whether the development
result was persistent or specific to the sample.

Adaptive conformal inference (ACI) reached 94.89% coverage in development and
95.01% in holdout against a 95% target. It reduced rolling local coverage error
relative to fixed conformal and improved development Sharpe. In the holdout,
however, its Sharpe was essentially unchanged from fixed conformal and remained
below linear volatility sizing. The available evidence therefore does not
confirm that adaptive calibration improves sizing, but is still too limited to
reject ACI more generally.

## Experimental design

| Element | Fixed choice |
|---|---|
| Universe | 12 CME futures markets: ES, NQ, ZN, ZB, 6E, 6J, CL, NG, GC, HG, ZC and ZS |
| Data | Daily futures settlements, volume and contract definitions from 2011 |
| IDM estimation | 5 January 2012–3 January 2014 |
| Development | 6 January 2014–31 December 2021 |
| Holdout | 3 January 2022–30 April 2026 |
| Allocation | 25% per asset class, equal within each class |
| Risk convention | 10% annual portfolio volatility target |
| Costs | 1, 2 and 5 bps on each side of a trade |

The forecast sleeve follows the companion implementation for Robert Carver's
*Systematic Trading*: EWMAC 16/64, 32/128 and 64/256 plus smoothed carry, with
the published scalars, weights, forecast diversification multiplier and
forecast caps. Following *Advanced Futures Trading Strategies*, volatility
combines the span 32 EWMA estimate with its rolling ten year average, using
70/30 weights.

Signals use an additive stitched price series to avoid artificial roll jumps
and are lagged by one trading day. Turnover accounts for position drift, rolls
incur both closing and opening costs, and positions are not subsequently
normalized to a fixed daily gross exposure.

## Sizing methods

### Nonlinear forecast response

Carver's forecast linearity tests motivate asking whether extreme trend and
carry forecasts deserve proportionally larger positions. The linear baseline
uses $x=F/10$ as its position strength. To express the fixed inverse
quadratic benchmark from
[Nonlinear Time Series Momentum](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5933974),
in standardized signal units, we use a mean zero Gaussian reference calibrated
to Carver's forecast convention: expected absolute magnitude 10 before
applying the cap at $\pm20$. Under this reference, the capped position
strength has variance $v$ and standardized value $s$:

$$
v=1.279,
\qquad
s=\frac{F/10}{\sqrt v}.
$$

The inverse quadratic candidate uses $s/(1+s^2)$, while the clipped control
uses $\operatorname{clip}(s,-1,1)$. Both are given the same root mean square
magnitude as the linear forecast under the reference. The smooth response
reaches its maximum and the control stops increasing at $|F|=11.31$.

### Conformal risk sizing

For each market, the symmetric conformal score is the absolute raw daily
return. A rolling window of 256 completed scores produces a symmetric
conformal prediction interval for the next day's return. Its half width
replaces volatility in the final position denominator. ACI adapts the coverage
level by market, while fixed conformal serves as the control.

Coverage levels from 95% through 99% were examined during development, and the
conventional 95% level was selected from the broad 95–97% performance
plateau.

## Repository

```text
src/systematic_macro_uq/
  data/          futures construction and roll schedules
  signals/       volatility, EWMAC, carry and forecast mappings
  forecasting/   shared forecast pipeline and diagnostics
  conformal/     rolling calibration and risk application
  backtest/      positions, roll costs and evaluation
scripts/         dataset, development and holdout runners
reports/report/  report source, PDF, figures and compact results
tests/           focused timing and portfolio safeguards
```

The complete daily positions and portfolio outputs are generated under the
ignored `data/processed/` directory.

## Reproduction

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test,report]"
pytest -q
```

With a local `.env` containing `DATABENTO_API_KEY`, the main workflows are:

```bash
python scripts/build_futures_dataset.py
python scripts/calculate_idm.py
python scripts/run_development_backtest.py
python scripts/run_development_conformal_sizing.py
python scripts/run_holdout_backtest.py
python scripts/run_holdout_conformal_sizing.py
```

Additional scripts reproduce the forecast bin and volatility diagnostics. To
rebuild the tracked report outputs and PDF:

```bash
python scripts/build_report_assets.py
quarto render reports/report/report.qmd --to pdf
```

Quarto must be installed separately.

## Limitations

- Twelve markets and a holdout of approximately 4.3 years provide limited
  independent evidence for futures strategies with medium and long horizons.
- The backtest uses continuous notional weights and does not translate them
  into integer contracts or test margin feasibility.
- Costs use stylized linear bps rather than separate spread, commission and
  slippage estimates for each contract.

## AI assistance

OpenAI Codex assisted with research discussion, implementation and refactoring,
debugging, analysis, code review and writing. The author initiated and directed
the project, contributed substantial portions of the implementation, reviewed
the outputs and takes responsibility for the final code, results and
conclusions.

## References

- Robert Carver, *Systematic Trading*, Harriman House, 2015.
- Robert Carver, *Advanced Futures Trading Strategies*, Harriman House, 2023.
- Wayne Ferson and Andrew Siegel, “The Efficient Use of Conditioning
  Information in Portfolios,” *Journal of Finance*, 2001.
- Tobias Moskowitz, Riccardo Sabbatucci, Andrea Tamoni and Björn Uhl,
  “Nonlinear Time Series Momentum,” working paper, 2025.
- Isaac Gibbs and Emmanuel Candes, “Adaptive Conformal Inference Under
  Distribution Shift,” NeurIPS, 2021.
