# Slovak GDP Nowcasting — Mixed-Frequency Dynamic Factor Model (MFDFM)

A small, reproducible pipeline that nowcasts Slovak quarterly real GDP growth from a panel
of higher-frequency public indicators, using a single-factor mixed-frequency dynamic factor
model (statsmodels `DynamicFactorMQ`: Kalman filter + EM, native ragged-edge handling).

Built from the data-architecture research in
[`Slovakia GDP Nowcasting Data Research.md`](Slovakia%20GDP%20Nowcasting%20Data%20Research.md).

## Baseline model (frozen 2026-07)

**[`src/baseline.py`](src/baseline.py)** is the production spec — the end point of the
2026-07 improvement round (anchor fix, foreign-IP drop, dead-series pruning, real
services, vacancies): 14 monthly series + quarterly [GDP, job vacancies], estimated
2013+, `DynamicFactorMQ` (1 factor, AR(2), AR(1) idio). True real-time 2024Q1–2026Q1:
**RMSE 0.23pp vs flash, bias ≈ 0** (start of round: 0.46 / +0.35). Full audit trail:
[`ANCHOR.md`](ANCHOR.md) (why), [`VARIABLES.md`](VARIABLES.md) (what, per variable).

```bash
python3 src/baseline.py    # current-quarter nowcast -> outputs/nowcast_baseline.csv
```

The spec is frozen for honest out-of-sample tracking from 2026Q3: score challengers on
quarters this spec never saw, and always report the naive benchmark (mean of last 4
flashes) alongside — on calm windows it is a serious competitor (see `ANCHOR.md`).

## Structure

| Part | Script | Output |
|---|---|---|
| 1. Data collection | [`src/fetch_data.py`](src/fetch_data.py) | `data/raw/*.csv`, [`DATA_CATALOGUE.md`](DATA_CATALOGUE.md) |
| 2. Modelling | [`src/preprocess.py`](src/preprocess.py) → [`src/dfm_statespace.py`](src/dfm_statespace.py) | `data/processed/*`, `outputs/nowcast_ssm.csv`, `outputs/loadings_ssm.csv`, `outputs/backtest_ssm.csv`, [`METHODOLOGY.md`](METHODOLOGY.md) |
| 3. Results | [`src/results.py`](src/results.py) | [`RESULTS.md`](RESULTS.md), `outputs/weights_vs_shares.csv/.png` |
| Enhanced model | [`src/model_v2.py`](src/model_v2.py) | [`IMPROVEMENTS.md`](IMPROVEMENTS.md), `outputs/{nowcast_v2,backtest_v2,horizon_rmse,news_v2}.csv` |
| Weekly workflow | [`src/weekly_nowcast.py`](src/weekly_nowcast.py) | [`WEEKLY.md`](WEEKLY.md), `outputs/weekly_nowcast_<Q>.png`, `outputs/weekly_backtest.png` |
| Real-time backtest | [`src/realtime_data.py`](src/realtime_data.py) → [`src/realtime_backtest.py`](src/realtime_backtest.py) | [`REALTIME.md`](REALTIME.md), `outputs/realtime_backtest.png` |
| Real-time weekly chart | [`src/weekly_realtime.py`](src/weekly_realtime.py) | `outputs/weekly_realtime_<Q>.png` (single quarter), `outputs/weekly_realtime_<year>.png` (grid), `outputs/weekly_timeline_<year>.png` (continuous) |
| Bias investigation | [`src/bias_investigation.py`](src/bias_investigation.py) | [`BIAS.md`](BIAS.md), `outputs/bias_investigation.png` |
| Audit + intercept correction | [`AUDIT.md`](AUDIT.md) | `outputs/audit_intercept_correction.csv`, `outputs/weekly_realtime_2025_corrected.png` |
| Improvements #1, #3 | [`IMPROVEMENTS_1_3.md`](IMPROVEMENTS_1_3.md) | real release calendar (`weekly_realtime.py`) + domestic-demand series |
| Plot | [`src/plot_nowcast.py`](src/plot_nowcast.py) | `outputs/nowcast_backtest.png` |

**Real-time / flash evaluation:** [`src/realtime_backtest.py`](src/realtime_backtest.py) uses the
OECD revisions database to reconstruct true data vintages and score the nowcast against the **GDP
flash (first release)** rather than final revised data — the credible real-time test. See
[`REALTIME.md`](REALTIME.md).

Three models, each with a purpose:

- **[`src/dfm_statespace.py`](src/dfm_statespace.py)** — the *methodology-faithful* model: a
  from-scratch Kalman filter matching the derivation in [`METHODOLOGY.md`](METHODOLOGY.md)
  (single factor, Mariano–Murasawa aggregation, white-noise idiosyncratic).
- **[`src/model.py`](src/model.py)** — single-factor `DynamicFactorMQ`, used for the
  interpretable weights-vs-GDP-shares analysis and as a cross-check of the explicit model.
- **[`src/model_v2.py`](src/model_v2.py)** — the *enhanced production nowcaster*: block factors,
  AR(1) idiosyncratic errors, extended data, honest recursive backtest, multi-horizon evaluation
  and news decomposition. See [`IMPROVEMENTS.md`](IMPROVEMENTS.md).

## Run

```bash
pip install -r requirements.txt

python src/fetch_data.py      # download public series (Eurostat + FRED); --refresh to redownload
python src/preprocess.py      # transform to stationary monthly panel + quarterly GDP
python src/dfm_statespace.py  # explicit state-space DFM (methodology): estimate, nowcast, backtest
python src/model.py           # DynamicFactorMQ cross-check (also feeds results.py)
python src/results.py         # weights-vs-GDP-shares comparison, writes RESULTS.md
python src/model_v2.py        # enhanced model: blocks, AR1 idio, recursive backtest, news
python src/realtime_data.py   # fetch OECD real-time vintages (GDP flash + first-release inputs)
python src/realtime_backtest.py  # true real-time backtest scored against the GDP flash
python src/plot_nowcast.py    # nowcast + backtest chart
```

Each script is independent and reads the previous stage's files, so you can re-run any step alone.

## Method (short)

- **Target:** Slovak real GDP, chain-linked, seasonally/calendar adjusted, QoQ % growth (Eurostat `namq_10_gdp`).
- **Panel:** 13 monthly indicators — industrial production, retail, construction, exports/imports,
  unemployment, sentiment (SK + German ESI), bond yield, EUR/USD — spanning the production,
  expenditure and income sides of GDP. See [`DATA_CATALOGUE.md`](DATA_CATALOGUE.md).
- **Model:** one common factor, AR(2) factor dynamics, AR(1) idiosyncratic components. The
  quarterly GDP series is linked to the monthly factor via the Mariano–Murasawa aggregation
  that `DynamicFactorMQ` applies internally; missing recent months (the "ragged edge") are
  filled by the Kalman filter.
- **Backtest:** fixed-parameter pseudo-real-time replay from 2010Q1, reconstructing each data
  vintage from publication lags (true real-time vintages are non-public — see the catalogue).
- **Interpretation:** the model's per-sector weights are compared against actual GVA shares;
  closer alignment ⇒ the model better reflects the real economy.

## Scope notes

- **Public data only.** Non-public series named in the research (SIPS interbank payments, RRZ
  real-time-vintage DB, employer insurance-contribution data, Ifo/ZEW, motor-vehicle IPP
  breakdown) are excluded, with public substitutes documented in `DATA_CATALOGUE.md`.
- Requires internet access for the Eurostat/FRED endpoints.
