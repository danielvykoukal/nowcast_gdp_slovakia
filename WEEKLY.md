# Weekly nowcast workflow

Simulates the real operating cadence: **every Friday**, assemble the data that has actually
been *released* by that date, re-estimate the dynamic factor model, and nowcast the target
quarter's GDP growth. As the quarter fills in, the nowcast converges to the realised value.
Implemented in [`src/weekly_nowcast.py`](src/weekly_nowcast.py).

## How the weekly vintage is built

Each series has a **publication lag in days** (`LAG_DAYS`): a monthly observation for reference
month *M* becomes available around `end-of-month(M) + lag`. Surveys (ESI, confidence) arrive at
month-end (~0 days), hard data ~40 days, the financial monthly value a few days after month-end.
Quarterly GDP is published as a flash ~45 days after the quarter ends.

For a given Friday, the model sees exactly the observations whose release date has passed — a
genuinely time-consistent ragged edge. The model is **re-estimated whenever the vintage changes**
(i.e. when a new release lands); between releases the nowcast is unchanged, so the weekly path is
a step function.

> Note: because the macro inputs are monthly, the nowcast moves on release dates (several clusters
> per quarter), not with fresh information every single week. Truly weekly movement would require
> weekly/daily inputs (electricity, toll, payment-system data — see `DATA_CATALOGUE.md`).

## Outputs

- **Single-quarter convergence** — `outputs/weekly_nowcast_<Q>.png`: each Friday's nowcast as a
  step, with the realised GDP as a horizontal line and markers at quarter-end and the flash
  release. For the latest completed quarter (2026Q1) the nowcast moves from +0.70% early in the
  quarter toward +0.55% (actual +0.20%).
- **Weekly backtest** — `outputs/weekly_backtest.png` (+ `weekly_backtest_rmse.csv`): the weekly
  process replayed for every quarter 2016Q1–2025Q4.
  - *Left:* each quarter's error path (`nowcast − actual`) collapses toward 0 as the quarter fills
    in (the few large excursions are 2020).
  - *Right:* nowcast **RMSE by lead time** — it falls from ~1.9pp mid-quarter to ~1.1pp just after
    quarter-end and ~0.5pp once the quarter's full hard data and the flash arrive (~7 weeks after
    quarter-end). This is the core value of the model: accuracy sharpens monotonically as the
    information set grows.

The biggest single accuracy gain comes **after** quarter-end, when the last month's hard data
(industrial production, trade, retail) is finally released — i.e. the "backcast" refinement.

## Running it

```bash
python src/weekly_nowcast.py
```

The historical backtest re-estimates the model for many vintages and takes ~10–12 minutes.
Narrow `start_q`/`end_q` in `backtest()` to run a shorter window quickly.

## Two versions — and why the chronologically-correct one is different

- **`src/weekly_nowcast.py`** applies the correct release *timing* but uses **today's revised
  values** for every series (we only have monthly Eurostat data). It's a fast approximation.
- **`src/weekly_realtime.py`** is the **point-in-time** version: at every Friday it uses the OECD
  data **edition that actually existed on that date**. A value keeps its first-release number
  until an OECD edition revises it, and that correction enters the nowcast only the *following*
  Friday — never earlier. It scores convergence against the **flash** (first release). This is the
  chronologically rigorous chart (`outputs/weekly_realtime_<Q>.png`).

The difference is visible: the real-time version starts *further* from the answer (early in the
quarter it genuinely only had older editions) and converges as new editions arrive — whereas the
revised-data version looks artificially well-informed early on. The real-time chart is the honest
one.

Run a single quarter or a whole year:

```bash
python src/weekly_realtime.py            # single showcase quarter (2026Q1)
python src/weekly_realtime.py 2025       # 2x2 grid of all four quarters of a year
```
