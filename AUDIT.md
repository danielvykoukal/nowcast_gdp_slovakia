# Due diligence audit — weekly real-time nowcast

Prompted by the observation: *"the nowcast dips close to the flash mid-quarter, then jumps
away after new information arrives — this is weird."* Full audit of the scripts, logic and
data; every suspect was tested empirically rather than argued.

## Checks that PASSED (no bug found)

| Suspect | Test | Verdict |
|---|---|---|
| Nowcast readout (`res.predict()` = one-step-ahead vs smoothed) | Compared `predicted` / `filtered` / `smoothed` information sets on the same fitted vintage (2026Q1, end-May) | All within 0.01pp (0.548 / 0.537 / 0.544) — readout choice immaterial |
| Warm-start parameter leak (vintage refits warm-started from full-sample params) | Refit end-May-2026 vintage cold (default EM starts) vs warm | Identical nowcast (0.548 both); EM converges to the same optimum |
| OECD vintage vs Eurostat series mismatch | corr of `ip_total` monthly growth, latest OECD edition vs Eurostat | corr 0.997, mean abs diff 0.12pp — consistent |
| Chronology of vintages | Editions enter end-of-month; revisions enter only the following Friday; target GDP withheld until flash | By construction; verified on 2025Q1/Q3 first-Friday vintages (different editions, different GDP history) |

## The actual explanation of "dip then jump"

The mid-quarter dip toward the flash was **not skill** — it was the model extrapolating the
previous quarter's weak flash while it had *no* within-quarter hard data. The subsequent jump
up was **real information**: e.g. 2026Q1 monthly data was genuinely strong (manufacturing
+4.8% MoM in Jan, retail +2.5% in Mar, construction +11.1% in Mar). The model correctly
revised up — and GDP then printed far weaker (+0.20) than the indicators implied (+0.55).

So the wedge is a **level bias between what the monthly indicators say and what GDP prints**,
concentrated in 2024Q3–2025. The chronology is fine; the model's mapping from indicators to
GDP drifted.

## Fix attempt 1 — spec surgery: REJECTED by validation

Dropping foreign IP + estimating on 2016+ looked spectacular on 2025 (mean error +0.48 →
+0.02pp) — but on untouched 2022–2024 quarters it *under*-predicts by −0.68pp with RMSE 0.92
vs 0.52 for the full spec. It simply recenters the model on the recent stagnation and breaks
everywhere else. Classic overfitting to the evaluation window; rejected. (Lesson recorded:
the 2025 bias is time-varying, not a permanent spec error.)

## Fix attempt 2 — real-time intercept correction: ADOPTED

Standard central-bank practice: subtract the mean of the last 4 quarters' nowcast errors,
using **only flashes already released** at the time of the nowcast (chronologically legal).
Tested over 2020Q1–2026Q1 (`outputs/audit_intercept_correction.csv`):

| Window | raw mean err | raw RMSE | corrected mean err | corrected RMSE |
|---|---|---|---|---|
| 2022Q1–2026Q1 (17 q) | +0.38 | 0.53 | **+0.02** | **0.41** |
| 2025Q1–2026Q1 (5 q) | +0.48 | 0.55 | **+0.07** | **0.39** |

It improves both the biased window *and* the untouched validation years — unlike spec
surgery. Cost: it lags (overcorrects quarters where the raw model happened to be right, e.g.
2025Q3) and would trail a genuine regime turn by ~2–3 quarters. That is the standard tradeoff
of adaptive bias corrections.

The corrected weekly convergence chart: `outputs/weekly_realtime_2025_corrected.png`
(`python src/weekly_realtime.py 2025 --corrected`; purple = corrected path).

## Remaining known caveats (documented, not hidden)

1. **OECD edition timing**: editions are assumed available end-of-month. SUSR publishes
   domestic IP ~day 10 and the flash ~day 45; our simulation therefore delays *domestic* data
   by ~2–3 weeks relative to reality, while foreign IP (non-OECD, final values, day-40 lag)
   is comparatively favoured. The true analyst would see domestic data earlier than this
   backtest assumes — i.e. the real weekly path would move sooner, not later.
2. **Non-OECD series carry today's (revised) values** — sentiment/financial don't revise;
   foreign IP and construction do, so a small hindsight residual remains there.
3. **Intercept correction warm-up**: needs ≥2 observed errors; the correction series here is
   built from quarterly errors at a fixed (~quarter-end +2mo) reference point, not from the
   full weekly machinery — close, and disclosed.

## Ranked further improvements

1. **True release calendar** for domestic series (SUSR publishes exact dates) — removes the
   2–3-week domestic delay distortion in the weekly path.
2. **Time-varying level** (local-level component or rolling standardisation mean for GDP) —
   the model-internal version of the intercept correction, smoother and self-updating.
3. **Domestic demand data** (real wages, VAT receipts, card payments) to counterweight the
   industrial/foreign block that drove the 2025 wedge.
4. **Forecast combination** (full spec + domestic-only spec) — hedges regime dependence
   without hard spec choices.

*Generated during the audit; scripts: `src/weekly_realtime.py`, `outputs/audit_intercept_correction.csv`.*
