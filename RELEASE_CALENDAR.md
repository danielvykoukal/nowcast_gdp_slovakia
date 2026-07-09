# Release calendar — when each dataset becomes available

For backtesting. This consolidates the publication timing that drives the real-time
reconstruction in the code, **checked against the actual 2025–26 release calendars** of
Eurostat, DG ECFIN and ŠÚ SR (sources at the bottom). Two representations exist in the code:

- **`LAG_DAYS`** ([`src/weekly_nowcast.py:38`](src/weekly_nowcast.py:38)) — day-level release timing.
  An observation for **reference month M** is treated as available on
  `end-of-month(M) + LAG_DAYS` days. Used by the *weekly* real-time engine.
- **`PUB_LAG`** ([`src/model.py:33`](src/model.py:33), [`src/model_v2.py:41`](src/model_v2.py:41)) — the same lag in **whole months**:
  how many recent monthly observations are missing when the nowcast is made at the **end of
  the target quarter**. Used by the *quarterly* backtest (`realtime_backtest.py` defaults to `2`).

**Bottom line:** the lags are mostly accurate. One is clearly wrong (`esi_ea`), two are
mildly conservative (`retail_vol`, `eur_usd`). Details in [§ Discrepancies](#discrepancies-to-fix).

---

## Verified timing

`Release ≈ end of reference month + LAG_DAYS`. "Observed" = actual 2025–26 release dates.

### Monthly indicators

| Slug | Source | Code lag (d) | Observed real lag | Verdict |
|---|---|---:|---|---|
| `esi_sk`, `ind_conf_sk`, `cons_conf_sk`, `esi_de` | DG ECFIN BCS (Eurostat `ei_bssi_m_r2`) | 0 | **−2 d** (released ~2 working days *before* month-end) | ✅ correct |
| `bond_10y` | Eurostat `irt_lt_mcby_m` | 5 | ~5–10 d | ✅ ok |
| `eur_usd` | FRED `DEXUSEU` | 5 | monthly mean complete **at month-end (~0 d)** | ⚠️ slightly slow |
| `hicp` | Eurostat `prc_hicp` | 17 | **~17 d** (May 2026 → 17 Jun 2026) | ✅ correct |
| `unemp_rate` | Eurostat `une_rt_m` | 30 | **~31 d** (Eurostat metadata) | ✅ correct |
| `esi_ea` | DG ECFIN BCS (Eurostat `ei_bssi_m_r2`) | 30 | **−2 d** — same release as `esi_sk` | ❌ **wrong** |
| `ip_total`, `ip_manuf` | Eurostat `sts_inpr_m` | 40 | SK national ~t+40; Eurostat aggregate **t+44–45** | ✅ ok (edge) |
| `exports_vol`, `imports_vol` | Eurostat `ei_eteu27_2020_m` | 40 | SK national ~t+40; Eurostat extra-EU ~t+45, intra-EU later | ✅ ok |
| `ip_de`, `ip_de_auto` | Eurostat `sts_inpr_m` | 40 | Destatis ~t+37; Eurostat t+45 *(series dropped)* | ✅ ok |
| `retail_vol` | Eurostat `sts_trtu_m` | 40 | **t+33–37** (~35 d) | ⚠️ slightly slow |
| `construction` | Eurostat `sts_copr_m` | 45 | ~t+45–50 | ✅ ok |
| `ip_ea` | Eurostat `sts_inpr_m` | 45 | Eurostat t+45 *(series dropped)* | ✅ ok |
| `real_wage_bill` | ŠÚ SR DATAcube | 55 | wages ~t+60 | ✅ ok (edge) |
| `services_iaf`, `services_H`, `services_J`, `services_N` | Eurostat `sts_setu_m` | 60 | ~t+60–65 | ✅ ok |

### Quarterly series

| Slug | Source | Code timing | Observed real timing | Verdict |
|---|---|---|---|---|
| `gdp_qoq` (**target**) | ŠÚ SR `namq_10_gdp` / OECD | flash **t+45** | ŠÚ SR flash ("rýchly odhad") **t+45** (Q3'25 → 14 Nov 2025); refined at **t+66** (5 Dec 2025) | ✅ correct |
| `vacancies` | Eurostat `jvs_q_r21` | ~2 months | ~t+70 | ✅ ok |

---

## Discrepancies to fix

**1. `esi_ea` — the one real error.** Code sets `LAG_DAYS = 30` / `PUB_LAG = 1`, and
`DATA_CATALOGUE.md` says "~1 mo". But the euro-area ESI is published in the **same DG ECFIN
Business & Consumer Survey release as the Slovak ESI** — all countries + EA + EU aggregates
go out together, ~2 working days before month-end. So `esi_ea` is available at the **same time**
as `esi_sk`, not a month later.
- **Effect on backtest:** the model is throwing away a fresh euro-area sentiment reading at
  every quarter-end — exactly the ragged edge where soft data carries most of the weight
  (see [`VARIABLES.md`](VARIABLES.md)). Since `esi_de` (correctly at lag 0) already carries the
  largest Kalman weight in the panel, giving `esi_ea` its true lag-0 timing could matter.
- **Fix:** set `esi_ea` to `LAG_DAYS = 0`, `PUB_LAG = 0` in [`weekly_nowcast.py:44`](src/weekly_nowcast.py:44),
  [`model.py:39`](src/model.py:39), [`model_v2.py:46`](src/model_v2.py:46). Worth a before/after real-time RMSE check.

**2. `retail_vol` — mildly conservative.** Real Eurostat euro-area retail lands ~t+35, not
t+40. At quarter-end this rarely changes the month-count (`PUB_LAG` still 2), so low priority,
but the weekly engine brings retail in ~5 days late.

**3. `eur_usd` — trivially conservative.** A monthly mean of a daily FRED series is fully known
at month-end (~t+0), not t+5. The series is dropped from the best model anyway, so cosmetic.

Everything else (GDP flash, IP, surveys, HICP, unemployment, construction, trade, services,
vacancies) matches the published calendars within a few days — fine for a monthly/quarterly
ragged-edge reconstruction.

---

## The two-stage GDP release (important for the target)

Slovakia publishes GDP QoQ **twice**:

| Stage | Timing | Content | Used as |
|---|---|---|---|
| Flash ("rýchly odhad") | **t+45** | headline GDP + employment growth | the **backtest target** — what the nowcast is scored against |
| Refined estimate | **t+66** | full expenditure/production structure, revised headline | later revisions in the OECD vintage editions |

Eurostat separately publishes a euro-area/EU **preliminary flash at t+30** and **flash at t+45**,
but for the SK country figure the ŠÚ SR t+45 flash is the first release. Scoring against the
flash (not the t+66 refined or today's final) is the correct real-time target — and it's the
choice `REALTIME.md` already argues for.

---

## How this maps to a nowcast date

At a run date **D** the ragged edge is: for each monthly series, the latest month **M** with
`end-of-month(M) + LAG_DAYS ≤ D` is observed. Worked example, nowcast at quarter-end (~Jun 30
for Q2):

- Surveys (`esi_sk`, confidences, `esi_de`, **and `esi_ea` once fixed**) → **June** in (lag ≤ 0).
- `hicp` → May (t+17); `unemp_rate` → May (t+31, borderline).
- Hard data (IP, retail, trade) → **April** only (t+40–45); May & June missing.
- Services, `real_wage_bill` → March/April (t+55–60) — slowest block.

So at quarter-end the only within-quarter hard data is month 1 of the quarter; months 2–3 are
carried by surveys + the factor. This is why the sentiment block dominates the effective filter
weight — and why the `esi_ea` timing error is worth fixing.

---

*Verified 2026-07 against: ŠÚ SR flash GDP release pages; Eurostat metadata (`une_rt_m`,
`prc_hicp`, `ei_bcs`) and euro-indicator press releases (IP, retail); DG ECFIN BCS calendar.
Code source of truth: `LAG_DAYS` in [`src/weekly_nowcast.py`](src/weekly_nowcast.py), `PUB_LAG`
in [`src/model.py`](src/model.py). Keep this table in sync if those dicts change.*
