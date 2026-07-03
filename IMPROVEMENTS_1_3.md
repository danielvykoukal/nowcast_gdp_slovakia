# Improvements #1 (release calendar) and #3 (domestic-demand data)

Implemented from the [`AUDIT.md`](AUDIT.md) shortlist. Honest results below — one is a clear
correctness win, the other is a legitimate enrichment that did **not** fix the 2025 level bias.

## #1 — Real release calendar (correctness win)

**Problem.** In the point-in-time weekly nowcast, the domestic revisable series (production,
retail, unemployment) were gated only by the OECD *edition*, which lags the national release by
~2–3 months (the June-2026 OECD edition carries IP only through March, while SUSR/Eurostat had
April out in early June). So the simulation withheld domestic data ~1–2 months longer than a real
analyst would have had it.

**Fix.** Every series is now gated by a realistic **day-level release calendar** (IP T+40, retail
T+40, construction T+45, unemployment T+30, surveys T+0, financial T+5, services T+60, HICP T+17).
For the revisable series we still use the OECD **first-release vintage values**, but the newest
month the OECD edition hasn't published yet (which the calendar says is available) is filled with
the Eurostat value — a near-first-release proxy for that one month, documented as a minimal
hindsight residual. Code: `src/weekly_realtime.py::panel_asof`.

**Effect.** Domestic data now enters at the correct Friday: e.g. at 26-Jun-2026 the last available
IP month is **April** (was ~February). This changes the *within-quarter path* — the nowcast reacts
to weak domestic prints ~2 months earlier — while end-of-window nowcasts are essentially unchanged
(by then all data is in either way). This makes the real-time chart faithful to an analyst's true
information set; it does not by itself change the flash-tracking at the endpoint.

## #3 — Domestic-demand data (enrichment; did NOT fix the bias)

**Added** two clean monthly Eurostat series (see [`DATA_CATALOGUE.md`](DATA_CATALOGUE.md)):
`services_iaf` (accommodation & food turnover — discretionary consumption) and `hicp` (consumer
prices — real-income control). Panel is now **19 monthly indicators + GDP**.

**Result — negative, and worth stating plainly.** Adding these did not reduce the 2025
over-prediction:

| Model | mean 2025 overshoot vs flash |
|---|---|
| Full single-factor, before (17 series) | +0.52pp |
| Full single-factor, after (19 series) | +0.52pp |
| Block model, foreign+domestic blocks | +0.44pp |

The reason is structural, not informational: in a common-factor model the new domestic series load
on the **same** factor that the (strong, high-comovement) industrial/foreign block dominates, so
they cannot pull GDP down. Even giving domestic demand its **own factor block** barely helps
(+0.47 → +0.44), because GDP still loads mainly on the industrial Global factor. The 2025 bias is a
**level/regime problem**, not a missing-series problem.

**What actually fixes the level bias** remains the real-time **intercept correction** from the
audit (2022–2026 RMSE 0.53 → 0.41; `python src/weekly_realtime.py 2025 --corrected`), plus the
regime-window idea. Adding domestic data is still worthwhile — more information, and a genuine
demand signal for future regimes — but it is not the lever for this particular bias.

## Files
- `src/fetch_data.py`, `src/preprocess.py` — new series + transforms.
- `src/model.py`, `src/model_v2.py`, `src/weekly_nowcast.py` — publication lags (months / days).
- `src/weekly_realtime.py::panel_asof` — day-level calendar + Eurostat edge-fill.
- Charts refreshed: `outputs/weekly_realtime_2025.png`, `..._corrected.png`, `bias_investigation.png`.
