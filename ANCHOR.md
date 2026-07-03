# The anchor problem — why the nowcast over-predicts, and the two fixes

## Diagnosis (src/anchor_decomposition.py, src/anchor_decomposition_vintage.py)

The nowcast decomposes exactly as

    nowcast = mu (sample-mean anchor) + factor signal + GDP idiosyncratic state

and over 2024Q1–2026Q1 the vintage-level decomposition shows:

- **mu = +0.804** — the full-sample (2002+) GDP mean the model standardizes with. The
  model's no-data resting point is exactly +0.804; every quarter starts there. The
  current regime mean is +0.39 (2021+) / +0.26 (2024+).
- **Factor signal is correctly negative but small** (−0.1 to −0.6). The monthly
  indicators average z ≈ 0 over 2024–25 (ip_total −0.14, retail 0.00, foreign IP ≈ 0):
  the *trend* slowdown that halved GDP growth is invisible in MoM growth rates. Only
  sentiment reads it (esi_de −0.87) and sentiment loads weakly.
- **The GDP idiosyncratic state carries nothing forward**: its AR(1) is ρ = −0.86
  (oscillating, fast mean-reverting). Ex post the model explains each low 2024–25
  quarter with a fresh −0.2 to −0.9 one-off shock, then forgets it by the next quarter.

**Why the Kalman filter doesn't fix this:** the filter updates *states*; the mean is
not a state — it is a fixed standardization constant. A stationary DFM plus a shifted
mean is a misspecification: the filter is forced to file nine consecutive low quarters
as nine independent transitory shocks. No amount of filtering recovers what the state
space cannot represent.

## The fixes

**Fix 1 — local-level DFM** ([`src/dfm_locallevel.py`](src/dfm_locallevel.py)): add a
random-walk trend state μ_t to the GDP equation (state dim 10: factor AR(2) block +
trend block, both Mariano–Murasawa aggregated; trend block diffuse-initialized;
σ²_trend estimated by ML). Now the mean *is* a state and the filter can learn shifts.
The smoothed trend indeed falls +0.99 (2005) → +0.67 (2025), but σ²_trend ≈ 1.8e-5
(near the classic zero pile-up), so it adapts slowly.

**Fix 2 — post-regime estimation window** ([`src/model.py`](src/model.py)):
`load_processed(start="2013-01")` trims the estimation sample, dropping the pre-crisis
boom quarters from the anchor (mu falls +0.80 → +0.55).

## Real-time validation (src/realtime_anchor_fixes.py)

2024Q1–2026Q1, vintage inputs → flash target, expanding-window refit per quarter —
identical protocol to `outputs/realtime_2024_2026.csv`:

| variant | RMSE | bias | MAE |
|---|---|---|---|
| old panel (19 series) | 0.455 | +0.350 | 0.376 |
| new panel (23 series) | 0.410 | +0.322 | 0.351 |
| **new panel, 2013+ (fix 2)** | **0.295** | +0.143 | **0.224** |
| local level (fix 1) | 0.469 | +0.157 | 0.294 |
| local level + 2013+ (both) | 0.372 | **+0.133** | 0.261 |

**Verdict.** Both fixes remove most of the level bias (+0.32 → +0.13/+0.16). But the
local-level model pays for it in variance: one blow-up quarter (2025Q1: +1.43 vs flash
+0.19) drives its RMSE above the simpler fix. Three reasons: (i) σ²_trend pile-up makes
the trend estimate fragile per vintage (the 2013+ warm start drove it to exactly 0);
(ii) the explicit SSM has white-noise idiosyncratics, no AR(1) idio and no block
structure, so single noisy releases hit the GDP signal harder than in `DynamicFactorMQ`;
(iii) 51 ML parameters refit per vintage on a short window.

**Production spec: fix 2 — `DynamicFactorMQ`, 23-series panel, estimation window
2013+** (`base.load_processed(start="2013-01")`). Best RMSE (0.295, −35% vs old spec),
near-best bias (+0.14), and no new machinery. The local-level model stays in the repo
as the structurally correct answer — worth revisiting with a calibrated (fixed)
σ²_trend or AR(1) idiosyncratics if the regime keeps drifting.

## Series-dropping ablation (src/realtime_drop_series.py)

On top of the 2013+ spec, blocks flagged by BIAS.md / the loadings table were dropped
and re-run through the identical real-time protocol:

| variant | RMSE | bias | MAE |
|---|---|---|---|
| new_2013 (all 23 series) | 0.295 | +0.143 | 0.224 |
| **drop foreign IP** (ip_de, ip_de_auto, ip_ea) | 0.240 | **−0.012** | 0.182 |
| drop deadweight (bond_10y, eur_usd, hicp, cons_conf_sk) | 0.294 | +0.142 | 0.227 |
| drop both | 0.264 | −0.043 | 0.187 |
| drop both + imports_vol | **0.236** | +0.002 | 0.182 |

Dropping **foreign IP is the active ingredient**: it removes the remaining bias
entirely (+0.14 → −0.01) and cuts RMSE another 19% — consistent with BIAS.md, where
the German/EA block was the single largest overshoot driver. The deadweight series are
confirmed inert (identical scores); dropping imports_vol adds nothing beyond noise
level. Cost: 2024Q1 is undershot (+0.07 vs flash +0.67) — when the German chain
genuinely pulls Slovakia up, the signal is gone.

**Final production spec: DynamicFactorMQ, 20 series (23 minus foreign IP), 2013+
window.** RMSE 0.240, bias ≈ 0 over 2024Q1–2026Q1. Caveats: 9 evaluation quarters and
the variant was selected on this same window — treat the exact numbers as optimistic;
The early-quarter question was tested in the weekly pipeline
([`src/weekly_foreign_ip_test.py`](src/weekly_foreign_ip_test.py), point-in-time
Friday-by-Friday replay, same 2013+ window): dropping foreign IP wins in **every**
lead-time bucket, including month 1 (RMSE 0.334→0.267) — and the with-foreign-IP path
actually *deteriorates* mid-quarter (M2 0.423, M3 0.449 vs ~0.25–0.28 without) as
German IP prints land and inject decoupled signal. On 2024–2026 data there is no
early-quarter case for keeping the block either (`outputs/weekly_fip_test.{csv,png}`).
Caveat: this window is the decoupling regime; in a re-coupled cycle the block may
regain value — revisit if the German chain and Slovak GDP re-align.

*Generated from `src/anchor_decomposition*.py`, `src/dfm_locallevel.py`,
`src/realtime_anchor_fixes.py`, `src/realtime_drop_series.py`. Full per-quarter paths
in `outputs/realtime_anchor_fixes.csv` and `outputs/realtime_drop_series.csv`, trend
path in `outputs/ll_trend_path.csv`.*
