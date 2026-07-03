# Variable reference — Slovak GDP nowcasting panel

One entry per variable: source, exact dataset, transformation, sample, publication lag,
economic rationale, and current model status. Statuses refer to the production spec
(**estimation window 2013+, foreign IP excluded** — see `ANCHOR.md`); model statistics
(loading, signal-to-noise weight, |t|) are from the full-sample fit of that spec
(`outputs/loadings_best.csv`, SEs in `outputs/kalman_weights_monthly.xlsx`, sheet 3).
Loading signs: the factor's sign is arbitrary; activity series load together (negative
block), countercyclical series load opposite. Interpret magnitudes.

Real-time evaluation used throughout: 2024Q1–2026Q1, vintage inputs → flash target
(`outputs/realtime_lean_test.csv` and predecessors).

---

## Target

### `gdp_qoq` — Real GDP, chain-linked volumes, SCA, QoQ growth
- **Source:** Eurostat `namq_10_gdp` (B1GQ / CLV_PCH_PRE / SCA / SK); real-time vintages + flash from OECD STES revisions DB (`B1GQ_Q`).
- **Transform:** none (already QoQ %). **Sample:** 1995Q2–present (124 q). **Pub lag:** flash ~45 d after quarter end.
- **Rationale:** target. Estimation from 2013Q1 — pre-2013 quarters contain a structurally higher-growth regime (mean +0.96 pre-2019 vs +0.40 post-2021) that biased the nowcast up ~0.2pp (see `ANCHOR.md`).

## Domestic hard data (in best model)

### `ip_total` — Industrial production, total industry (B–D), SCA
- **Source:** Eurostat `sts_inpr_m` (B-D/SCA/I21); vintages from OECD (`PRVM`/BTE). **Transform:** Δlog×100. **Sample:** 2000-01–present (~316 m). **Pub lag:** ~40 d (2 m in model).
- **Model:** loading −0.31, s/n 0.12, |t| 0.8.
- **Rationale:** the classic coincident indicator; industry ≈ 35 % of Slovak GVA.
- **Status: borderline.** Overlaps `ip_manuf` (manufacturing ≈ ¾ of total industry) — the two share idiosyncratic shocks, which the DFM's diagonal-idio assumption dislikes; dropping `ip_total` gave the best lean-panel RMSE (0.230 vs 0.238), inside noise but directionally consistent with the double-counting concern.

### `ip_manuf` — Industrial production, manufacturing (C), SCA
- Same source/transform/lag as `ip_total`. **Model:** loading −0.33, s/n 0.14, |t| 1.0.
- **Rationale:** the export-oriented cyclical core (autos and supply chain); the single most informative hard monthly series. **Status: keep.**

### `retail_vol` — Retail trade volume (G47), SCA
- **Source:** Eurostat `sts_trtu_m`; OECD vintages (`TOVM`/G47). **Transform:** Δlog×100. **Sample:** 2000-01–present. **Pub lag:** ~40 d.
- **Model:** loading −0.24, s/n 0.07. **Rationale:** timeliest hard read on household consumption (~59 % of GDP). **Status: keep.**

### `construction` — Construction production (F), SCA
- **Source:** Eurostat `sts_copr_m`. **Transform:** Δlog×100. **Sample:** 2000-01–present. **Pub lag:** ~45 d.
- **Model:** loading −0.08, s/n 0.006 — weakest hard series; long project pipelines and weather noise decouple it from the quarterly cycle.
- **Status: dropped in lean14** (with `services_J`): real-time RMSE unchanged (0.229 vs 0.230, `outputs/realtime_lean14.csv`). Construction is 13 % of GVA but contributes no extractable cycle.

### `exports_vol` / `imports_vol` — Goods trade volume indices, world, SA
- **Source:** Eurostat `ei_eteu27_2020_m` (EXP/IMP, IVOL-SA). **Transform:** Δlog×100. **Sample:** 2002-01–present (bounds the panel start). **Pub lag:** ~40 d.
- **Model:** loadings −0.30/−0.33, s/n 0.11/0.16 — the top-weighted block after foreign IP was dropped.
- **Rationale:** exports ≈ 85 % of GDP; with foreign IP excluded, Slovak customs flows are how the model reads external demand. Not a total-and-part pair — imports carry independent signal on domestic demand and re-export inputs. **Status: keep both.**

### `unemp_rate` — Unemployment rate, SA, first difference
- **Source:** Eurostat `une_rt_m`; OECD vintages (`UNEMP`). **Transform:** Δ (pp). **Sample:** 1998-01–present. **Pub lag:** ~30 d (1 m) — fastest hard series.
- **Model:** loading +0.23 (countercyclical, correct sign), s/n 0.07, |t| 0.3.
- **Status: keep — tested against employment and won.** Replacing it with monthly employment growth (DATAcube `od0007ms`, avg YoY across 5 sectors) worsened the real-time test (RMSE 0.257 vs 0.238; bias +0.06): employment is a smooth, lagging stock, and it already enters the panel inside `real_wage_bill` (same cube), so swapping adds redundancy while losing the timelier rate signal. **Vacancies** (growth of job vacancies — a leading flow) would be the better labour upgrade, but no monthly public API: Eurostat `jvs_q_nace2` is quarterly for SK; monthly vacancy counts exist only in ÚPSVaR (labour office) XLS reports — a manual-scrape candidate alongside VAT receipts.

## Services & income block (panel expansion, 2026-07 session)

### `services_H` / `services_J` / `services_N` — Turnover: transport (H), ICT (J), admin. services (N), SCA
- **Source:** Eurostat `sts_setu_m` (NETTUR/I21; nominal — Eurostat publishes no volume series for SK services). **Transform:** Δlog×100 **minus HICP inflation** (`dlog_real`) — deflated to real growth since 2026-07; previously entered nominal, overstating activity in the 2022–23 inflation burst. **Sample:** 2000-01–present. **Pub lag:** ~60 d (2 m).
- **Model:** H loads well (−0.26, s/n 0.09 — freight tracks the cycle); J and N barely (−0.04/−0.14) — contract-revenue sectors, weakly cyclical.
- **Rationale:** broaden services coverage (services > 55 % of GVA vs one series before). Enabled scoring 5 sectors against GVA shares (`outputs/weights_vs_shares_best.csv`).
- **Status: H keep; J dropped in lean14** (s/n 0.1 %, no real-time cost); **N kept as border case** (s/n 0.6 %). Dropping J and construction left RMSE unchanged (0.229).

### `services_iaf` — Accommodation & food turnover (I), SCA
- Same source class; nominal at source, **HICP-deflated (`dlog_real`) since 2026-07**. **Sample:** 2000-01–present. **Model:** loading −0.20, s/n 0.04.
- **Rationale:** discretionary household demand. **Status: keep.**

### `vacancies` — Job vacancies, SA count, NACE B–T (quarterly)
- **Source:** Eurostat `jvs_q_r21` (SK / SA / JOBVAC / TOTAL size class / B-T). **Transform:** Δlog×100 (QoQ growth). **Sample:** 2008Q1–present (73 q). **Pub lag:** ~2 months after quarter end; enters as a **second quarterly variable** beside GDP (`DynamicFactorMQ` `endog_quarterly`), observed through target−1 at nowcast time. No vintage history — latest values used.
- **Rationale:** labour-*demand* flow, the leading cyclical sensor the panel lacked (surveys were the only forward-looking block). Vacancies fell 18 % over 2025 — the cooling signal, confirmed. Real-time 2024–26: RMSE 0.227 vs 0.229 without — neutral on a calm window; its value case is turning points. **Status: adopted.** (Monthly vacancies exist only as ÚPSVaR XLS — still the eventual upgrade.)

### `real_wage_bill` — Real wage-bill growth, YoY %
- **Source:** built (`src/fetch_new_data.py`): ŠÚ SR DATAcube avg nominal wage (`od0008ms`) × employment (`od0007ms`), 5 sectors, deflated by HICP. **Transform:** none (already a YoY growth rate). **Sample:** 2011-01–2025-12 (cube publishes with a lag). **Pub lag:** ~2 m.
- **Model:** loading −0.20 with **|t| = 6.1 — the only individually significant loading in the panel**; s/n 0.05.
- **Rationale:** the income side of GDP, previously absent; the domestic-demand counterweight motivated by the 2025 foreign-IP bias (`BIAS.md`). **Status: keep.**

## Sentiment (in best model)

### `esi_sk`, `ind_conf_sk`, `cons_conf_sk` — Slovak ESI, industrial & consumer confidence, SA
- **Source:** Eurostat `ei_bssi_m_r2` (DG ECFIN surveys). **Transform:** level. **Sample:** 1993-08–present. **Pub lag:** ~0 d (month-end).
- **Model:** loadings −0.27/−0.27/−0.11.
- **Rationale:** zero-lag coverage of the ragged edge — at the end of the quarter surveys are the only fresh data; sentiment was also the only block that read the 2024–25 slowdown (z ≈ −0.3 while hard data sat at z ≈ 0). **Status: keep.**

### `esi_de`, `esi_ea` — German / euro-area ESI, SA
- Same source/lag. **Model:** loadings −0.24/−0.25; `esi_de` carries the largest cumulative Kalman weight in the panel (|w| 0.94, mostly sign-alternating: the filter reads *changes*).
- **Rationale:** external demand at zero lag — the surviving Germany channel after foreign IP was dropped (survey expectations decoupled less than hard IP did in 2024–25). **Status: keep.**

## Dropped from the best model

### `ip_de`, `ip_de_auto`, `ip_ea` — German/EA industrial production, German motor vehicles (C29)
- **Source:** Eurostat `sts_inpr_m`. **Sample:** 1991-01–present. **Pub lag:** ~40 d.
- **Why dropped:** the single largest source of the 2024–26 over-prediction. German industry held up while Slovak domestic demand decoupled; the block kept injecting phantom strength (`BIAS.md`, `ANCHOR.md`). Dropping it: real-time RMSE 0.295 → 0.240, bias +0.14 → −0.01; the weekly A/B showed it hurts at *every* lead time, including month 1 (`outputs/weekly_fip_test.csv`). **Revisit if the German chain and Slovak GDP re-couple.**

### `hicp` — Consumer prices (all-items), monthly inflation
- **Source:** Eurostat `prc_hicp_minr` (ECOICOP v2, `TOTAL`/`I15`) — **`prc_hicp_midx` was frozen by Eurostat at 2025-12** and silently stopped updating; source switched 2026-07. **Sample:** 1996-12–present. **Pub lag:** ~17 d.
- **Why dropped as an indicator:** zero factor weight (loading 0.003, s/n 0.000) — monthly inflation carries no Slovak real-cycle signal. Still fetched: deflates `real_wage_bill` and the services turnover block.

### `eur_usd` — EUR/USD monthly average
- **Source:** FRED `DEXUSEU`. **Sample:** 1999-01–present. **Pub lag:** ~0 d.
- **Why dropped:** loading 0.009, no weight. The euro moves on ECB/Fed news, not the Slovak cycle.

### `bond_10y` — Slovak 10Y government bond yield, first difference
- **Source:** Eurostat `irt_lt_mcby_m`. **Sample:** 2001-01–present. **Pub lag:** ~10 d.
- **Why dropped:** loading 0.13 but s/n 0.017 and near-zero real-time value; yield changes reflect euro-area duration, not Slovak activity.

### `emp_total` — Monthly employment growth, avg YoY % (5 sectors)
- **Source:** ŠÚ SR DATAcube `od0007ms` (fetched, kept in `data/raw/emp_total.csv`). **Sample:** 2010-01–present. **Pub lag:** ~2 m.
- **Why not used:** tested as an `unemp_rate` replacement — worse (RMSE 0.257 vs 0.238), lagging-stock dynamics + duplicates the employment input inside `real_wage_bill`.

## Named in the research doc, excluded for access (see `DATA_CATALOGUE.md`)

VAT receipts (XLS only), new car registrations (no MoI API), electricity load (ENTSO-E
token), German truck-toll index (GENESIS token), SIPS interbank payments (NBS-internal),
health/social-insurance contributions (admin micro-data), Ifo/ZEW (licensed), monthly
job vacancies (ÚPSVaR XLS — flagged above as the preferred labour-market upgrade).

## Totals-and-parts policy

The panel avoids aggregate/component pairs with one exception. `ip_total` ⊃ `ip_manuf`
(~¾ overlap) — originally kept because total adds mining/energy while manufacturing
isolates the export core; but overlapping aggregates share idiosyncratic shocks,
violating the DFM's diagonal-idiosyncratic assumption and tilting weight toward
industry (industry weight 43 % vs 35 % GVA share). The lean test (drop `ip_total`)
gave the best RMSE (0.230) — within noise on 9 quarters, but the structural argument
and the test point the same way. Exports/imports are two flows, not a total and part;
the services series (G47, H, I, J, N) are disjoint NACE sections; `real_wage_bill`
overlaps the labour block only through employment, which is why `emp_total` stays out.

---
## BASELINE spec (frozen 2026-07) — `src/baseline.py`

14 monthly series, estimation 2013+: `ip_manuf`, `retail_vol`, `exports_vol`,
`imports_vol`, `unemp_rate`, `services_H`, `services_iaf`, `services_N` (all services
HICP-deflated), `real_wage_bill`, `esi_sk`, `ind_conf_sk`, `cons_conf_sk`, `esi_de`,
`esi_ea`; quarterly: `gdp_qoq` (target) + `vacancies`.
Real-time 2024Q1–2026Q1: RMSE 0.227, bias +0.03 (`outputs/realtime_real_vac.csv`).
Real-time 2024Q1–2026Q1: RMSE 0.229, bias +0.03 (vs 0.455 / +0.35 at the session
start). Selection caveat: variants were chosen on this same 9-quarter window — treat
lean14 ≈ lean16 ≈ lean17 ≈ best20 as equivalent on score; lean14 wins on parsimony.
Watch: sentiment now carries the majority of the effective filter weight — if surveys
decouple from activity, revisit the balance before pruning anything else.

---
*Real-time scores: `outputs/realtime_lean14.csv`, `outputs/realtime_lean_test.csv`,
`outputs/realtime_drop_series.csv`, `outputs/realtime_anchor_fixes.csv`,
`outputs/realtime_2024_2026.csv`. Generated 2026-07; regenerate stats via
`src/results_best.py`, `src/kalman_weights_excel.py`, `src/realtime_lean_test.py`.*
