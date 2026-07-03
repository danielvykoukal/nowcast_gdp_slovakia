# Paper-faithful replication — comparison with the current model

Implements the three methodology mismatches flagged against Camacho & Perez-Quiros (2008)
in [`src/dfm_paper.py`](src/dfm_paper.py), as an **ablation** so each feature's effect on the
nowcast is isolated. All models use the identical fixed-parameter pseudo-real-time backtest
protocol as [`src/dfm_statespace.py`](src/dfm_statespace.py) (2010Q1+, 65 quarters,
end-of-quarter information set), so RMSE is directly comparable to the current **0.739**.

> **Vintage caveat.** To match the current model's stated 0.739, the backtest reuses its exact
> `PUB_LAG` dict, which *omits* the German/euro-area IP, services-IAF and HICP series — so those
> six are left unlagged (present up to the reference month). That is optimistic: in real time
> German IP lags ~2 months. Applying honest lags to all series raises **every** row's RMSE by a
> similar amount (C0 ≈ 0.85 ex-2020), so it does not change the *ranking* below, but the current
> **0.739 headline is itself mildly overstated**.

## The three mismatches implemented

1. **AR(1) idiosyncratic in-state** (paper eq. 8-11) — the current `dfm_statespace.py` used
   white-noise idiosyncratic and its `METHODOLOGY.md` wrongly claimed that was "exactly as the
   PDF specifies". The paper's whole contribution vs Angelini/Banbura is *non-white*
   idiosyncratic dynamics. The GDP idiosyncratic is Mariano-Murasawa aggregated (eq. 7).
2. **Soft indicators → year-on-year loading** (paper eq. 7): surveys load on
   `sum_{j=0..11} f_{t-j}` (12 factor lags), not the single current factor.
3. **Multiple GDP releases** (paper eq. 5-6): a real-time **flash** row (built from OECD
   vintages, n=143 quarters) shares the factor+idio signal with the **final**
   value and differs by a white revision noise.

## Ablation results (nowcast 2026Q2, RMSE vs final GDP)

| Config | # params | Nowcast (QoQ%) | RMSE (all) | RMSE (ex-2020) |
|---|---|---|---|---|
| *Current model (`dfm_statespace.py`, reference)* | ~40 | +0.32 | **0.739** | — |
| C0 current-equiv (scalar-R WN idio) | 42 | +0.32 | 0.730 | 0.658 |
| C1 + MM-aggregated idio in-state (eq.7) | 42 | +0.32 | 0.728 | 0.642 |
| C2 + AR(1) idio persistence (eq.8-11) | 62 | +0.60 | 0.767 | 0.692 |
| C3 + soft YoY loading (eq.7) | 62 | +0.44 | 0.752 | 0.677 |
| C4 + multi-release flash+final [FULL] | 63 | +0.44 | 0.752 | 0.677 |

**C0** reproduces the current model (scalar white-noise idiosyncratic as measurement error);
0.730/0.658 ≈ the reference 0.739/0.662. Each later row *adds* one paper feature on top.

### Verdict — does paper-faithfulness improve the nowcast? At equal information, no.

- **C1 (MM-aggregated idiosyncratic, eq.7)** — the one unambiguous structural win, but tiny:
  0.642 vs 0.658 ex-2020. The paper's quarterly-noise aggregation is marginally correct.
- **C2 (AR(1) idiosyncratic persistence, eq.8-11)** — log-likelihood leaps (−7148 → −5437) but
  out-of-sample RMSE **worsens** to 0.692 and the nowcast swings to +0.60: a textbook overfit.
  19 extra persistence parameters do not pay on a 65-quarter sample. **This directly refutes the
  earlier `METHODOLOGY.md` claim** — the paper's non-white idiosyncratic is *specified* by the
  PDF, but on Slovak data it hurts out-of-sample.
- **C3 soft-YoY / C4 multi-release (same timing)** — neutral. Matches the paper's own conclusion
  that surveys and early releases help *timeliness*, not in-sample fit.

The best same-information model (C1) beats the current one only at the third decimal. The paper's
full spec (C4) is *worse* at equal information (0.677 vs 0.658) because AR(1) overfitting
outweighs the MM-idio gain.

## Real-time flash channel (FULL model)

Under the same protocol the flash is not yet published at the end-of-quarter information set, so
the multi-release row helps only through modelling historical revisions. When the nowcast is
instead made once the **flash is available for the target quarter** (its true real-time timing,
~6-10 weeks after quarter end), accuracy improves to **RMSE all=0.617, ex-2020=0.574** —
this is the paper's point that early releases carry genuine within-quarter information.

The flash-timing gain is real but comes from **earlier information** (a t+6-10-week GDP print),
not a smarter model — a fair same-information comparison does not credit it. Still, in practice
the nowcast *is* remade when the flash lands, so 0.574 is the operationally relevant number.

## Overall conclusion

| Question | Answer |
|---|---|
| Do the paper features beat the current model at equal information? | **No** — neutral to slightly worse (AR(1) overfits). |
| Which single feature helps? | MM-aggregated idiosyncratic (C1), marginally. |
| Where is the real gain? | The **flash real-time channel**: 0.574 ex-2020 vs 0.662, ~13% lower — but that is extra information. |
| Was the old `METHODOLOGY.md` right that white-noise idio is "exactly as the PDF specifies"? | **No.** The PDF specifies AR(1) idio; the doc claim was backwards. But AR(1) also *hurts* out-of-sample here, so white-noise was the better modelling choice for the wrong stated reason. |

Net: keep the simple robust `dfm_statespace.py` / `model_v2.py` for accuracy; the value of this
exercise is (1) fixing the false methodology claim, (2) adding the real-time flash channel, and
(3) exposing that the 0.739 headline leans on several unlagged hard indicators.

*Generated by `src/dfm_paper.py`; narrative curated post-run.*
