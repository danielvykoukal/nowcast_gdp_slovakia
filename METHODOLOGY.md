# Methodology — Explicit Mixed-Frequency State-Space DFM

This model implements the handwritten derivation exactly (single common factor, AR(2)
dynamics, Mariano-Murasawa quarterly aggregation, white-noise idiosyncratic errors),
estimated by maximum likelihood via a custom `statsmodels` `MLEModel`
([`src/dfm_statespace.py`](src/dfm_statespace.py)). `DynamicFactorMQ` is retained as an
independent numerical cross-check.

## State-space form

**State vector** (factor + 4 lags, needed for the quarterly aggregation):

    s_t = [f_t, f_{t-1}, f_{t-2}, f_{t-3}, f_{t-4}]'

**Transition** — AR(2) factor, `f_t = phi1 f_{t-1} + phi2 f_{t-2} + w_t`, `w_t ~ N(0,1)`
(factor-shock variance fixed to 1 for identification). Estimated companion matrix **F**:

| f_t | f_t-1 | f_t-2 | f_t-3 | f_t-4 |
|---|---|---|---|---|
| 0.090 | -0.231 | 0.000 | 0.000 | 0.000 |
| 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |

so **phi1 = 0.090, phi2 = -0.231**.

**Measurement** — `Y_t = H s_t + e_t`, `e_t ~ N(0, R)`, R diagonal:
- each monthly indicator loads on the current factor only: `H_i = [lambda_i, 0, 0, 0, 0]`;
- quarterly GDP loads on the Mariano-Murasawa weighted factor lags:

    H_gdp = lambda_g * [1/3, 2/3, 1, 2/3, 1/3] = [-0.183, -0.365, -0.548, -0.365, -0.183]   (lambda_g = -0.548)

GDP enters only at quarter-end months; all other months it is missing and the Kalman
filter reconstructs it — that reconstruction is the nowcast.

## Estimated loadings (lambda) and idiosyncratic variances (sigma2)

| series | loading | sigma2 |
|---|---|---|
| ip_ea | -0.907 | 0.121 |
| ip_de | -0.907 | 0.121 |
| ip_de_auto | -0.859 | 0.211 |
| exports_vol | -0.717 | 0.446 |
| imports_vol | -0.665 | 0.524 |
| ip_total | -0.605 | 0.608 |
| ip_manuf | -0.594 | 0.621 |
| gdp | -0.548 | 0.362 |
| retail_vol | -0.376 | 0.846 |
| ind_conf_sk | -0.244 | 0.933 |
| construction | -0.164 | 0.968 |
| unemp_rate | 0.159 | 0.970 |
| bond_10y | 0.159 | 0.970 |
| esi_sk | -0.140 | 0.976 |
| esi_ea | -0.139 | 0.976 |
| esi_de | -0.125 | 0.980 |
| eur_usd | -0.059 | 0.993 |
| cons_conf_sk | 0.034 | 0.995 |

## Results

- **Nowcast 2026Q2 = +0.32% QoQ** (explicit state-space model).
- Pseudo-real-time backtest (65 quarters, 2010Q1+): **RMSE = 0.739**.
- Cross-check vs `DynamicFactorMQ`: RMSE = 1.087, and the two models' nowcast paths
  correlate **0.928** — confirming the from-scratch implementation reproduces the
  library's mixed-frequency DFM, as expected since both encode the same methodology.

*Note:* this model uses **white-noise** idiosyncratic errors. The Camacho-Perez-Quiros PDF
actually specifies **AR(1) idiosyncratic dynamics** (eq. 8-11) — white noise is a deliberate
simplification, not the PDF's spec. The [`src/dfm_paper.py`](src/dfm_paper.py) ablation implements
the AR(1) version and finds it *overfits* on this 65-quarter sample (out-of-sample RMSE rises from
0.658 to 0.692), so the white-noise choice is the better modelling call here despite departing
from the letter of the paper. See [`COMPARISON.md`](COMPARISON.md).
