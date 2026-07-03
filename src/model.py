"""
Part 2b - MFDFM estimation, nowcast, and pseudo-real-time backtest.

Uses statsmodels DynamicFactorMQ (Kalman filter + EM, native mixed monthly/quarterly
frequency and ragged-edge handling) to:
  1. estimate a single-factor dynamic factor model on the monthly panel + quarterly GDP,
  2. produce the current-quarter GDP nowcast,
  3. run a fixed-parameter pseudo-real-time backtest vs random-walk and AR(1) benchmarks,
  4. save factor loadings and signal-to-noise weights for the Part 3 interpretation.

Outputs: outputs/nowcast.csv, outputs/loadings.csv, outputs/backtest.csv
"""
from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
PROC = BASE / "data" / "processed"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

FACTOR_ORDERS = 2  # AR(2) on the common factor

# Publication lag in months used to reconstruct data vintages for the backtest:
# how many of the most recent monthly observations are NOT yet available when the
# nowcast is made at the end of the target quarter (see DATA_CATALOGUE.md).
PUB_LAG = {
    "ip_total": 2, "ip_manuf": 2, "retail_vol": 2, "construction": 2,
    "exports_vol": 2, "imports_vol": 2, "unemp_rate": 1,
    "ip_de": 2, "ip_de_auto": 2, "ip_ea": 2,
    "services_iaf": 2, "hicp": 1,
    "services_H": 2, "services_J": 2, "services_N": 2, "real_wage_bill": 2,
    "esi_sk": 0, "ind_conf_sk": 0, "cons_conf_sk": 0, "esi_de": 0, "esi_ea": 1,
    "bond_10y": 0, "eur_usd": 0,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_processed(start: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """`start` (e.g. "2013-01") trims the estimation sample — the regime-shift fix
    validated in outputs/realtime_2024_2026.csv (bias +0.35 -> +0.14 vs flash)."""
    m = pd.read_csv(PROC / "monthly_panel.csv", parse_dates=["date"]).set_index("date")
    m.index = m.index.to_period("M")
    g = pd.read_csv(PROC / "gdp_quarterly.csv", parse_dates=["date"]).set_index("date")["gdp_qoq"]
    g.index = g.index.to_period("Q")
    if start is not None:
        m = m.loc[m.index >= pd.Period(start, "M")]
        g = g.loc[g.index >= pd.Period(start, "M").asfreq("Q")]
    # Extend quarterly index to cover the last monthly quarter (the nowcast target).
    qrange = pd.period_range(g.index.min(), m.index.max().asfreq("Q"), freq="Q")
    g = g.reindex(qrange)
    return m, g


def build_model(m: pd.DataFrame, g: pd.Series) -> DynamicFactorMQ:
    return DynamicFactorMQ(
        m, endog_quarterly=g.to_frame(), factors=1,
        factor_orders=FACTOR_ORDERS, idiosyncratic_ar1=True,
    )


def gdp_nowcast(res, target_q: pd.Period) -> float:
    """Model-implied GDP QoQ growth for `target_q`, read at its quarter-end month."""
    pm = res.predict()
    qend_m = target_q.asfreq("M", how="end")
    return float(pm.loc[qend_m, "gdp_qoq"])


# ---------------------------------------------------------------------------
# Loadings / signal-to-noise weights
# ---------------------------------------------------------------------------
def extract_loadings(res) -> pd.DataFrame:
    p = res.params
    rows = []
    for name in res.model.endog_names:
        lo = p[f"loading.0->{name}"]
        s2 = p[f"sigma2.{name}"]
        ar = p.get(f"L1.eps_M.{name}", p.get(f"L1.eps_Q.{name}", 0.0))
        idio_level_var = s2 / (1 - ar ** 2) if abs(ar) < 1 else np.nan
        rows.append(dict(series=name, loading=lo, sigma2=s2, idio_ar1=ar,
                         idio_level_var=idio_level_var,
                         sn_weight=lo ** 2 / idio_level_var))
    df = pd.DataFrame(rows).set_index("series")
    return df


# ---------------------------------------------------------------------------
# Pseudo-real-time backtest (fixed parameters)
# ---------------------------------------------------------------------------
def make_vintage(m: pd.DataFrame, g: pd.Series, target_q: pd.Period):
    """Data as available at the END of `target_q`: apply publication lags to the
    monthly panel and drop GDP for `target_q` and later."""
    ref_m = target_q.asfreq("M", how="end")
    mv = m.loc[:ref_m].copy()
    for col, lag in PUB_LAG.items():
        if lag > 0:
            cutoff = ref_m - lag
            mv.loc[mv.index > cutoff, col] = np.nan
    gv = g.copy()
    gv[gv.index >= target_q] = np.nan
    gv = gv.loc[:target_q]
    return mv, gv


def backtest(m: pd.DataFrame, g: pd.Series, params, start_q="2010Q1") -> pd.DataFrame:
    actual = g.dropna()
    test_qs = [q for q in actual.index if q >= pd.Period(start_q, "Q")]
    records = []
    for q in test_qs:
        mv, gv = make_vintage(m, g, q)
        try:
            res_v = build_model(mv, gv).smooth(params)
            nc = gdp_nowcast(res_v, q)
        except Exception:  # noqa: BLE001
            nc = np.nan
        prev = actual.loc[:q - 1]
        rw = prev.iloc[-1] if len(prev) else np.nan          # random walk
        ar = ar1_forecast(prev)                               # expanding AR(1)
        records.append(dict(quarter=str(q), actual=actual.loc[q],
                            dfm=nc, rw=rw, ar1=ar))
    return pd.DataFrame(records).set_index("quarter")


def ar1_forecast(y: pd.Series) -> float:
    y = y.dropna()
    if len(y) < 8:
        return float(y.mean()) if len(y) else np.nan
    y0, y1 = y.values[:-1], y.values[1:]
    b = np.polyfit(y0, y1, 1)  # slope, intercept
    return float(b[0] * y.values[-1] + b[1])


def rmse(a, b) -> float:
    d = (pd.Series(a).astype(float) - pd.Series(b).astype(float)).dropna()
    return float(np.sqrt((d ** 2).mean()))


# ---------------------------------------------------------------------------
def main() -> None:
    m, g = load_processed()
    target_q = g.index[-1]  # last (all-NaN) quarter = nowcast target
    print(f"Panel: {m.shape[0]} months x {m.shape[1]} series | "
          f"GDP observed through {g.dropna().index[-1]} | nowcast target = {target_q}\n")

    print("Estimating DynamicFactorMQ (1 factor, AR(2), AR(1) idiosyncratic)...")
    res = build_model(m, g).fit(disp=0, maxiter=200)
    n_iter = res.mle_retvals.get("iterations", res.mle_retvals.get("iter", "n/a"))
    print(f"  log-likelihood = {res.llf:.1f}, EM iterations = {n_iter}\n")

    # --- current-quarter nowcast ---
    nc = gdp_nowcast(res, target_q)
    prev_actual = g.dropna().iloc[-1]
    pd.DataFrame([dict(target_quarter=str(target_q), nowcast_qoq_pct=round(nc, 3),
                       prev_quarter=str(g.dropna().index[-1]),
                       prev_actual_qoq_pct=round(prev_actual, 3))]
                 ).to_csv(OUT / "nowcast.csv", index=False)
    print(f"NOWCAST {target_q}: real GDP QoQ growth = {nc:+.2f}%  "
          f"(previous actual {g.dropna().index[-1]} = {prev_actual:+.2f}%)\n")

    # --- loadings / weights ---
    loads = extract_loadings(res)
    loads.to_csv(OUT / "loadings.csv")
    print("Top factor loadings (|loading|):")
    print(loads["loading"].abs().sort_values(ascending=False).head(6).round(3).to_string(), "\n")

    # --- backtest ---
    print("Pseudo-real-time backtest (fixed parameters, 2010Q1-onward)...")
    bt = backtest(m, g, res.params)
    bt.to_csv(OUT / "backtest.csv")
    r_dfm, r_rw, r_ar = rmse(bt.dfm, bt.actual), rmse(bt.rw, bt.actual), rmse(bt.ar1, bt.actual)
    print(f"  n = {bt.dfm.notna().sum()} quarters")
    print(f"  RMSE  DFM   = {r_dfm:.3f}")
    print(f"  RMSE  RW    = {r_rw:.3f}   (DFM/RW  = {r_dfm/r_rw:.2f})")
    print(f"  RMSE  AR(1) = {r_ar:.3f}   (DFM/AR1 = {r_dfm/r_ar:.2f})")
    verdict = "beats" if r_dfm < min(r_rw, r_ar) else "does NOT beat"
    print(f"  => DFM {verdict} both benchmarks.")


if __name__ == "__main__":
    main()
