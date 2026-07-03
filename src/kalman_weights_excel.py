"""
Excel workbook of the Kalman filter's dataset weights, month by month, best model
(20 series, 2013+, no foreign IP). outputs/kalman_weights_monthly.xlsx, 3 sheets:

  1. weights_by_month  - rows = information-set months 2024-01..2026-06. Each month
     targets the LIVE quarter (earliest quarter whose flash, ~2 months after quarter
     end, is not yet out). Cell = cumulative signed Koopman-Harvey weight of that
     dataset in the smoothed GDP forecast (standardized scale: gdp-sd per 1-sd obs).
     In the month a flash is released the target's GDP weight = 1, all else 0 - the
     filter simply reproduces the released number.
  2. decomposition     - same rows; cell = contribution in pp of GDP QoQ
     (weight x observed standardized value x gdp sd). constant = mean anchor + prior;
     rows sum to the nowcast column.
  3. loadings          - factor loadings with standard errors, idio AR(1), variances,
     signal-to-noise weights from the full-sample fit.

Parameters fixed at the full-sample fit; each month re-smoothed on that month's
publication-lag vintage (latest data values, not OECD editions - this is a weights
exercise, not a vintage replay).

Run:  python3 src/kalman_weights_excel.py
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.tools import compute_smoothed_state_weights

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
OUT = BASE / "outputs"

import model as base

FOREIGN = ["ip_de", "ip_de_auto", "ip_ea"]
MONTHS = pd.period_range("2024-01", "2026-06", freq="M")
FLASH_LAG_M = 2  # flash ~45 days after quarter end -> 2nd month of next quarter


def flash_month(q: pd.Period) -> pd.Period:
    return q.asfreq("M", how="end") + FLASH_LAG_M


def target_of(month: pd.Period) -> pd.Period:
    q = (month - 12).asfreq("Q")
    while flash_month(q) < month:
        q += 1
    return q


def vintage(m_full: pd.DataFrame, g_full: pd.Series, month: pd.Period, tq: pd.Period):
    mv = m_full.loc[:month].copy()
    for col, lag in base.PUB_LAG.items():
        if col in mv.columns and lag > 0:
            mv.loc[mv.index > month - lag, col] = np.nan
    gv = g_full.copy()
    gv[[q for q in gv.index if flash_month(q) > month]] = np.nan
    end_q = max(tq, month.asfreq("Q"))
    gv = gv.reindex(pd.period_range(gv.index.min(), end_q, freq="Q"))
    return mv, gv


def main():
    m_full, g_full = base.load_processed(start="2013-01")
    m_full = m_full.drop(columns=FOREIGN)
    series = list(m_full.columns)

    print("Full-sample fit (params + standard errors)...", flush=True)
    res_full = base.build_model(m_full, g_full).fit(disp=0, maxiter=200)

    w_rows, c_rows = [], []
    for month in MONTHS:
        tq = target_of(month)
        mv, gv = vintage(m_full, g_full, month, tq)
        mod = base.build_model(mv, gv)
        res = mod.smooth(res_full.params)
        t_end = mv.index.get_loc(tq.asfreq("M", how="end"))

        w = compute_smoothed_state_weights(res, compute_t=[t_end])
        weights = np.nan_to_num(w[0] if isinstance(w, tuple) else w)[0]  # (nobs,k_states,k_endog)
        names = list(res.model.endog_names)
        Zg = np.asarray(res.model["design"])[names.index("gdp_qoq"), :]
        Wmat = np.einsum("s,jsk->jk", Zg, weights)                       # (nobs, k_endog)

        y_std = np.nan_to_num(np.asarray(res.model.endog, dtype=float))
        g_sd = float(gv.dropna().std())
        cum_w = pd.Series(Wmat.sum(axis=0), index=names)                  # signed, std scale
        contrib = pd.Series((Wmat * y_std).sum(axis=0) * g_sd, index=names)  # pp

        nc = float(res.predict().iloc[t_end]["gdp_qoq"])
        released = flash_month(tq) == month
        w_rows.append({"month": str(month), "target_quarter": str(tq),
                       "released": released, **cum_w.round(4).to_dict()})
        c_rows.append({"month": str(month), "target_quarter": str(tq),
                       "nowcast_pp": round(nc, 3),
                       "constant_pp": round(nc - contrib.sum(), 3),
                       **contrib.round(4).to_dict()})
        print(f"  {month} -> {tq}{' [flash released]' if released else ''}: "
              f"nowcast {nc:+.2f}, gdp weight {cum_w['gdp_qoq']:+.3f}", flush=True)

    wdf = pd.DataFrame(w_rows).set_index("month")[["target_quarter", "released", "gdp_qoq"] + series]
    cdf = pd.DataFrame(c_rows).set_index("month")[
        ["target_quarter", "nowcast_pp", "constant_pp", "gdp_qoq"] + series]

    # sheet 3: loadings + standard errors
    loads = base.extract_loadings(res_full)
    bse = None
    try:
        bse = res_full.bse
    except Exception as e:  # noqa: BLE001
        print(f"standard errors unavailable ({e})")
    ldf = loads.copy()
    if bse is not None:
        ldf["loading_se"] = [bse.get(f"loading.0->{s}", np.nan) for s in ldf.index]
        ldf["loading_t"] = ldf["loading"] / ldf["loading_se"]
    ldf = ldf[["loading"] + (["loading_se", "loading_t"] if bse is not None else [])
              + ["sigma2", "idio_ar1", "idio_level_var", "sn_weight"]].round(4)

    xls = OUT / "kalman_weights_monthly.xlsx"
    with pd.ExcelWriter(xls, engine="openpyxl") as xw:
        wdf.to_excel(xw, sheet_name="weights_by_month")
        cdf.to_excel(xw, sheet_name="decomposition")
        ldf.to_excel(xw, sheet_name="loadings")
    print(f"\nWrote {xls}")


if __name__ == "__main__":
    main()
