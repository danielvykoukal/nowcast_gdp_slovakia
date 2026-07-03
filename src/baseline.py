"""
BASELINE MODEL — the production spec, frozen 2026-07-03. Single source of truth.

Spec ("lean14 + vacancies, real services"):
  * 14 monthly series (services HICP-deflated in preprocess.py):
      ip_manuf, retail_vol, exports_vol, imports_vol, unemp_rate,
      services_H, services_iaf, services_N, real_wage_bill,
      esi_sk, ind_conf_sk, cons_conf_sk, esi_de, esi_ea
  * quarterly block: gdp_qoq (target) + vacancies (jvs_q_r21, dlog QoQ)
  * estimation window 2013Q1+ (regime fix, ANCHOR.md)
  * DynamicFactorMQ: 1 factor, AR(2), AR(1) idiosyncratic
  * excluded: foreign IP (decoupling bias), hicp/eur_usd/bond_10y/ip_total/
    services_J/construction (zero weight / double counting) — see VARIABLES.md

Real-time credentials (2024Q1-2026Q1, vintage inputs -> flash target):
RMSE 0.227, bias +0.03 (outputs/realtime_real_vac.csv). From 2026Q3 this spec is
FROZEN for honest out-of-sample tracking — change it only with a documented reason,
and score any challenger on quarters this spec never saw.

Run:  python3 src/baseline.py     -> current-quarter nowcast, outputs/nowcast_baseline.csv
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
OUT = BASE / "outputs"

import model as engine

EST_START = "2013-01"
DROP = ["ip_de", "ip_de_auto", "ip_ea", "hicp", "eur_usd", "bond_10y",
        "ip_total", "services_J", "construction"]


def load_vacancies() -> pd.Series:
    v = pd.read_csv(BASE / "data/raw/vacancies_q.csv").set_index("quarter")["value"]
    v.index = pd.PeriodIndex(v.index, freq="Q")
    return (100 * np.log(v).diff()).rename("vacancies")


def load_data():
    """Baseline panel: 14 monthly series (2013+) + quarterly [gdp, vacancies]."""
    m, g = engine.load_processed(start=EST_START)
    m = m.drop(columns=[c for c in DROP if c in m.columns])
    return m, g


def build(m: pd.DataFrame, g: pd.Series, target_q: pd.Period | None = None) -> DynamicFactorMQ:
    """target_q: vacancies are masked from that quarter on (observed through target-1).
    Defaults to the last quarter of `g` (the nowcast target)."""
    gq = g.to_frame()
    v = load_vacancies().reindex(g.index)
    v[v.index >= (target_q or g.index[-1])] = np.nan
    gq["vacancies"] = v
    return DynamicFactorMQ(m, endog_quarterly=gq, factors=1,
                           factor_orders=engine.FACTOR_ORDERS, idiosyncratic_ar1=True)


def main():
    m, g = load_data()
    target_q = g.index[-1]
    print(f"BASELINE (lean14 + vacancies, real services, {EST_START}+): "
          f"{m.shape[1]} monthly series | target {target_q}")
    res = build(m, g).fit(disp=0, maxiter=200)
    nc = engine.gdp_nowcast(res, target_q)
    prev = g.dropna()
    pd.DataFrame([dict(target_quarter=str(target_q), nowcast_qoq_pct=round(nc, 3),
                       prev_quarter=str(prev.index[-1]),
                       prev_actual_qoq_pct=round(float(prev.iloc[-1]), 3),
                       spec="baseline-2026-07 (lean14+vac, real services, 2013+)")]
                 ).to_csv(OUT / "nowcast_baseline.csv", index=False)
    print(f"NOWCAST {target_q}: {nc:+.2f}% QoQ  "
          f"(prev {prev.index[-1]} = {float(prev.iloc[-1]):+.2f}%)")
    print("Wrote outputs/nowcast_baseline.csv")


if __name__ == "__main__":
    main()
