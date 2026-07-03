"""Weekly point-in-time replay of the current production spec (lean14, real services,
+ quarterly vacancies), 2024Q1-2026Q1.

Vacancy handling: base.build_model is wrapped so the quarterly block is
[gdp_qoq, vacancies], with vacancies observed through target-1 (same rule as
src/realtime_real_vac.py; the exact within-quarter release Friday is not modelled).

Output: outputs/weekly_lean14v.csv
Run:  python3 src/weekly_lean14v.py
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

import model as base
import weekly_realtime as wk

vac_raw = pd.read_csv(BASE / "data/raw/vacancies_q.csv").set_index("quarter")["value"]
vac_raw.index = pd.PeriodIndex(vac_raw.index, freq="Q")
VAC = (100 * np.log(vac_raw).diff()).rename("vacancies")


def build_model_vac(m: pd.DataFrame, g: pd.Series) -> DynamicFactorMQ:
    gq = g.to_frame()
    target = g.index[-1]
    v = VAC.reindex(g.index)
    v[v.index >= target] = np.nan          # observed through target-1
    gq["vacancies"] = v
    return DynamicFactorMQ(m, endog_quarterly=gq, factors=1,
                           factor_orders=base.FACTOR_ORDERS, idiosyncratic_ar1=True)


base.build_model = build_model_vac        # weekly engine picks this up

wk.DROP_COLS = ["ip_de", "ip_de_auto", "ip_ea", "hicp", "eur_usd", "bond_10y",
                "ip_total", "services_J", "construction"]
wk.SAMPLE_START = pd.Period("2013Q1", "Q")
wk.TAG = "_lean14v"

rows = []
ctx = wk._context()
print(f"lean14v weekly replay: panel {ctx['m_final'].shape[1]} monthly series + vacancies (Q)",
      flush=True)
for q in pd.period_range("2024Q1", "2026Q1", freq="Q"):
    path, flash, final, fdate = wk._run_one(q, ctx)
    for d, nc in path.dropna().items():
        rows.append(dict(quarter=str(q), date=d, nowcast=nc, flash=flash))
    print(f"  {q}: {path.notna().sum():2d} Fridays | flash {flash:+.2f} | "
          f"first {path.dropna().iloc[0]:+.2f} last {path.dropna().iloc[-1]:+.2f}", flush=True)
pd.DataFrame(rows).to_csv(OUT / "weekly_lean14v.csv", index=False)
print("Wrote outputs/weekly_lean14v.csv")
