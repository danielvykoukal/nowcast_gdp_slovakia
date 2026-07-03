"""Live nowcast: run the frozen production spec (lean14 + real services + quarterly
vacancies) on the LATEST available data and print/write the current-quarter nowcast.

Unlike weekly_realtime.py (a point-in-time backtest that replays completed quarters and
needs the flash release to exist), this driver just fits the frozen spec on whatever is
published today and nowcasts the first quarter that has no official GDP yet. The target
auto-advances when the flash for that quarter is released and picked up by preprocess.

Run:  python3 src/run_live.py
Output: outputs/nowcast_<YYYY-MM-DD>.csv and outputs/nowcast_latest.csv
"""
from __future__ import annotations
import datetime as dt
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
OUT.mkdir(exist_ok=True)

import model as base

# --- frozen lean14v spec (mirrors weekly_lean14v.py) ---
DROP_COLS = ["ip_de", "ip_de_auto", "ip_ea", "hicp", "eur_usd", "bond_10y",
             "ip_total", "services_J", "construction"]
SAMPLE_START = pd.Period("2013Q1", "Q")

vac_raw = pd.read_csv(BASE / "data/raw/vacancies_q.csv").set_index("quarter")["value"]
vac_raw.index = pd.PeriodIndex(vac_raw.index, freq="Q")
VAC = (100 * np.log(vac_raw).diff()).rename("vacancies")


def build_model_vac(m: pd.DataFrame, g: pd.Series, target_q: pd.Period) -> DynamicFactorMQ:
    gq = g.to_frame()
    v = VAC.reindex(g.index)
    v[v.index >= target_q] = np.nan            # vacancies observed through target-1
    gq["vacancies"] = v
    return DynamicFactorMQ(m, endog_quarterly=gq, factors=1,
                           factor_orders=base.FACTOR_ORDERS, idiosyncratic_ar1=True)


def main() -> None:
    m = pd.read_csv(BASE / "data/processed/monthly_panel.csv",
                    parse_dates=["date"]).set_index("date")
    m.index = m.index.to_period("M")
    m = m.drop(columns=[c for c in DROP_COLS if c in m.columns])
    m = m.loc[m.index >= SAMPLE_START.asfreq("M", "start")]

    g = pd.read_csv(BASE / "data/processed/gdp_quarterly.csv",
                    parse_dates=["date"]).set_index("date")["gdp_qoq"]
    g.index = g.index.to_period("Q")
    g = g.loc[g.index >= SAMPLE_START].dropna()

    target_q = g.index.max() + 1               # first quarter with no official GDP yet

    # extend both panels so the state space reaches the target quarter's end month
    qend_m = target_q.asfreq("M", how="end")
    m = m.reindex(pd.period_range(m.index.min(), max(m.index.max(), qend_m), freq="M"))
    g = g.reindex(pd.period_range(g.index.min(), target_q, freq="Q"))

    res = build_model_vac(m, g, target_q).fit(disp=0, maxiter=200)
    nowcast = base.gdp_nowcast(res, target_q)

    asof = dt.date.today().isoformat()
    n_monthly = int(m.loc[target_q.asfreq("M", "start"):qend_m].notna().sum().sum())
    row = pd.DataFrame([dict(asof=asof, target_q=str(target_q), nowcast=round(nowcast, 3),
                             monthly_obs_in_quarter=n_monthly, n_series=m.shape[1])])
    row.to_csv(OUT / f"nowcast_{asof}.csv", index=False)
    row.to_csv(OUT / "nowcast_latest.csv", index=False)
    print(f"{asof}  {target_q} nowcast = {nowcast:+.2f}% QoQ  "
          f"({n_monthly} monthly obs in quarter, {m.shape[1]} series)")
    print(f"Wrote outputs/nowcast_{asof}.csv and outputs/nowcast_latest.csv")


if __name__ == "__main__":
    main()
