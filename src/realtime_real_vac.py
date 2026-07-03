"""
Real-time test of (a) DEFLATED services turnover and (b) quarterly JOB VACANCIES,
on the lean14 spec (2013+, no foreign IP/financial/ip_total/services_J/construction).
2024Q1-2026Q1, vintage inputs -> FLASH target, same protocol as realtime_lean14.

  lean14_real     - lean14 on the rebuilt panel (services_H/iaf/N now HICP-deflated
                    real growth; previously nominal). Everything else identical.
  lean14_real_vac - + job vacancies (Eurostat jvs_q_r21, SK, SA, B-T, dlog QoQ) as a
                    SECOND QUARTERLY variable next to GDP. Vintage rule: vacancies
                    observed through target-1 (published ~2 months after quarter end;
                    latest-vintage values, no revision history available).

Outputs: outputs/realtime_real_vac.csv + console table (lean14 nominal ref merged).
Run:  python3 src/realtime_real_vac.py
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
import realtime_backtest as rb

QUARTERS = pd.period_range("2024Q1", "2026Q1", freq="Q")
START = "2013-01"
DROP = ["ip_de", "ip_de_auto", "ip_ea", "hicp", "eur_usd", "bond_10y",
        "ip_total", "services_J", "construction"]

vac_raw = pd.read_csv(BASE / "data/raw/vacancies_q.csv").set_index("quarter")["value"]
vac_raw.index = pd.PeriodIndex(vac_raw.index, freq="Q")
VAC = (100 * np.log(vac_raw).diff()).rename("vacancies")  # QoQ growth of vacancies


def nowcast(em, ref, gv, q, use_vac):
    m, g = rb.build_panel(em, ref, gv, use_vintage=True)
    m = m.drop(columns=DROP).loc[m.index >= pd.Period(START, "M")]
    g = g.loc[g.index >= pd.Period(START, "M").asfreq("Q")]
    gq = g.to_frame()
    if use_vac:
        v = VAC.reindex(g.index)
        v[v.index >= q] = np.nan                    # observed through target-1 only
        gq["vacancies"] = v
    mod = DynamicFactorMQ(m, endog_quarterly=gq, factors=1,
                          factor_orders=base.FACTOR_ORDERS, idiosyncratic_ar1=True)
    res = mod.fit(disp=0, maxiter=150)
    return base.gdp_nowcast(res, q)


def main():
    flash, final, gv, editions = rb.flash_and_final_gdp()
    rows = []
    for q in QUARTERS:
        ref = q.to_timestamp(how="end") + pd.Timedelta(days=rb.NOWCAST_OFFSET_DAYS)
        em = max(e for e in editions if rb._edition_month(e).to_timestamp(how="end") <= ref)
        row = dict(quarter=str(q), flash=float(flash.get(q, np.nan)))
        for tag, uv in [("lean14_real", False), ("lean14_real_vac", True)]:
            try:
                row[tag] = nowcast(em, ref, gv, q, uv)
            except Exception as e:  # noqa: BLE001
                row[tag] = np.nan
                print(f"  {q} {tag}: FAILED ({e})", flush=True)
        rows.append(row)
        print(f"{q}: flash {row['flash']:+.2f} | real {row['lean14_real']:+.2f} "
              f"| real+vac {row['lean14_real_vac']:+.2f}", flush=True)

    bt = pd.DataFrame(rows).set_index("quarter")
    prev = pd.read_csv(OUT / "realtime_lean14.csv").set_index("quarter")
    bt = bt.join(prev[["lean14"]].rename(columns={"lean14": "lean14_nominal"}))
    bt.to_csv(OUT / "realtime_real_vac.csv")

    print("\n=== 2024Q1-2026Q1, vintage -> FLASH ===")
    print(f"{'variant':<18}{'RMSE':>8}{'bias':>10}{'MAE':>8}")
    for tag in ["lean14_nominal", "lean14_real", "lean14_real_vac"]:
        e = bt[tag] - bt.flash
        print(f"{tag:<18}{np.sqrt((e**2).mean()):>8.3f}{e.mean():>+10.3f}{e.abs().mean():>8.3f}")
    print("\nWrote outputs/realtime_real_vac.csv")


if __name__ == "__main__":
    main()
