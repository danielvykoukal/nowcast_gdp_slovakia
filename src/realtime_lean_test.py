"""
Lean-panel real-time test on top of the best spec (2013+, no foreign IP), 2024Q1-2026Q1,
vintage inputs -> FLASH target (identical protocol to realtime_drop_series.py):

  lean17        - also drop hicp, eur_usd, bond_10y (the financial/nominal tail)
  lean17_emp    - lean17 with unemp_rate REPLACED by monthly employment growth
                  (emp_total: SU SR DATAcube od0007ms, avg YoY % across 5 sectors)
  lean16_noiptot- lean17 minus ip_total (tests total-vs-parts double counting:
                  ip_manuf already covers ~3/4 of industry)

Reference best20 merged from outputs/realtime_drop_series.csv.
Outputs: outputs/realtime_lean_test.csv + console table.
Run:  python3 src/realtime_lean_test.py
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
OUT = BASE / "outputs"

import model as base
import realtime_backtest as rb

QUARTERS = pd.period_range("2024Q1", "2026Q1", freq="Q")
START = "2013-01"
FOREIGN = ["ip_de", "ip_de_auto", "ip_ea"]
FIN = ["hicp", "eur_usd", "bond_10y"]
EMP_LAG = 2  # DATAcube employment publication lag, months (same as wages)

emp = pd.read_csv(BASE / "data/raw/emp_total.csv", parse_dates=["date"]).set_index("date")["value"]
emp.index = emp.index.to_period("M")

VARIANTS = {
    "lean17": dict(drop=FOREIGN + FIN, emp=False),
    "lean17_emp": dict(drop=FOREIGN + FIN + ["unemp_rate"], emp=True),
    "lean16_noiptot": dict(drop=FOREIGN + FIN + ["ip_total"], emp=False),
}


def nowcast(em, ref, gv, q, drop, use_emp):
    m, g = rb.build_panel(em, ref, gv, use_vintage=True)
    m = m.drop(columns=drop)
    if use_emp:
        s = emp.reindex(m.index)
        s[s.index > m.index.max() - EMP_LAG] = np.nan   # publication-lag ragged edge
        m["emp_total"] = s
    m = m.loc[m.index >= pd.Period(START, "M")]
    g = g.loc[g.index >= pd.Period(START, "M").asfreq("Q")]
    res = base.build_model(m, g).fit(disp=0, maxiter=150)
    return base.gdp_nowcast(res, q)


def main():
    flash, final, gv, editions = rb.flash_and_final_gdp()
    rows = []
    for q in QUARTERS:
        ref = q.to_timestamp(how="end") + pd.Timedelta(days=rb.NOWCAST_OFFSET_DAYS)
        em = max(e for e in editions if rb._edition_month(e).to_timestamp(how="end") <= ref)
        row = dict(quarter=str(q), flash=float(flash.get(q, np.nan)))
        for tag, cfg in VARIANTS.items():
            try:
                row[tag] = nowcast(em, ref, gv, q, cfg["drop"], cfg["emp"])
            except Exception as e:  # noqa: BLE001
                row[tag] = np.nan
                print(f"  {q} {tag}: FAILED ({e})", flush=True)
        rows.append(row)
        print(f"{q}: flash {row['flash']:+.2f} | " +
              " | ".join(f"{t} {row[t]:+.2f}" for t in VARIANTS), flush=True)

    bt = pd.DataFrame(rows).set_index("quarter")
    prev = pd.read_csv(OUT / "realtime_drop_series.csv").set_index("quarter")
    bt = bt.join(prev[["drop_foreign_ip"]].rename(columns={"drop_foreign_ip": "best20"}))
    bt.to_csv(OUT / "realtime_lean_test.csv")

    print("\n=== 2024Q1-2026Q1, 2013+ window, no foreign IP, vintage -> FLASH ===")
    print(f"{'variant':<16}{'RMSE':>8}{'bias':>10}{'MAE':>8}")
    for tag in ["best20"] + list(VARIANTS):
        e = bt[tag] - bt.flash
        print(f"{tag:<16}{np.sqrt((e**2).mean()):>8.3f}{e.mean():>+10.3f}{e.abs().mean():>8.3f}")
    print("\nWrote outputs/realtime_lean_test.csv")


if __name__ == "__main__":
    main()
