"""
True real-time test on 2024Q1-2026Q1 (9 quarters): does the panel expansion help?

For each quarter, reconstruct the vintage a forecaster saw ~50 days after quarter-end
(OECD first-release values for production/retail/unemployment + vintage GDP history,
publication-lag ragged edge on everything) and score against the FLASH release.

Variants:
  old      - pre-expansion panel (19 monthly series)
  new      - expanded panel (+ services_H/J/N, real_wage_bill = 23 series)
  new_2013 - expanded panel, estimated on 2013+ only (drops the pre-crisis growth anchor,
             per BIAS.md regime-shift finding)

Outputs: outputs/realtime_2024_2026.csv + console table.

Run:  python3 src/realtime_2024_2026.py
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

NEW_COLS = ["services_H", "services_J", "services_N", "real_wage_bill"]
QUARTERS = pd.period_range("2024Q1", "2026Q1", freq="Q")


def nowcast(em, ref, gv, q, drop_new=False, start=None) -> float:
    m, g = rb.build_panel(em, ref, gv, use_vintage=True)
    if drop_new:
        m = m.drop(columns=NEW_COLS)
    if start is not None:
        m = m.loc[m.index >= pd.Period(start, "M")]
        g = g.loc[g.index >= pd.Period(start, "M").asfreq("Q")]
    res = base.build_model(m, g).fit(disp=0, maxiter=150)
    return base.gdp_nowcast(res, q)


def main():
    flash, final, gv, editions = rb.flash_and_final_gdp()
    rows = []
    for q in QUARTERS:
        ref = q.to_timestamp(how="end") + pd.Timedelta(days=rb.NOWCAST_OFFSET_DAYS)
        em = max(e for e in editions if rb._edition_month(e).to_timestamp(how="end") <= ref)
        row = dict(quarter=str(q), edition=em,
                   flash=float(flash.get(q, np.nan)), final=float(final.get(q, np.nan)))
        for tag, kw in [("old", dict(drop_new=True)),
                        ("new", dict()),
                        ("new_2013", dict(start="2013-01"))]:
            try:
                row[tag] = nowcast(em, ref, gv, q, **kw)
            except Exception as e:  # noqa: BLE001
                row[tag] = np.nan
                print(f"  {q} {tag}: FAILED ({e})")
        rows.append(row)
        print(f"{q}: flash {row['flash']:+.2f} | old {row['old']:+.2f} "
              f"| new {row['new']:+.2f} | new_2013 {row['new_2013']:+.2f}", flush=True)

    bt = pd.DataFrame(rows).set_index("quarter")
    bt.to_csv(OUT / "realtime_2024_2026.csv")

    print("\n=== 2024Q1-2026Q1, vintage inputs -> FLASH target ===")
    print(f"{'variant':<10}{'RMSE':>8}{'mean err (bias)':>18}{'MAE':>8}")
    for tag in ["old", "new", "new_2013"]:
        e = bt[tag] - bt.flash
        print(f"{tag:<10}{np.sqrt((e**2).mean()):>8.3f}{e.mean():>+18.3f}{e.abs().mean():>8.3f}")
    print("\nWrote outputs/realtime_2024_2026.csv")


if __name__ == "__main__":
    main()
