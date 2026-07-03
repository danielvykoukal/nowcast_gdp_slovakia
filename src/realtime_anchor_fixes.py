"""
Real-time validation of the two anchor fixes (2024Q1-2026Q1, vintage inputs -> FLASH
target, same protocol as realtime_2024_2026.py):

  ll        - fix 1: local-level DFM (dfm_locallevel.py), full 2002+ sample
  ll_2013   - fix 1 + fix 2 combined
  (old / new / new_2013 DynamicFactorMQ results merged in from realtime_2024_2026.csv)

Per-vintage: expanding-window refit (warm-started from the final-data fit for speed;
MLE optimum does not depend on the start point), per-vintage standardisation moments.

Outputs: outputs/realtime_anchor_fixes.csv + console table.

Run:  python3 src/realtime_anchor_fixes.py
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

import realtime_backtest as rb
import dfm_locallevel as ll

QUARTERS = pd.period_range("2024Q1", "2026Q1", freq="Q")
VARIANTS = {"ll": None, "ll_2013": "2013-01"}


def to_timestamps(m: pd.DataFrame, g: pd.Series):
    """rb.build_panel returns PeriodIndex frames; the SSM wants month-end timestamps."""
    mt = m.copy()
    mt.index = m.index.to_timestamp(how="end").normalize() + pd.offsets.MonthEnd(0)
    gt = g.copy()
    gt.index = g.index.asfreq("M", how="end").to_timestamp(how="end").normalize() + pd.offsets.MonthEnd(0)
    return mt, gt.reindex(mt.index)


def trim(m, g, start):
    if start is None:
        return m, g
    return m.loc[m.index >= start], g.loc[g.index >= start]


def main():
    flash, final, gv, editions = rb.flash_and_final_gdp()

    # warm starts: one final-data fit per variant
    warm = {}
    for tag, start in VARIANTS.items():
        m = pd.read_csv(BASE / "data/processed/monthly_panel.csv",
                        parse_dates=["date"]).set_index("date")
        g = pd.read_csv(BASE / "data/processed/gdp_quarterly.csv",
                        parse_dates=["date"]).set_index("date")["gdp_qoq"]
        m, g = trim(m, g.reindex(m.index), start)
        res, _ = ll.fit_ll(m, g)
        warm[tag] = np.asarray(res.params)
        print(f"warm start {tag}: llf {res.llf:.1f}, sigma2.trend {res.params[-1]:.6f}", flush=True)

    rows = []
    for q in QUARTERS:
        ref = q.to_timestamp(how="end") + pd.Timedelta(days=rb.NOWCAST_OFFSET_DAYS)
        em = max(e for e in editions if rb._edition_month(e).to_timestamp(how="end") <= ref)
        mv, gvq = rb.build_panel(em, ref, gv, use_vintage=True)
        mt, gt = to_timestamps(mv, gvq)
        row = dict(quarter=str(q), flash=float(flash.get(q, np.nan)),
                   final=float(final.get(q, np.nan)))
        for tag, start in VARIANTS.items():
            try:
                m2, g2 = trim(mt, gt, start)
                res, mom = ll.fit_ll(m2, g2, start_params=warm[tag], maxiter=300)
                row[tag] = ll.gdp_signal(res, q.asfreq("M", how="end").to_timestamp(
                    how="end").normalize() + pd.offsets.MonthEnd(0), mom)
            except Exception as e:  # noqa: BLE001
                row[tag] = np.nan
                print(f"  {q} {tag}: FAILED ({e})", flush=True)
        rows.append(row)
        print(f"{q}: flash {row['flash']:+.2f} | ll {row['ll']:+.2f} "
              f"| ll_2013 {row['ll_2013']:+.2f}", flush=True)

    bt = pd.DataFrame(rows).set_index("quarter")
    prev = pd.read_csv(OUT / "realtime_2024_2026.csv").set_index("quarter")
    bt = bt.join(prev[["old", "new", "new_2013"]])
    bt.to_csv(OUT / "realtime_anchor_fixes.csv")

    print("\n=== 2024Q1-2026Q1, vintage inputs -> FLASH target ===")
    print(f"{'variant':<10}{'RMSE':>8}{'mean err (bias)':>18}{'MAE':>8}")
    for tag in ["old", "new", "new_2013", "ll", "ll_2013"]:
        e = bt[tag] - bt.flash
        print(f"{tag:<10}{np.sqrt((e**2).mean()):>8.3f}{e.mean():>+18.3f}{e.abs().mean():>8.3f}")
    print("\nWrote outputs/realtime_anchor_fixes.csv")


if __name__ == "__main__":
    main()
