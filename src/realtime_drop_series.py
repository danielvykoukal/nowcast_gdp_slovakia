"""
Series-dropping ablation on the production spec (DynamicFactorMQ, 23-series panel,
2013+ window), true real-time 2024Q1-2026Q1 (vintage inputs -> FLASH target).

Full leave-one-out is ~200 refits; instead drop blocks flagged by earlier findings:
  drop_foreign_ip  - ip_de, ip_de_auto, ip_ea       (BIAS.md: biggest 2025 overshoot driver)
  drop_deadweight  - bond_10y, eur_usd, hicp, cons_conf_sk  (near-zero factor weights)
  drop_both        - union of the two
  drop_both_imp    - drop_both + imports_vol        (near-duplicate of exports_vol)

Baseline new_2013 merged in from outputs/realtime_2024_2026.csv.

Outputs: outputs/realtime_drop_series.csv + console table.
Run:  python3 src/realtime_drop_series.py
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

FOREIGN_IP = ["ip_de", "ip_de_auto", "ip_ea"]
DEADWEIGHT = ["bond_10y", "eur_usd", "hicp", "cons_conf_sk"]
VARIANTS = {
    "drop_foreign_ip": FOREIGN_IP,
    "drop_deadweight": DEADWEIGHT,
    "drop_both": FOREIGN_IP + DEADWEIGHT,
    "drop_both_imp": FOREIGN_IP + DEADWEIGHT + ["imports_vol"],
}


def nowcast(em, ref, gv, q, drop):
    m, g = rb.build_panel(em, ref, gv, use_vintage=True)
    m = m.drop(columns=drop).loc[m.index >= pd.Period(START, "M")]
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
        for tag, drop in VARIANTS.items():
            try:
                row[tag] = nowcast(em, ref, gv, q, drop)
            except Exception as e:  # noqa: BLE001
                row[tag] = np.nan
                print(f"  {q} {tag}: FAILED ({e})", flush=True)
        rows.append(row)
        print(f"{q}: flash {row['flash']:+.2f} | " +
              " | ".join(f"{t} {row[t]:+.2f}" for t in VARIANTS), flush=True)

    bt = pd.DataFrame(rows).set_index("quarter")
    prev = pd.read_csv(OUT / "realtime_2024_2026.csv").set_index("quarter")
    bt = bt.join(prev[["new_2013"]])
    bt.to_csv(OUT / "realtime_drop_series.csv")

    print("\n=== 2024Q1-2026Q1, 2013+ window, vintage inputs -> FLASH target ===")
    print(f"{'variant':<18}{'RMSE':>8}{'bias':>10}{'MAE':>8}")
    for tag in ["new_2013"] + list(VARIANTS):
        e = bt[tag] - bt.flash
        print(f"{tag:<18}{np.sqrt((e**2).mean()):>8.3f}{e.mean():>+10.3f}{e.abs().mean():>8.3f}")
    print("\nWrote outputs/realtime_drop_series.csv")


if __name__ == "__main__":
    main()
