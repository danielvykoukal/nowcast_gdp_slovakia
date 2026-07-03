"""Weekly point-in-time replay of the lean14 production spec, 2024Q1-2026Q1.
Output: outputs/weekly_lean14.csv (per-Friday nowcast per target quarter).
Run:  python3 src/weekly_lean14.py
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
OUT = BASE / "outputs"

import weekly_realtime as wk

wk.DROP_COLS = ["ip_de", "ip_de_auto", "ip_ea", "hicp", "eur_usd", "bond_10y",
                "ip_total", "services_J", "construction"]
wk.SAMPLE_START = pd.Period("2013Q1", "Q")
wk.TAG = "_lean14"

rows = []
ctx = wk._context()
print(f"lean14 weekly replay: panel {ctx['m_final'].shape[1]} series", flush=True)
for q in pd.period_range("2024Q1", "2026Q1", freq="Q"):
    path, flash, final, fdate = wk._run_one(q, ctx)
    for d, nc in path.dropna().items():
        rows.append(dict(quarter=str(q), date=d, nowcast=nc, flash=flash))
    print(f"  {q}: {path.notna().sum():2d} Fridays | flash {flash:+.2f} | "
          f"first {path.dropna().iloc[0]:+.2f} last {path.dropna().iloc[-1]:+.2f}", flush=True)
pd.DataFrame(rows).to_csv(OUT / "weekly_lean14.csv", index=False)
print("Wrote outputs/weekly_lean14.csv")
