"""
Weekly real-time backtest of the WINNER config — drop the foreign IP block and estimate 2013+
(see REGIME.md) — run through the rigorous point-in-time engine in src/weekly_realtime.py
(OECD editions, Friday-by-Friday, target GDP withheld until its flash).

Produces the winner's 2025 convergence grid + continuous timeline, tagged "_winner" so they sit
alongside the current-model versions for comparison.

Run:  python src/weekly_winner.py [year]      (default 2025)
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
import weekly_realtime as wk

FOREIGN = ["ip_de", "ip_de_auto", "ip_ea"]
wk.DROP_COLS = FOREIGN
wk.SAMPLE_START = pd.Period("2013Q1", "Q")
# expanded panel (real_wage_bill + services H/J/N) is now in monthly_panel.csv, so dropping only
# foreign IP = the "winner + all new data" config from REGIME.md. Tagged _winnernew to keep the
# earlier no-new-data winner plots for comparison.
wk.TAG = "_winnernew"


def main():
    year = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 2025
    print(f"WINNER config (drop foreign IP {FOREIGN}, estimate {wk.SAMPLE_START}+) — "
          f"weekly real-time {year}\n")
    wk.run_year(year)
    wk.continuous_timeline(year)
    print("Done. Compare vs current model: weekly_realtime_"
          f"{year}.png / _winner.png, weekly_timeline_{year}.png / _winner.png")


if __name__ == "__main__":
    main()
