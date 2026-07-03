"""2026Q1 weekly real-time nowcast: winner+new-data config vs current model."""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
import weekly_realtime as wk

OUT = BASE / "outputs"
Q = pd.Period("2026Q1", "Q")
FOREIGN = ["ip_de", "ip_de_auto", "ip_ea"]


def one(drop, start, tag, label):
    wk.DROP_COLS = drop
    wk.SAMPLE_START = start
    wk.TAG = tag
    path, flash, final, fdate = wk.run(Q)
    path.to_csv(OUT / f"weekly_realtime_{Q}{tag}.csv")
    wk.plot(path, flash, final, fdate, Q)
    u = path.dropna()
    print(f"{label:16s} 2026Q1: {path.notna().sum()} Fridays | first {u.iloc[0]:+.2f} -> "
          f"last {u.iloc[-1]:+.2f} | flash {flash:+.2f} final {final:+.2f}")


if __name__ == "__main__":
    print("2026Q1 weekly real-time — winner+new-data vs current\n")
    one(FOREIGN, pd.Period("2013Q1", "Q"), "_winnernew", "winner+new")
    one([], None, "", "current")
    print("\nWrote weekly_realtime_2026Q1_winnernew.png and weekly_realtime_2026Q1.png")
