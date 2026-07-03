"""
Weekly-pipeline A/B test: does the foreign IP block (ip_de, ip_de_auto, ip_ea) earn its
keep EARLY in the quarter, when domestic hard data have not arrived yet?

Both configs use the production window (2013+) and the full 23-series panel except for
the block under test. Each target quarter 2024Q1-2026Q1 is replayed Friday-by-Friday
through the point-in-time engine (src/weekly_realtime.py: OECD editions, day-level
release calendar, target GDP withheld until flash). Errors vs the FLASH are bucketed by
lead time:

  M1 / M2 / M3     - Fridays in months 1-3 of the target quarter
  post-Q backcast  - Fridays after quarter-end, before the flash

Outputs: outputs/weekly_fip_test.csv (per-Friday) + console RMSE-by-bucket table
         + outputs/weekly_fip_test.png

Run:  python3 src/weekly_foreign_ip_test.py
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
OUT = BASE / "outputs"

import weekly_realtime as wk

FOREIGN = ["ip_de", "ip_de_auto", "ip_ea"]
QUARTERS = pd.period_range("2024Q1", "2026Q1", freq="Q")
CONFIGS = {"with_fip": [], "no_fip": FOREIGN}

SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
COLORS = {"with_fip": "#2a78d6", "no_fip": "#1baf7a"}
LABELS = {"with_fip": "with foreign IP (23 series)", "no_fip": "without foreign IP (20 series)"}


def bucket(d: pd.Timestamp, q: pd.Period) -> str:
    qstart = q.to_timestamp(how="start")
    if d <= q.to_timestamp(how="end"):
        month = (d.year - qstart.year) * 12 + d.month - qstart.month  # 0,1,2 within quarter
        return f"M{month + 1}"
    return "post-Q"


def main():
    rows = []
    for tag, drop in CONFIGS.items():
        wk.DROP_COLS = drop
        wk.SAMPLE_START = pd.Period("2013Q1", "Q")
        wk.TAG = f"_fiptest_{tag}"
        ctx = wk._context()
        print(f"config {tag}: panel {ctx['m_final'].shape[1]} series", flush=True)
        for q in QUARTERS:
            path, flash, final, fdate = wk._run_one(q, ctx)
            for d, nc in path.dropna().items():
                rows.append(dict(config=tag, quarter=str(q), date=d,
                                 bucket=bucket(d, q), nowcast=nc, flash=flash,
                                 err=nc - flash))
            print(f"  {q}: {path.notna().sum():2d} Fridays | flash {flash:+.2f} | "
                  f"first {path.dropna().iloc[0]:+.2f} last {path.dropna().iloc[-1]:+.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "weekly_fip_test.csv", index=False)

    order = ["M1", "M2", "M3", "post-Q"]
    print("\n=== weekly RMSE vs flash by lead-time bucket, 2024Q1-2026Q1, 2013+ window ===")
    print(f"{'bucket':<8}" + "".join(f"{LABELS[t]:>32}" for t in CONFIGS))
    tbl = {}
    for b in order:
        line, vals = f"{b:<8}", []
        for tag in CONFIGS:
            e = df[(df.config == tag) & (df.bucket == b)].err
            r, bias = float(np.sqrt((e ** 2).mean())), float(e.mean())
            vals.append(r)
            line += f"{r:>18.3f} ({bias:+.2f})   "
        tbl[b] = vals
        print(line + f"  n={len(e)}")

    # plot: RMSE by bucket, two lines
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    x = np.arange(len(order))
    for tag in CONFIGS:
        y = [tbl[b][list(CONFIGS).index(tag)] for b in order]
        ax.plot(x, y, "-o", color=COLORS[tag], lw=2, ms=7,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=LABELS[tag])
        ax.annotate(LABELS[tag].split(" (")[0], (x[-1], y[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=INK)
    ax.set_xticks(x, ["month 1", "month 2", "month 3", "post-quarter\n(backcast)"], fontsize=9)
    ax.set_xlim(-0.3, len(order) + 0.9)
    ax.set_ylabel("weekly nowcast RMSE vs flash (pp)", color=MUTED, fontsize=9)
    ax.set_title("Does foreign IP help early in the quarter? 2024Q1–2026Q1, 2013+ window",
                 color=INK, fontsize=11, loc="left")
    ax.legend(fontsize=8.5, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "weekly_fip_test.png", dpi=150, facecolor=SURFACE)
    print("\nWrote outputs/weekly_fip_test.csv, outputs/weekly_fip_test.png")


if __name__ == "__main__":
    main()
