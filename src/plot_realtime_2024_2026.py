"""Plot the 2024Q1-2026Q1 true real-time test (outputs/realtime_2024_2026.csv).

Left: nowcast variants vs the flash release. Right: RMSE / |bias| per variant.
Run:  python3 src/plot_realtime_2024_2026.py
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs"

SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {  # slug -> (label, categorical slot)
    "old": ("Old panel (19 series)", "#2a78d6"),
    "new": ("New panel (23 series)", "#1baf7a"),
    "new_2013": ("New panel, est. 2013+", "#eda100"),
}

bt = pd.read_csv(OUT / "realtime_2024_2026.csv").set_index("quarter")
x = np.arange(len(bt))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5), width_ratios=[2.1, 1],
                               facecolor=SURFACE)
for ax in (ax1, ax2):
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)

# --- left: lines vs flash ---
ax1.axhline(0, color=AXIS, lw=0.8)
ax1.plot(x, bt.flash, "-o", color=INK, lw=2, ms=6, zorder=5,
         label="Flash GDP (first release)")
for slug, (label, c) in SERIES.items():
    ax1.plot(x, bt[slug], "-o", color=c, lw=2, ms=6, label=label,
             markeredgecolor=SURFACE, markeredgewidth=1.2)
    dy = 7 if slug == "new_2013" else 0   # keep clear of the Flash end label
    ax1.annotate(label.split(" (")[0].replace("New panel, est. ", "est. "),
                 (x[-1], bt[slug].iloc[-1]), xytext=(8, dy),
                 textcoords="offset points", va="center", fontsize=8.5, color=INK)
ax1.annotate("Flash", (x[-1], bt.flash.iloc[-1]), xytext=(8, -7),
             textcoords="offset points", va="center", fontsize=8.5,
             color=INK, fontweight="bold")
ax1.set_xticks(x, bt.index, rotation=0, fontsize=8.5)
ax1.set_xlim(-0.4, len(bt) + 1.4)
ax1.set_ylabel("GDP growth, QoQ %", color=MUTED, fontsize=9)
ax1.set_title("True real-time nowcast vs flash GDP — 2024Q1–2026Q1",
              color=INK, fontsize=11, loc="left")
ax1.legend(fontsize=8.5, frameon=False, loc="upper right", labelcolor=INK)

# --- right: RMSE and |bias| ---
stats = {s: ((bt[s] - bt.flash).pow(2).mean() ** 0.5, (bt[s] - bt.flash).mean())
         for s in SERIES}
xi = np.arange(len(SERIES))
w = 0.38
for j, (metric, off) in enumerate([("RMSE", -w / 2), ("bias", w / 2)]):
    vals = [stats[s][j] for s in SERIES]
    bars = ax2.bar(xi + off, vals, w * 0.94,
                   color=[SERIES[s][1] for s in SERIES],
                   alpha=1.0 if metric == "RMSE" else 0.45, edgecolor="none")
    for b, v in zip(bars, vals):
        ax2.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=8.5, color=INK)
ax2.set_xticks(xi, ["old", "new", "new\nest. 2013+"], fontsize=8.5)
ax2.set_ylabel("percentage points (vs flash)", color=MUTED, fontsize=9)
ax2.set_title("Error vs flash (solid RMSE, faded mean bias)",
              color=INK, fontsize=11, loc="left")

fig.tight_layout()
fig.savefig(OUT / "realtime_2024_2026.png", dpi=150, facecolor=SURFACE)
print("Wrote outputs/realtime_2024_2026.png")
