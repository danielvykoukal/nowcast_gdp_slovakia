"""Weekly point-in-time nowcast path for the LIVE quarter 2026Q2 under the baseline
spec — quarter start through today (flash not out; due ~mid-Aug 2026).

Output: outputs/nowcast_2026Q2_weekly.png, outputs/nowcast_2026Q2_weekly.csv
Run:  python3 src/plot_q2_2026.py
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

import model as base
import baseline
import weekly_realtime as wk

base.build_model = baseline.build            # baseline quarterly block [gdp, vacancies]
wk.DROP_COLS = baseline.DROP
wk.SAMPLE_START = pd.Period("2013Q1", "Q")
wk.TAG = "_baselineQ2"

SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
BLUE = "#2a78d6"

TARGET = pd.Period("2026Q2", "Q")
TODAY = pd.Timestamp("2026-07-03")


def main():
    ctx = wk._context()
    gv, editions = ctx["gv"], ctx["editions"]
    fridays = pd.date_range(TARGET.to_timestamp(how="start"), TODAY, freq="W-FRI")
    out, last_sig, last_nc = {}, None, np.nan
    for d in fridays:
        m, g, em = wk.panel_asof(d, TARGET, ctx["m_final"], ctx["rel"], gv, editions)
        sig = (em, int(m[[c for c in wk.NONOECD if c in m.columns]].notna().sum().sum()))
        if sig != last_sig:
            res = base.build_model(m, g).fit(disp=0, maxiter=100, start_params=ctx["warm"])
            last_nc = base.gdp_nowcast(res, TARGET)
            last_sig = sig
        out[d] = last_nc
        print(f"  {d.date()}: {last_nc:+.3f}", flush=True)
    path = pd.Series(out, name="nowcast")
    path.rename_axis("date").to_frame().to_csv(OUT / "nowcast_2026Q2_weekly.csv")

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)

    prev_flash = 0.20  # 2026Q1 flash, for context
    ax.axhline(prev_flash, color=MUTED, ls="--", lw=1.2, alpha=0.8)
    ax.annotate("2026Q1 flash +0.20", (path.index[0], prev_flash), xytext=(0, -12),
                textcoords="offset points", fontsize=8, color=MUTED)
    qend = TARGET.to_timestamp(how="end")
    ax.axvline(qend, color=AXIS, ls=":", lw=1)
    ax.annotate(" quarter end", (qend, 0.97), xycoords=("data", "axes fraction"),
                va="top", fontsize=8, color=MUTED)
    ax.step(path.index, path.values, where="post", color=BLUE, lw=2)
    ax.plot(path.index, path.values, "o", color=BLUE, ms=5,
            markeredgecolor=SURFACE, markeredgewidth=1)
    last = path.dropna().iloc[-1]
    ax.annotate(f"now: {last:+.2f}%", (path.index[-1], last), xytext=(10, 0),
                textcoords="offset points", va="center", fontsize=10,
                color=INK, fontweight="bold")
    ax.set_xlim(path.index[0] - pd.Timedelta(days=5), path.index[-1] + pd.Timedelta(days=30))
    ax.set_ylabel("GDP growth nowcast, QoQ %", color=MUTED, fontsize=9)
    ax.set_title("2026Q2 weekly real-time nowcast — baseline model "
                 "(flash due ~mid-Aug 2026)", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "nowcast_2026Q2_weekly.png", dpi=150, facecolor=SURFACE)
    print("Wrote outputs/nowcast_2026Q2_weekly.png")


if __name__ == "__main__":
    main()
