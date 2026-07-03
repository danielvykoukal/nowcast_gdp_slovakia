"""Nice weekly nowcast graphs for the frozen lean14v spec, true real-time 2024Q1-2026Q1.

  outputs/weekly_grid.png      - 3x3 grid: weekly point-in-time path per quarter vs flash/final
  outputs/weekly_timeline.png  - one continuous timeline; each segment tracks the live quarter

Data: outputs/weekly_lean14v.csv (regenerate with `python3 src/weekly_lean14v.py`).
Run:  python3 src/plot_weekly.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs"
sys.path.insert(0, str(BASE / "src"))
import realtime_backtest as rb

SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, AQUA, RED = "#2a78d6", "#1baf7a", "#e34948"


def _edition_date(em: int) -> pd.Timestamp:
    return pd.Period(f"{em // 100}-{em % 100:02d}", freq="M").to_timestamp(how="end").normalize()


def gdp_release_dates():
    """Per quarter: flash value/date, final (today) value, first-revision date/value."""
    flash, final, gv, editions = rb.flash_and_final_gdp()

    def growth(ed):
        s = rb.series_asof(gv, ed, "Q")
        return 100 * np.log(s).diff()

    cache = {ed: growth(ed) for ed in editions}
    out = {}
    for q, fv in flash.items():
        eds = [e for e in editions if q in cache[e].dropna().index]
        if not eds:
            continue
        flash_ed = min(eds)
        rev_ed, rev_val = None, None
        for e in sorted(eds):
            v = cache[e].get(q)
            if e > flash_ed and pd.notna(v) and abs(v - fv) > 0.005:
                rev_ed, rev_val = e, float(v)
                break
        out[q] = dict(flash=float(fv), flash_date=_edition_date(flash_ed),
                      final=float(final.get(q, np.nan)),
                      rev_date=_edition_date(rev_ed) if rev_ed else None,
                      rev_val=rev_val)
    return out


def style(ax):
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def weekly():
    df = pd.read_csv(OUT / "weekly_lean14v.csv", parse_dates=["date"])
    quarters = sorted(df.quarter.unique())
    rel = {str(q): v for q, v in gdp_release_dates().items()}

    fig, axes = plt.subplots(3, 3, figsize=(13, 8), facecolor=SURFACE, sharey=True)
    for q, ax in zip(quarters, axes.ravel()):
        style(ax)
        seg = df[df.quarter == q].sort_values("date")
        flash = seg.flash.iloc[0]
        r = rel.get(q, {})
        qend = pd.Period(q, "Q").to_timestamp(how="end")
        ax.axhline(flash, color=INK, ls="--", lw=1.4)
        final = r.get("final")
        if final is not None and pd.notna(final):
            ax.axhline(final, color=AQUA, ls="-.", lw=1.4, alpha=0.9)
        ax.axvline(qend, color=AXIS, ls=":", lw=1)
        frel = r.get("flash_date", seg.date.max())
        ax.axvline(frel, color=RED, ls=":", lw=1.2)
        ax.annotate("flash", (frel, 0.97), xycoords=("data", "axes fraction"),
                    xytext=(-2, 0), textcoords="offset points",
                    ha="right", va="top", fontsize=7.5, color=RED)
        rd = r.get("rev_date")
        if rd is not None:
            ax.axvline(rd, color=AQUA, ls=":", lw=1.2)
            ax.annotate("revised", (rd, 0.97), xycoords=("data", "axes fraction"),
                        xytext=(2, 0), textcoords="offset points",
                        ha="left", va="top", fontsize=7.5, color=AQUA)
            ax.plot([rd], [r["rev_val"]], "D", color=AQUA, ms=5, zorder=6)
        ax.step(seg.date, seg.nowcast, where="post", color=BLUE, lw=1.8)
        ax.plot(seg.date, seg.nowcast, "o", color=BLUE, ms=3.5,
                markeredgecolor=SURFACE, markeredgewidth=0.8)
        right = max([seg.date.max()] + ([rd] if rd is not None else []))
        ax.set_xlim(seg.date.min() - pd.Timedelta(days=10), right + pd.Timedelta(days=20))
        ftxt = f"final {final:+.2f}%" if final is not None and pd.notna(final) else ""
        ax.set_title(f"{q}   flash {flash:+.2f}%   {ftxt}", color=INK, fontsize=10, loc="left")
        ax.set_ylim(-0.15, 1.0)
        ax.tick_params(axis="x", labelrotation=30)
        for lb in ax.get_xticklabels():
            lb.set_ha("right")
        ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %y"))
    for ax in axes[:, 0]:
        ax.set_ylabel("QoQ %", color=MUTED, fontsize=8.5)
    fig.suptitle("Weekly real-time nowcast (blue) vs flash (black dashed) and final GDP (green dash-dot)\n"
                 "vlines: quarter end (gray) · flash release (red) · first revision (green; diamond = revised value)",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "weekly_grid.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def continuous():
    """One weekly timeline: each Friday shows the nowcast of the LIVE quarter (the earliest
    quarter whose flash is not yet released); each segment ends at that flash."""
    df = pd.read_csv(OUT / "weekly_lean14v.csv", parse_dates=["date"])
    rel = {str(q): v for q, v in gdp_release_dates().items()}
    quarters = sorted(df.quarter.unique())

    fig, ax = plt.subplots(figsize=(13.5, 5.6), facecolor=SURFACE)
    style(ax)
    for q in quarters:
        r = rel[q]
        frel, fprev = r["flash_date"], rel.get(str(pd.Period(q, "Q") - 1), {}).get("flash_date")
        seg = df[df.quarter == q].sort_values("date")
        live = seg[(seg.date <= frel) & ((fprev is None) | (seg.date > fprev))]
        if live.empty:
            continue
        ax.step(live.date, live.nowcast, where="post", color=BLUE, lw=1.8)
        ax.plot(live.date, live.nowcast, "o", color=BLUE, ms=3.2,
                markeredgecolor=SURFACE, markeredgewidth=0.8)
        ax.axvline(frel, color=AXIS, ls=":", lw=0.9)
        ax.plot([frel], [r["flash"]], "D", color=INK, ms=7, zorder=6)
        ax.annotate(f"{q}\nflash {r['flash']:+.2f}", (frel, r["flash"]),
                    xytext=(4, -16), textcoords="offset points", fontsize=7.5, color=INK)
        mid = live.date.iloc[len(live) // 2]
        ax.annotate(f"→{q}", (mid, live.nowcast.max()), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=8, color=MUTED)
    fx = [rel[q]["flash_date"] for q in quarters]
    ax.plot(fx, [rel[q]["flash"] for q in quarters], "--", color=INK, lw=1, alpha=0.5,
            label="flash actuals (at release)")
    ax.plot(fx, [rel[q]["final"] for q in quarters], "-.", color=AQUA, lw=1.2, alpha=0.8,
            label="final GDP (today's value)")
    ax.plot([], [], "-o", color=BLUE, lw=1.8, ms=3.2, label="live-quarter weekly nowcast")
    ax.set_ylabel("GDP growth, QoQ %", color=MUTED, fontsize=9)
    ax.set_title("Continuous weekly real-time nowcast — each segment tracks the live quarter; "
                 "diamonds = flash at its release", color=INK, fontsize=11, loc="left")
    ax.legend(fontsize=8.5, frameon=False, loc="upper right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "weekly_timeline.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    weekly()
    continuous()
    print("Wrote outputs/weekly_grid.png and outputs/weekly_timeline.png")
