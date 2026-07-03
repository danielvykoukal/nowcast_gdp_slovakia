"""Plots of the best model so far (lean14 + quarterly vacancies, real services, 2013+),
true real-time 2024Q1-2026Q1.

  outputs/best_model_quarterly.png - quarterly nowcast vs flash (+ starting spec for context)
  outputs/best_model_weekly.png    - 3x3 grid: weekly point-in-time path per quarter

Data: outputs/realtime_drop_series.csv, outputs/realtime_2024_2026.csv,
      outputs/weekly_lean14.csv.
Run:  python3 src/plot_best_model.py
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
import sys
sys.path.insert(0, str(BASE / "src"))
import realtime_backtest as rb
import numpy as _np

SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, AQUA, RED = "#2a78d6", "#1baf7a", "#e34948"


def _edition_date(em: int) -> pd.Timestamp:
    return pd.Period(f"{em // 100}-{em % 100:02d}", freq="M").to_timestamp(how="end").normalize()


def gdp_release_dates():
    """Per quarter: flash value/date, final (today) value, first-revision date/value."""
    flash, final, gv, editions = rb.flash_and_final_gdp()

    def growth(ed):
        s = rb.series_asof(gv, ed, "Q")
        return 100 * _np.log(s).diff()

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
                      final=float(final.get(q, _np.nan)),
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


# ---------------------------------------------------------------- quarterly
def quarterly():
    best = pd.read_csv(OUT / "realtime_real_vac.csv").set_index("quarter")
    best["best"] = best["lean14_real_vac"]
    old = pd.read_csv(OUT / "realtime_2024_2026.csv").set_index("quarter")["old"]
    x = np.arange(len(best))

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=SURFACE)
    style(ax)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.plot(x, old, "-o", color=MUTED, lw=1.4, ms=4, alpha=0.6,
            label="Starting spec (19 series, full window)")
    rel = gdp_release_dates()
    fin = pd.Series({str(q): v["final"] for q, v in rel.items()}).reindex(best.index)
    ax.plot(x, fin, "-s", color=AQUA, lw=1.6, ms=5, alpha=0.9, zorder=4,
            label="Final GDP (revised, today's value)")
    ax.plot(x, best.flash, "-o", color=INK, lw=2, ms=6, zorder=5,
            label="Flash GDP (first release)")
    ax.plot(x, best.best, "-o", color=BLUE, lw=2, ms=6,
            markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=6,
            label="Best model (lean14 + vacancies, real services, 2013+)")
    for xi, (f, b) in enumerate(zip(best.flash, best.best)):
        ax.plot([xi, xi], [f, b], color=BLUE, lw=0.8, alpha=0.35, zorder=1)
    ax.annotate("Best model", (x[-1], best.best.iloc[-1]), xytext=(8, 6),
                textcoords="offset points", va="center", fontsize=8.5, color=INK)
    ax.annotate("Flash", (x[-1], best.flash.iloc[-1]), xytext=(8, -6),
                textcoords="offset points", va="center", fontsize=8.5,
                color=INK, fontweight="bold")
    e = best.best - best.flash
    r, bias = float(np.sqrt((e ** 2).mean())), float(e.mean())
    eo = old - best.flash
    ax.set_title(f"Best model vs flash, true real-time — RMSE {r:.2f}pp, bias {bias:+.2f}pp "
                 f"(starting spec: {float(np.sqrt((eo**2).mean())):.2f}, {float(eo.mean()):+.2f})",
                 color=INK, fontsize=10.5, loc="left")
    ax.set_xticks(x, best.index, fontsize=8.5)
    ax.set_xlim(-0.4, len(best) + 1.1)
    ax.set_ylabel("GDP growth, QoQ %", color=MUTED, fontsize=9)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "best_model_quarterly.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- weekly grid
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
    fig.suptitle("Best model weekly nowcast (blue) vs flash (black dashed) and final GDP (green dash-dot)\n"
                 "vlines: quarter end (gray) · flash release (red) · first revision (green; diamond = revised value)",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "best_model_weekly.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------- continuous timeline
def continuous():
    """One weekly timeline: each Friday shows the nowcast of the LIVE quarter (the
    earliest quarter whose flash is not yet released); segment ends at each flash."""
    df = pd.read_csv(OUT / "weekly_lean14v.csv", parse_dates=["date"])
    rel = {str(q): v for q, v in gdp_release_dates().items()}
    quarters = sorted(df.quarter.unique())

    fig, ax = plt.subplots(figsize=(13.5, 5.6), facecolor=SURFACE)
    style(ax)
    for i, q in enumerate(quarters):
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
                    xytext=(4, -16), textcoords="offset points",
                    fontsize=7.5, color=INK)
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
    fig.savefig(OUT / "best_model_weekly_timeline.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    quarterly()
    weekly()
    continuous()
    print("Wrote outputs/best_model_quarterly.png, outputs/best_model_weekly.png, "
          "outputs/best_model_weekly_timeline.png")
