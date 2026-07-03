"""
Chronologically-correct weekly real-time nowcast convergence chart.

For a target quarter, walk forward Friday by Friday. At each Friday `d` we use ONLY what was
knowable on that date:

  * Revisable series (production BTE & C, retail, unemployment) and the GDP history come from
    the OECD **edition available as of `d`** (an edition dated YYYYMM is treated as available
    from the end of that month). A value keeps its first-release number until an OECD edition
    actually revises it - and that correction enters the nowcast only on the following Friday,
    never earlier. No look-ahead.
  * Non-revisable / unavailable series (sentiment, financial, foreign IP, construction, trade)
    are ragged by their real release dates (publication lag in days). These revise little.
  * The target quarter's GDP is withheld until its flash release, at which point the path stops.

The model is re-estimated whenever the information set changes (a new OECD edition, or a new
non-OECD release). The nowcast should converge to the flash (first release) - the number that was
actually published for the quarter.

Output: outputs/weekly_realtime_<Q>.png, outputs/weekly_realtime_<Q>.csv
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

import model as base                     # single-factor DFM (fast) + PUB_LAG
import realtime_backtest as rb           # vintages, series_asof, transform, OECD_COLS, TRANSFORM
import weekly_nowcast as wk              # LAG_DAYS, release_matrix

NONOECD = [c for c in base.PUB_LAG if c not in rb.OECD_COLS]

# --- optional config (set by callers, e.g. src/weekly_winner.py) ---
DROP_COLS: list[str] = []      # series to drop from the panel (e.g. foreign IP)
SAMPLE_START = None            # a quarterly pd.Period to cut the estimation window, or None
TAG = ""                       # filename suffix so different configs don't overwrite each other


# ---------------------------------------------------------------------------
def edition_release(em: int) -> pd.Timestamp:
    """An edition dated YYYYMM is treated as available at the end of that month."""
    y, mm = divmod(em, 100)
    return pd.Period(f"{y}-{mm:02d}", freq="M").to_timestamp(how="end").normalize()


def oecd_edition_asof(editions, d: pd.Timestamp):
    avail = [e for e in editions if edition_release(e) <= d]
    return max(avail) if avail else None


def panel_asof(d, target_q, m_final, rel, gv, editions):
    """Point-in-time panel + GDP history knowable as of Friday `d`."""
    em = oecd_edition_asof(editions, d)
    m = m_final.copy()
    d_m = d.to_period("M")

    # non-OECD series: keep obs whose real release date <= d
    for col in [c for c in NONOECD if c in m.columns]:
        m[col] = m[col].where(rel[col] <= d)

    # revisable series: OECD first-release vintage values where the edition has them, with the
    # newest calendar-available month (which OECD editions publish ~1 month late) filled by the
    # Eurostat value, and everything gated by the real release calendar (improvement #1).
    if em is not None:
        for col in [c for c in rb.OECD_COLS if c in m.columns]:
            v = rb.series_asof(rb.vintages(col), em, "M")
            v_oecd = (rb.transform(v, rb.TRANSFORM[col]).reindex(m.index)
                      if v is not None else pd.Series(np.nan, index=m.index))
            combined = v_oecd.where(v_oecd.notna(), m_final[col])   # fill OECD edge with Eurostat
            m[col] = combined.where(rel[col] <= d)                  # gate by day-level calendar
        gs = rb.series_asof(gv, em, "Q")
        g = (100 * np.log(gs).diff()).rename("gdp_qoq")
    else:
        g = pd.read_csv(BASE / "data/processed/gdp_quarterly.csv",
                        parse_dates=["date"]).set_index("date")["gdp_qoq"]
        g.index = g.index.to_period("Q")

    m = m.loc[:d_m]
    if SAMPLE_START is not None:
        g = g.loc[g.index >= SAMPLE_START]
    g = g.loc[g.index < target_q]                       # target withheld until its flash
    qr = pd.period_range(g.index.min(), target_q, freq="Q")
    g = g.reindex(qr)
    return m, g, em


# ---------------------------------------------------------------------------
def _context():
    """Shared inputs computed once (vintages, final panel, release calendar, warm-start)."""
    flash, final, gv, editions = rb.flash_and_final_gdp()
    m_final = pd.read_csv(BASE / "data/processed/monthly_panel.csv",
                          parse_dates=["date"]).set_index("date")
    m_final.index = m_final.index.to_period("M")
    if DROP_COLS:                                        # winner config: drop foreign IP etc.
        m_final = m_final.drop(columns=[c for c in DROP_COLS if c in m_final.columns])
    if SAMPLE_START is not None:                         # winner config: cut estimation window
        m_final = m_final.loc[m_final.index >= SAMPLE_START.asfreq("M", "start")]
    rel = wk.release_matrix(m_final)
    warm = base.build_model(*_final_fit_data(m_final)).fit(disp=0, maxiter=150).params
    return dict(flash=flash, final=final, gv=gv, editions=editions,
                m_final=m_final, rel=rel, warm=warm)


def _run_one(target_q: pd.Period, ctx):
    gv, editions = ctx["gv"], ctx["editions"]
    flash_ed = min(e for e in editions if target_q in _quarters_in(gv, e))
    end = edition_release(flash_ed)                    # stop at the flash release
    fridays = pd.date_range(target_q.to_timestamp(how="start"), end, freq="W-FRI")

    out, last_sig, last_nc = {}, None, np.nan
    for d in fridays:
        m, g, em = panel_asof(d, target_q, ctx["m_final"], ctx["rel"], gv, editions)
        sig = (em, int(m[[c for c in NONOECD if c in m.columns]].notna().sum().sum()))
        if sig != last_sig:
            try:
                res = base.build_model(m, g).fit(disp=0, maxiter=100, start_params=ctx["warm"])
                last_nc = base.gdp_nowcast(res, target_q)
            except Exception:  # noqa: BLE001
                pass
            last_sig = sig
        out[d] = last_nc
    return pd.Series(out, name="nowcast"), float(ctx["flash"][target_q]), float(ctx["final"][target_q]), end


def run(target_q: pd.Period):
    return _run_one(target_q, _context())


def _quarters_in(gv, ed):
    s = rb.series_asof(gv, ed, "Q")
    return set(s.dropna().index) if s is not None else set()


def _final_fit_data(m_final):
    g = pd.read_csv(BASE / "data/processed/gdp_quarterly.csv",
                    parse_dates=["date"]).set_index("date")["gdp_qoq"]
    g.index = g.index.to_period("Q")
    if SAMPLE_START is not None:
        g = g.loc[g.index >= SAMPLE_START]
    qr = pd.period_range(g.index.min(), m_final.index.max().asfreq("Q"), freq="Q")
    return m_final, g.reindex(qr)


# ---------------------------------------------------------------------------
def plot(path, flash, final, flash_date, target_q):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axhline(flash, color="crimson", ls="--", lw=2, label=f"Flash (first release) = {flash:+.2f}%")
    ax.axhline(final, color="tab:green", ls=":", lw=1.5, alpha=0.8,
               label=f"Final (revised today) = {final:+.2f}%")
    ax.step(path.index, path.values, where="post", color="tab:blue", lw=1.5)
    ax.plot(path.index, path.values, "o", color="tab:blue", ms=5,
            label="Weekly real-time nowcast (point-in-time vintages)")
    qend = target_q.to_timestamp(how="end")
    ax.axvline(qend, color="0.5", ls=":", lw=1); ax.text(qend, ax.get_ylim()[1], " quarter end", va="top", fontsize=8, color="0.4")
    ax.axvline(flash_date, color="crimson", ls=":", lw=1); ax.text(flash_date, ax.get_ylim()[1], " flash", va="top", fontsize=8, color="crimson")
    ax.set_ylabel("GDP growth nowcast, QoQ %")
    ax.set_title(f"Chronologically-correct weekly real-time nowcast — {target_q}"
                 + (f"  [{TAG.strip('_')}]" if TAG else ""))
    ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / f"weekly_realtime_{target_q}{TAG}.png", dpi=130)
    plt.close(fig)


def _panel_ax(ax, path, flash, final, flash_date, target_q):
    ax.axhline(flash, color="crimson", ls="--", lw=1.8, label=f"Flash = {flash:+.2f}%")
    ax.axhline(final, color="tab:green", ls=":", lw=1.5, label=f"Final = {final:+.2f}%")
    ax.step(path.index, path.values, where="post", color="tab:blue", lw=1.4)
    ax.plot(path.index, path.values, "o", color="tab:blue", ms=4)
    ax.axvline(target_q.to_timestamp(how="end"), color="0.5", ls=":", lw=0.9)
    ax.set_title(str(target_q), fontsize=11)
    ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.25)
    ax.set_ylabel("QoQ %")


def _intercept_correction(path: pd.Series, ctx, k: int = 4) -> pd.Series:
    """Real-time intercept correction: at each Friday, subtract the mean error of the last
    `k` quarters whose flash was already released by that date. Errors come from the audited
    quarterly real-time error history (outputs/audit_intercept_correction.csv)."""
    audit_f = OUT / "audit_intercept_correction.csv"
    if not audit_f.exists():
        return path * np.nan
    aud = pd.read_csv(audit_f).set_index("q")
    gv, editions = ctx["gv"], ctx["editions"]
    frel = {pd.Period(q, "Q"): _flash_release_of(pd.Period(q, "Q"), gv, editions)
            for q in aud.index}
    out = {}
    for d in path.index:
        known = [(q, aud.loc[str(q), "err"]) for q in frel if frel[q] < d]
        known.sort(key=lambda x: x[0])
        errs = [e for _, e in known[-k:]]
        c = float(np.mean(errs)) if len(errs) >= 2 else 0.0
        out[d] = path[d] - c
    return pd.Series(out)


def run_year(year: int, corrected: bool = False):
    ctx = _context()
    qs = [pd.Period(f"{year}Q{i}", "Q") for i in (1, 2, 3, 4)]
    results = {}
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharex=False)
    for q, ax in zip(qs, axes.ravel()):
        path, flash, final, fdate = _run_one(q, ctx)
        results[q] = (path, flash, final, fdate)
        path.to_csv(OUT / f"weekly_realtime_{q}{TAG}.csv")
        _panel_ax(ax, path, flash, final, fdate, q)
        if corrected:
            cpath = _intercept_correction(path, ctx)
            ax.step(cpath.index, cpath.values, where="post", color="tab:purple", lw=1.4, alpha=0.9)
            ax.plot(cpath.index, cpath.values, "s", color="tab:purple", ms=3.5,
                    label="intercept-corrected")
            ax.legend(loc="best", fontsize=8)
        upd = path.dropna()
        print(f"  {q}: {path.notna().sum():2d} Fridays -> flash {flash:+.2f}% final {final:+.2f}% "
              f"| nowcast {upd.iloc[0]:+.2f}% -> {upd.iloc[-1]:+.2f}%"
              + ("   [revised]" if abs(flash - final) > 0.05 else ""))
    tag = ("_corrected" if corrected else "") + TAG
    fig.suptitle(f"Chronologically-correct weekly real-time nowcasts — {year} "
                 f"(point-in-time OECD vintages; converge to the flash)"
                 + (f"  [{TAG.strip('_')}]" if TAG else ""), fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / f"weekly_realtime_{year}{tag}.png", dpi=130)
    plt.close(fig)
    print(f"  Wrote outputs/weekly_realtime_{year}{tag}.png")
    return results


def _flash_release_of(q, gv, editions):
    fe = min(e for e in editions if q in _quarters_in(gv, e))
    return edition_release(fe)


def continuous_timeline(year: int):
    """One weekly timeline: at each Friday, nowcast the current *live* quarter (the earliest
    quarter whose flash has not yet been released). Actuals drop in as each flash lands."""
    ctx = _context()
    gv, editions = ctx["gv"], ctx["editions"]
    qs = [pd.Period(f"{year}Q{i}", "Q") for i in (1, 2, 3, 4)]
    frel = {q: _flash_release_of(q, gv, editions) for q in qs}
    start = _flash_release_of(qs[0] - 1, gv, editions) + pd.Timedelta(days=1)
    fridays = pd.date_range(start, frel[qs[-1]], freq="W-FRI")

    recs, last_sig, last_nc = [], None, np.nan
    for d in fridays:
        live = [q for q in qs if frel[q] > d]
        if not live:
            break
        lq = live[0]
        m, g, em = panel_asof(d, lq, ctx["m_final"], ctx["rel"], gv, editions)
        sig = (em, lq, int(m[[c for c in NONOECD if c in m.columns]].notna().sum().sum()))
        if sig != last_sig:
            try:
                res = base.build_model(m, g).fit(disp=0, maxiter=100, start_params=ctx["warm"])
                last_nc = base.gdp_nowcast(res, lq)
            except Exception:  # noqa: BLE001
                pass
            last_sig = sig
        recs.append(dict(date=d, live_q=str(lq), nowcast=last_nc))
    df = pd.DataFrame(recs)
    flashes = {q: float(ctx["flash"][q]) for q in qs}
    finals = {q: float(ctx["final"][q]) for q in qs}
    _plot_timeline(df, flashes, finals, frel, qs, year)
    df.to_csv(OUT / f"weekly_timeline_{year}{TAG}.csv", index=False)
    print(f"  {len(df)} Fridays; live quarters {qs[0]}..{qs[-1]}")
    print(f"  Wrote outputs/weekly_timeline_{year}.png")
    return df


def _plot_timeline(df, flashes, finals, frel, qs, year):
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.tab10.colors
    for i, q in enumerate(qs):
        seg = df[df.live_q == str(q)]
        if seg.empty:
            continue
        ax.step(seg.date, seg.nowcast, where="post", color=colors[i], lw=1.6)
        ax.plot(seg.date, seg.nowcast, "o", color=colors[i], ms=4,
                label=f"nowcast {q}")
        # flash actual marker at release date
        ax.plot(frel[q], flashes[q], "D", color="black", ms=8, zorder=5)
        ax.annotate(f"{q} flash\n{flashes[q]:+.2f}%", (frel[q], flashes[q]),
                    textcoords="offset points", xytext=(6, 8), fontsize=8)
        ax.axvline(frel[q], color=colors[i], ls=":", lw=0.8, alpha=0.6)
    # realized flash path (black)
    fx = [frel[q] for q in qs]; fy = [flashes[q] for q in qs]
    ax.plot(fx, fy, "--", color="0.3", lw=1, alpha=0.7, label="flash actuals")
    ax.set_ylabel("GDP growth, QoQ %")
    ax.set_title(f"Continuous weekly real-time nowcast — {year} "
                 f"(each segment nowcasts the live quarter; diamonds = flash actuals)")
    ax.legend(loc="upper right", fontsize=8, ncol=2); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / f"weekly_timeline_{year}{TAG}.png", dpi=130)
    plt.close(fig)


def main():
    import sys as _sys
    args = [a for a in _sys.argv[1:] if not a.startswith("-")]
    if args and args[0] == "timeline":
        yr = int(args[1]) if len(args) > 1 and args[1].isdigit() else 2025
        print(f"Continuous weekly real-time timeline for {yr}...")
        continuous_timeline(yr)
    elif args and args[0].isdigit():
        year = int(args[0])
        corrected = "--corrected" in _sys.argv
        print(f"Chronological weekly real-time nowcasts for {year} (all four quarters"
              + (", intercept-corrected" if corrected else "") + ")...")
        run_year(year, corrected=corrected)
    else:
        target_q = pd.Period("2026Q1", "Q")
        print(f"Chronological weekly real-time nowcast for {target_q} (point-in-time OECD vintages)...")
        path, flash, final, flash_date = run(target_q)
        path.to_csv(OUT / f"weekly_realtime_{target_q}.csv")
        plot(path, flash, final, flash_date, target_q)
        upd = path.dropna()
        print(f"  {path.notna().sum()} Fridays through the flash ({flash_date.date()})")
        print(f"  first {upd.iloc[0]:+.2f}% -> last {upd.iloc[-1]:+.2f}% | "
              f"flash {flash:+.2f}% final {final:+.2f}%")


if __name__ == "__main__":
    main()
