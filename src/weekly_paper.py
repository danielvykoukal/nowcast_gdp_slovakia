"""
2025 real-time nowcast paths: current model vs the paper-faithful FLASH model.

For each 2025 quarter we walk the information date month-by-month across the paper's 9-month
window (two quarters before -> one quarter after the target) and, at each date, nowcast the
target quarter with TWO fixed-parameter models:

  * current  = src/dfm_statespace.py  (single factor, scalar white-noise idio, GDP only)
  * paper    = src/dfm_paper.py C4     (AR(1) idio, soft-YoY, + a real-time FLASH GDP row)

Both use full-sample parameters and only *smooth* each point-in-time vintage (fast). GDP `final`
is withheld until ~3 months after the quarter (paper's "second" release lag); the paper model
additionally receives the real-time `flash` value the moment its OECD edition is published, which
is the whole point -- you can see the paper path jump toward the truth when the flash lands.

Output: outputs/weekly_paper_2025.png, outputs/weekly_paper_2025.csv
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

import dfm_paper as P
import dfm_statespace as S

FINAL_LAG_M = 3     # GDP "second/final" published ~3 months after the quarter (paper: 102 days)


def flash_editions() -> dict:
    """Release month-end for each quarter's FLASH (earliest OECD edition that reports it)."""
    df = pd.read_csv(P.VDIR / "gdp_level.csv")
    df["q"] = df["time_period"].str.replace("-", "").apply(lambda s: pd.Period(s, "Q"))
    rel = {}
    for q in sorted(df["q"].unique()):
        e0 = int(df.loc[df["q"] == q, "edition"].min())
        y, mm = divmod(e0, 100)
        rel[q] = pd.Period(f"{y}-{mm:02d}", "M").to_timestamp(how="end").normalize() + pd.offsets.MonthEnd(0)
    return rel


def q_end_month(q: pd.Period) -> pd.Timestamp:
    return q.asfreq("M", how="end").to_timestamp(how="end").normalize() + pd.offsets.MonthEnd(0)


# ---------------------------------------------------------------------------
def vintage(m, g_final, flash, flash_rel, ref_m, target_q):
    """Point-in-time panel + GDP as knowable at end of month `ref_m`, extended to the target
    quarter-end so the smoother produces the nowcast there."""
    panel_end = max(ref_m, q_end_month(target_q))
    idx = pd.date_range(m.index.min(), panel_end, freq="ME")
    mv = m.reindex(idx)
    mv.loc[mv.index > ref_m] = np.nan                        # nothing after the info date
    for col, lag in P.PUB_LAG.items():                        # publication lags vs the info date
        if lag > 0 and col in mv:
            cut = ref_m - pd.offsets.MonthEnd(lag)
            mv.loc[mv.index > cut, col] = np.nan

    gv = g_final.reindex(idx.intersection(g_final.index)) if False else g_final.copy()
    gv = gv.reindex(pd.DatetimeIndex(mv.index))
    for t in list(gv.index):                                  # final GDP known ~FINAL_LAG_M later
        if pd.notna(gv[t]) and (t + pd.offsets.MonthEnd(FINAL_LAG_M)) > ref_m:
            gv[t] = np.nan
    gv[gv.index > q_end_month(target_q) - pd.offsets.MonthEnd(3)] = np.nan  # withhold target qtr

    fv = None
    if flash is not None:
        fv = pd.Series({q: v for q, v in flash.items() if flash_rel.get(q, pd.Timestamp.max) <= ref_m})
    return mv, gv, fv


def paper_nowcast(mv, gv, fv, cfg, params, mom, target_month):
    endog = P.make_endog(mv, gv, mom, cfg, fv)
    has_flash = cfg.multi_release and "gdp_flash" in endog.columns
    mod = P.PaperDFM(endog, list(mv.columns), cfg, has_flash)
    return P.gdp_signal(mod.smooth(params), target_month, mom)


def current_nowcast(mv, gv, params, momS, target_month):
    endog = S.make_endog(mv, gv, momS)
    mod = S.MMDynamicFactor(endog, n_monthly=mv.shape[1])
    return S.gdp_nowcast(mod.smooth(params), target_month, momS)


# ---------------------------------------------------------------------------
def build():
    m, g = P.load_raw()
    flash = P.build_flash()
    flash_rel = flash_editions()
    mom = P.moments(m, g, flash)
    momS = S.moments(m, g)

    print("Fitting full-sample parameters (paper C4 + current)...")
    cfg = P.Config("C4 FULL", True, True, True, True)
    _, res_paper, _ = P.fit_model(m, g, mom, cfg, flash)
    endogS = S.make_endog(m, g, momS)
    res_cur = S.MMDynamicFactor(endogS, n_monthly=m.shape[1]).fit(disp=False, maxiter=200)
    return dict(m=m, g=g, flash=flash, flash_rel=flash_rel, mom=mom, momS=momS,
                cfg=cfg, p_paper=res_paper.params, p_cur=res_cur.params)


def run_quarter(q, ctx):
    tm = q_end_month(q)
    refs = pd.date_range(q_end_month(q - 2), q_end_month(q + 1), freq="ME")
    rows = []
    for ref in refs:
        mv, gv, fv = vintage(ctx["m"], ctx["g"], ctx["flash"], ctx["flash_rel"], ref, q)
        try:
            ncp = paper_nowcast(mv, gv, fv, ctx["cfg"], ctx["p_paper"], ctx["mom"], tm)
        except Exception:  # noqa: BLE001
            ncp = np.nan
        try:
            ncc = current_nowcast(mv, gv.dropna().reindex(mv.index), ctx["p_cur"], ctx["momS"], tm)
        except Exception:  # noqa: BLE001
            ncc = np.nan
        rows.append(dict(ref=ref, current=ncc, paper=ncp))
    df = pd.DataFrame(rows).set_index("ref")
    return df


# ---------------------------------------------------------------------------
def main():
    ctx = build()
    year = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 2025
    qs = [pd.Period(f"{year}Q{i}", "Q") for i in (1, 2, 3, 4)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    allrows = []
    for q, ax in zip(qs, axes.ravel()):
        df = run_quarter(q, ctx)
        flash_v = float(ctx["flash"][q]) if q in ctx["flash"] else np.nan
        final_v = float(ctx["g"].dropna().reindex([q_end_month(q)]).iloc[0]) \
            if q_end_month(q) in ctx["g"].index else np.nan
        frel = ctx["flash_rel"].get(q)
        df["quarter"] = str(q); allrows.append(df.reset_index())

        ax.step(df.index, df["current"], where="post", color="tab:blue", lw=1.5, label="current model")
        ax.plot(df.index, df["current"], "o", color="tab:blue", ms=3.5)
        ax.step(df.index, df["paper"], where="post", color="tab:orange", lw=1.5, label="paper + flash")
        ax.plot(df.index, df["paper"], "s", color="tab:orange", ms=3.5)
        if not np.isnan(flash_v):
            ax.axhline(flash_v, color="crimson", ls="--", lw=1.4, label=f"flash {flash_v:+.2f}%")
        if not np.isnan(final_v):
            ax.axhline(final_v, color="tab:green", ls=":", lw=1.4, label=f"final {final_v:+.2f}%")
        if frel is not None:
            ax.axvline(frel, color="crimson", ls=":", lw=1)
            ax.text(frel, ax.get_ylim()[1], " flash out", va="top", fontsize=7, color="crimson")
        ax.axvline(q_end_month(q), color="0.5", ls=":", lw=0.9)
        ax.set_title(str(q), fontsize=11); ax.set_ylabel("QoQ %")
        ax.legend(loc="best", fontsize=7); ax.grid(alpha=0.25)

    fig.suptitle(f"Real-time nowcast paths {year} — current model vs paper-faithful FLASH model\n"
                 "(orange jumps toward the truth when the flash release lands)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / f"weekly_paper_{year}.png", dpi=130)
    plt.close(fig)
    pd.concat(allrows).to_csv(OUT / f"weekly_paper_{year}.csv", index=False)
    print(f"Wrote outputs/weekly_paper_{year}.png and .csv")


if __name__ == "__main__":
    main()
