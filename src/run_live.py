"""Live nowcast: run the frozen production spec (lean14 + real services + quarterly
vacancies) on the LATEST available data and print/write the current-quarter nowcast.

Unlike weekly_realtime.py (a point-in-time backtest that replays completed quarters and
needs the flash release to exist), this driver just fits the frozen spec on whatever is
published today and nowcasts the first quarter that has no official GDP yet. The target
auto-advances when the flash for that quarter is released and picked up by preprocess.

Run:  python3 src/run_live.py
Output: outputs/nowcast_<YYYY-MM-DD>.csv and outputs/nowcast_latest.csv
"""
from __future__ import annotations
import datetime as dt
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

import model as base

# --- frozen lean14v spec (mirrors weekly_lean14v.py) ---
DROP_COLS = ["ip_de", "ip_de_auto", "ip_ea", "hicp", "eur_usd", "bond_10y",
             "ip_total", "services_J", "construction"]
SAMPLE_START = pd.Period("2013Q1", "Q")

vac_raw = pd.read_csv(BASE / "data/raw/vacancies_q.csv").set_index("quarter")["value"]
vac_raw.index = pd.PeriodIndex(vac_raw.index, freq="Q")
VAC = (100 * np.log(vac_raw).diff()).rename("vacancies")


def build_model_vac(m: pd.DataFrame, g: pd.Series, target_q: pd.Period) -> DynamicFactorMQ:
    gq = g.to_frame()
    v = VAC.reindex(g.index)
    v[v.index >= target_q] = np.nan            # vacancies observed through target-1
    gq["vacancies"] = v
    return DynamicFactorMQ(m, endog_quarterly=gq, factors=1,
                           factor_orders=base.FACTOR_ORDERS, idiosyncratic_ar1=True)


def plot(g_hist: pd.Series, target_q: pd.Period, nowcast: float, benchmark: float,
         asof: str, n_hist: int = 12) -> None:
    """Recent actual GDP QoQ as bars + the current-quarter nowcast highlighted, with the
    naive mean-of-4-flashes benchmark as a reference line."""
    h = g_hist.dropna().iloc[-n_hist:]
    labels = [str(q) for q in h.index] + [str(target_q)]
    vals = list(h.values) + [nowcast]
    x = np.arange(len(vals))
    colors = ["0.7"] * len(h) + ["tab:blue"]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(x, vals, color=colors, width=0.7)
    ax.axhline(benchmark, color="crimson", ls="--", lw=1.4,
               label=f"Naive benchmark (mean of last 4) = {benchmark:+.2f}%")
    ax.bar_label(bars, fmt="%+.2f", padding=2, fontsize=8)
    ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_ylabel("GDP growth, QoQ %")
    ax.set_title(f"Slovak GDP nowcast — {target_q} = {nowcast:+.2f}%   (as of {asof}, lean14v spec)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / f"nowcast_{asof}.png", dpi=130)
    fig.savefig(OUT / "nowcast_latest.png", dpi=130)
    plt.close(fig)


def main() -> None:
    m = pd.read_csv(BASE / "data/processed/monthly_panel.csv",
                    parse_dates=["date"]).set_index("date")
    m.index = m.index.to_period("M")
    m = m.drop(columns=[c for c in DROP_COLS if c in m.columns])
    m = m.loc[m.index >= SAMPLE_START.asfreq("M", "start")]

    g = pd.read_csv(BASE / "data/processed/gdp_quarterly.csv",
                    parse_dates=["date"]).set_index("date")["gdp_qoq"]
    g.index = g.index.to_period("Q")
    g = g.loc[g.index >= SAMPLE_START].dropna()
    g_known = g.copy()                         # actual GDP history (before target padding)

    target_q = g.index.max() + 1               # first quarter with no official GDP yet
    benchmark = float(g_known.iloc[-4:].mean())  # naive mean-of-4 benchmark

    # extend both panels so the state space reaches the target quarter's end month
    qend_m = target_q.asfreq("M", how="end")
    m = m.reindex(pd.period_range(m.index.min(), max(m.index.max(), qend_m), freq="M"))
    g = g.reindex(pd.period_range(g.index.min(), target_q, freq="Q"))

    res = build_model_vac(m, g, target_q).fit(disp=0, maxiter=200)
    nowcast = base.gdp_nowcast(res, target_q)

    asof = dt.date.today().isoformat()
    n_monthly = int(m.loc[target_q.asfreq("M", "start"):qend_m].notna().sum().sum())
    row = pd.DataFrame([dict(asof=asof, target_q=str(target_q), nowcast=round(nowcast, 3),
                             monthly_obs_in_quarter=n_monthly, n_series=m.shape[1])])
    row.to_csv(OUT / f"nowcast_{asof}.csv", index=False)
    row.to_csv(OUT / "nowcast_latest.csv", index=False)
    plot(g_known, target_q, nowcast, benchmark, asof)
    print(f"{asof}  {target_q} nowcast = {nowcast:+.2f}% QoQ  (benchmark {benchmark:+.2f}%, "
          f"{n_monthly} monthly obs in quarter, {m.shape[1]} series)")
    print(f"Wrote outputs/nowcast_{asof}.csv/.png and outputs/nowcast_latest.csv/.png")


if __name__ == "__main__":
    main()
