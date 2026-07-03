"""
Data-combination x estimation-window experiment for the 2025 level bias.

The 2025 nowcasts overshoot the flash by ~1pp. BIAS.md traced this to (a) the foreign
German/EA industrial block dominating the factor while Slovak domestic demand decoupled, and
(b) a regime shift (trend growth halved) leaving the full-sample model anchored to an old,
higher mean. AUDIT.md warned that fixing 2025 by dropping foreign IP + cutting the sample
*overfits* — it breaks the untouched 2022-2024 quarters.

So this script sweeps {data combo} x {estimation window} and scores EACH on three COVID-free
windows, to find a config that lowers the 2025 bias WITHOUT wrecking validation:

  * 2025    = 2025Q1..2026Q1   (the biased window we want to fix)
  * val     = 2022Q1..2024Q4   (must NOT break — the overfitting guard)
  * normal  = 2016Q1..2019Q4   (pre-COVID sanity)

Engine: DynamicFactorMQ (fixed full-config params, pseudo-real-time vintages with publication
lags — same protocol as src/model.py). Blocks let us test "foreign in its own factor" so it
cannot dominate the domestic cycle.

Output: outputs/regime_experiment.csv, outputs/regime_experiment.png
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
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
OUT = BASE / "outputs"
import model as base   # load_processed, PUB_LAG, gdp_nowcast

FOREIGN = ["ip_de", "ip_de_auto", "ip_ea"]
CONSTR_EXP = ["construction", "exports_vol", "imports_vol"]
SOFT = ["esi_sk", "ind_conf_sk", "cons_conf_sk", "esi_de", "esi_ea"]
DOMESTIC_CORE = ["ip_total", "ip_manuf", "retail_vol", "unemp_rate", "construction"]
# panel expansion (fetch_new_data.py)
WAGE = ["real_wage_bill"]
NEWSERV = ["services_H", "services_J", "services_N"]
NEW = WAGE + NEWSERV

W25 = [pd.Period(f"2025Q{i}") for i in (1, 2, 3, 4)] + [pd.Period("2026Q1")]
WVAL = [pd.Period(f"{y}Q{i}") for y in (2022, 2023, 2024) for i in (1, 2, 3, 4)]
WNORM = [pd.Period(f"{y}Q{i}") for y in (2016, 2017, 2018, 2019) for i in (1, 2, 3, 4)]
ALLQ = WNORM + WVAL + W25


def build_single(m, g):
    return DynamicFactorMQ(m, endog_quarterly=g.to_frame(), factors=1,
                           factor_orders=2, idiosyncratic_ar1=True)


def build_foreign_block(m, g):
    """All series on a Global factor; the foreign IP block ALSO gets its own factor, so foreign
    comovement is captured there instead of dominating the domestic cycle that maps to GDP."""
    factors = {}
    for c in m.columns:
        factors[c] = ["Global", "Foreign"] if c in FOREIGN else ["Global"]
    factors["gdp_qoq"] = ["Global"]
    return DynamicFactorMQ(m, endog_quarterly=g.to_frame(), factors=factors,
                           factor_orders={"Global": 2, "Foreign": 1}, idiosyncratic_ar1=True)


COMBOS = {
    # baseline winner (drop foreign) WITHOUT any new data, to isolate the new series' effect
    "winner_base (no new data)":  (lambda cols: [c for c in cols if c not in FOREIGN + NEW], build_single),
    "winner + wage_bill":         (lambda cols: [c for c in cols if c not in FOREIGN + NEWSERV], build_single),
    "winner + new services":      (lambda cols: [c for c in cols if c not in FOREIGN + WAGE], build_single),
    "winner + all new data":      (lambda cols: [c for c in cols if c not in FOREIGN], build_single),
    "full + all new data":        (lambda cols: cols, build_single),
    "domestic+income":            (lambda cols: DOMESTIC_CORE + SOFT + NEW, build_single),
}
PERIODS = {"2013+": pd.Period("2013Q1"), "2016+": pd.Period("2016Q1")}


def make_vintage(m, g, target_q, start):
    ref = target_q.asfreq("M", how="end")
    mv = m.loc[:ref].copy()
    for col, lag in base.PUB_LAG.items():
        if lag > 0 and col in mv:
            mv.loc[mv.index > (ref - lag), col] = np.nan
    gv = g.copy()
    gv[gv.index >= target_q] = np.nan
    if start is not None:
        mv = mv.loc[mv.index >= start.asfreq("M", "start")]
        gv = gv.loc[gv.index >= start]
    gv = gv.loc[:target_q]
    return mv, gv


def run_config(m, g, colfn, builder, start):
    cols = colfn(list(m.columns))
    mc = m[cols]
    # fit once on the (period-cut) full data
    mf = mc if start is None else mc.loc[mc.index >= start.asfreq("M", "start")]
    gf = g if start is None else g.loc[g.index >= start]
    qr = pd.period_range(gf.index.min(), mf.index.max().asfreq("Q"), freq="Q")
    params = builder(mf, gf.reindex(qr)).fit(disp=0, maxiter=150).params
    # backtest the evaluation quarters with fixed params
    rows = {}
    for q in ALLQ:
        if start is not None and q < start:
            rows[q] = np.nan
            continue
        mv, gv = make_vintage(mc, g, q, start)
        try:
            rows[q] = base_gdp(builder(mv, gv).smooth(params), q)
        except Exception:  # noqa: BLE001
            rows[q] = np.nan
    return pd.Series(rows)


def base_gdp(res, q):
    pm = res.predict()
    return float(pm.loc[q.asfreq("M", how="end"), "gdp_qoq"])


def metrics(nc: pd.Series, actual: pd.Series):
    def stat(win):
        d = pd.DataFrame({"nc": nc.reindex(win), "a": actual.reindex(win)}).dropna()
        if d.empty:
            return np.nan, np.nan
        err = d["nc"] - d["a"]
        return float(err.mean()), float(np.sqrt((err ** 2).mean()))
    b25, r25 = stat(W25)
    bv, rv = stat(WVAL)
    bn, rn = stat(WNORM)
    return dict(bias_2025=b25, rmse_2025=r25, bias_val=bv, rmse_val=rv,
                bias_norm=bn, rmse_norm=rn)


def main():
    m, g = base.load_processed()
    actual = g.dropna()
    print(f"Sweep: {len(COMBOS)} data combos x {len(PERIODS)} windows = "
          f"{len(COMBOS)*len(PERIODS)} configs\n")
    rows = []
    for cname, (colfn, builder) in COMBOS.items():
        for pname, start in PERIODS.items():
            nc = run_config(m, g, colfn, builder, start)
            mt = metrics(nc, actual)
            rows.append(dict(data=cname, period=pname, **{k: round(v, 3) for k, v in mt.items()}))
            print(f"  {cname:20s} {pname:6s} | 2025 bias {mt['bias_2025']:+.2f} rmse {mt['rmse_2025']:.2f}"
                  f" | val bias {mt['bias_val']:+.2f} rmse {mt['rmse_val']:.2f}"
                  f" | norm rmse {mt['rmse_norm']:.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "regime_experiment.csv", index=False)

    # score: want small |2025 bias| AND small val RMSE (guard against overfit)
    df["score"] = df["bias_2025"].abs() + df["rmse_val"]
    best = df.sort_values("score").head(5)
    print("\nBest configs (min |2025 bias| + val RMSE):")
    print(best[["data", "period", "bias_2025", "rmse_2025", "bias_val", "rmse_val", "rmse_norm"]].to_string(index=False))
    _plot(df)
    print("\nWrote outputs/regime_experiment.{csv,png}")


def _plot(df):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, (metric, title) in zip(axes, [("bias_2025", "2025 bias (pp, 0 = good)"),
                                           ("rmse_val", "2022-24 validation RMSE (overfit guard)")]):
        piv = df.pivot(index="data", columns="period", values=metric)
        im = ax.imshow(piv.values, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                ax.text(j, i, f"{v:+.2f}" if not np.isnan(v) else "-", ha="center", va="center", fontsize=9)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Data combination x estimation window — 2025 bias vs validation RMSE", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "regime_experiment.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
