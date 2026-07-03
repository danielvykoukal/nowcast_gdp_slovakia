"""
Koopman-Harvey observation weights of the Kalman smoother for the GDP nowcast, best
model (20 series, 2013+, no foreign IP).

The smoothed nowcast is linear in the observed data:
    nowcast = sum_{series k, month j} w_{k,j} * y_std_{k,j} + const(prior, means)
w_{k,j} = dZ_gdp' alpha_hat(t_end) / dy_{k,j}, computed exactly via
statsmodels.tsa.statespace.tools.compute_smoothed_state_weights. Weights depend on the
parameters and the missingness pattern (ragged edge), NOT on the data values - they are
"the weight the model assigns each observation month in the forecast".

Reported in pp of GDP QoQ per +1 standard deviation of a series' monthly observation.

Outputs:
  outputs/kalman_weights.csv        - full (series x month) weight matrix, pp per 1 sd
  outputs/kalman_weights.png        - cumulative |weight| per series + monthly profile
  console: cumulative weights, sector aggregation vs GVA shares

Run:  python3 src/kalman_weights.py
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
from statsmodels.tsa.statespace.tools import compute_smoothed_state_weights

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
OUT = BASE / "outputs"

import model as base

FOREIGN = ["ip_de", "ip_de_auto", "ip_ea"]
BUCKETS = {  # production-side sectors with a GVA counterpart (as in results_best.py)
    "Industry (B-E)": ["ip_total", "ip_manuf"],
    "Construction (F)": ["construction"],
    "Trade, transp., acc. (G-I)": ["retail_vol", "services_H", "services_iaf"],
    "ICT (J)": ["services_J"],
    "Prof. & admin (M_N)": ["services_N"],
}
GVA = {"Industry (B-E)": "B-E", "Construction (F)": "F",
       "Trade, transp., acc. (G-I)": "G-I", "ICT (J)": "J", "Prof. & admin (M_N)": "M_N"}

SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, AQUA, YELLOW = "#2a78d6", "#1baf7a", "#eda100"


def main():
    m, g = base.load_processed(start="2013-01")
    m = m.drop(columns=FOREIGN)
    target_q = g.index[-1]
    t_end = m.index.get_loc(target_q.asfreq("M", how="end"))
    print(f"Fitting best model; nowcast target {target_q}, t_end index {t_end}", flush=True)
    res = base.build_model(m, g).fit(disp=0, maxiter=200)

    # exact observation weights on the smoothed state at the quarter-end month
    w = compute_smoothed_state_weights(res, compute_t=[t_end])  # (1, nobs, k_states, k_endog)
    weights = np.nan_to_num(w[0] if isinstance(w, tuple) else w)[0]  # (nobs, k_states, k_endog)

    names = list(res.model.endog_names)
    Zg = np.asarray(res.model["design"])[names.index("gdp_qoq"), :]
    g_std = float(g.dropna().std())
    # weight of obs (month j, series k) on the nowcast, in pp of GDP per 1 sd of the obs
    W = pd.DataFrame(np.einsum("s,jsk->jk", Zg, weights) * g_std,
                     index=m.index, columns=names)
    W = W.drop(columns=["gdp_qoq"])  # keep monthly indicators (GDP history weight noted below)
    W.to_csv(OUT / "kalman_weights.csv")

    cum_abs = W.abs().sum().sort_values(ascending=False)
    cum_signed = W.sum()
    print("\n=== cumulative Kalman observation weights on the nowcast ===")
    print("(sum over all months; pp of GDP QoQ per 1-sd observation)")
    tbl = pd.DataFrame({"cum_|w|": cum_abs, "cum_w_signed": cum_signed[cum_abs.index],
                        "last3m_|w|": W.abs().tail(3).sum()[cum_abs.index],
                        "share_last3m_%": 100 * W.abs().tail(3).sum()[cum_abs.index] / cum_abs})
    print(tbl.round(3).to_string())

    # sector aggregation vs GVA shares
    gva = pd.read_csv(BASE / "data/raw/gva_shares.csv").set_index("nace_r2")["2025"]
    rows = []
    for sec, series in BUCKETS.items():
        rows.append(dict(sector=sec, kalman_w=cum_abs[series].sum(), gva=gva[GVA[sec]]))
    comp = pd.DataFrame(rows).set_index("sector")
    comp["kalman_weight_%"] = 100 * comp.kalman_w / comp.kalman_w.sum()
    comp["actual_gva_share_%"] = 100 * comp.gva / comp.gva.sum()
    comp["gap_pp"] = comp["kalman_weight_%"] - comp["actual_gva_share_%"]
    comp = comp[["kalman_weight_%", "actual_gva_share_%", "gap_pp"]].round(1)
    print("\n=== cumulative Kalman weights vs actual GVA shares (scoreable sectors) ===")
    print(comp.to_string())
    mw, aw = comp["kalman_weight_%"], comp["actual_gva_share_%"]
    print(f"alignment: spearman {mw.rank().corr(aw.rank()):.2f}, "
          f"pearson {np.corrcoef(mw, aw)[0, 1]:.3f}, mean |gap| {np.mean(np.abs(mw - aw)):.1f}pp")

    # ---- plot: cumulative |w| per series + weight-by-month profile ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), width_ratios=[1, 1.35],
                                   facecolor=SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        ax.set_axisbelow(True)

    ax1.grid(axis="x", color=GRID, lw=0.8)
    y = np.arange(len(cum_abs))[::-1]
    ax1.barh(y, cum_abs.values, 0.72, color=BLUE, edgecolor="none")
    ax1.set_yticks(y, cum_abs.index, fontsize=8)
    ax1.set_xlabel("cumulative |weight| (pp per 1-sd obs)", color=MUTED, fontsize=9)
    ax1.set_title("Cumulative Kalman weight by series", color=INK, fontsize=10.5, loc="left")

    ax2.grid(axis="y", color=GRID, lw=0.8)
    prof = W.abs().sum(axis=1)
    months = prof.index.to_timestamp()
    ax2.fill_between(months, prof.values, color=BLUE, alpha=0.25, lw=0)
    ax2.plot(months, prof.values, color=BLUE, lw=1.8)
    qend = target_q.asfreq("M", how="end").to_timestamp(how="end")
    ax2.axvline(qend, color=AXIS, ls=":", lw=1)
    ax2.set_xlim(months[max(0, len(months) - 37)], qend + pd.Timedelta(days=20))
    ax2.set_ylabel("total |weight| across series (pp per 1-sd)", color=MUTED, fontsize=9)
    ax2.set_title(f"Weight by observation month (last 3 years shown) — nowcast {target_q}",
                  color=INK, fontsize=10.5, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "kalman_weights.png", dpi=150, facecolor=SURFACE)
    print("\nWrote outputs/kalman_weights.csv, outputs/kalman_weights.png")


if __name__ == "__main__":
    main()
