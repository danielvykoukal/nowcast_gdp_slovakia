"""
Weights of the BEST model (20 series = 23 minus foreign IP, estimated 2013+) matched
to Slovakia's actual GVA shares (Eurostat nama_10_a10, current prices, cached in
data/raw/gva_shares.csv).

The expanded panel maps five production-side sectors (the original analysis in
results.py could only score three):

  Industry (B-E)                     ip_total, ip_manuf
  Construction (F)                   construction
  Trade, transport, acc. & food (G-I) retail_vol, services_H, services_iaf
  Information & communication (J)    services_J
  Professional & admin (M_N)         services_N

Outputs: outputs/weights_vs_shares_best.csv/.png + console tables.
Run:  python3 src/results_best.py
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

FOREIGN = ["ip_de", "ip_de_auto", "ip_ea"]
BUCKETS = {
    "Industry (B-E)": {"nace": "B-E", "series": ["ip_total", "ip_manuf"]},
    "Construction (F)": {"nace": "F", "series": ["construction"]},
    "Trade, transp., acc. (G-I)": {"nace": "G-I",
                                   "series": ["retail_vol", "services_H", "services_iaf"]},
    "ICT (J)": {"nace": "J", "series": ["services_J"]},
    "Prof. & admin (M_N)": {"nace": "M_N", "series": ["services_N"]},
}
AUX = {
    "External trade": ["exports_vol", "imports_vol"],
    "Income / labour": ["real_wage_bill", "unemp_rate"],
    "Domestic demand": ["hicp"],
    "Sentiment": ["esi_sk", "ind_conf_sk", "cons_conf_sk", "esi_de", "esi_ea"],
    "Financial": ["bond_10y", "eur_usd"],
}

SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, AQUA, YELLOW = "#2a78d6", "#1baf7a", "#eda100"


def main():
    m, g = base.load_processed(start="2013-01")
    m = m.drop(columns=FOREIGN)
    print(f"Fitting best model ({m.shape[1]} series, 2013+)...", flush=True)
    res = base.build_model(m, g).fit(disp=0, maxiter=200)
    loads = base.extract_loadings(res)
    loads.to_csv(OUT / "loadings_best.csv")

    gva = pd.read_csv(BASE / "data/raw/gva_shares.csv").set_index("nace_r2")["2025"]
    absload, sn = loads["loading"].abs(), loads["sn_weight"]

    rows = []
    for bucket, cfg in BUCKETS.items():
        rows.append(dict(sector=bucket,
                         model_loading=absload[cfg["series"]].sum(),
                         model_sn=sn[cfg["series"]].sum(),
                         gva_meur=gva[cfg["nace"]]))
    comp = pd.DataFrame(rows).set_index("sector")
    out = pd.DataFrame(index=comp.index)
    out["model_weight_loading_%"] = 100 * comp.model_loading / comp.model_loading.sum()
    out["model_weight_sn_%"] = 100 * comp.model_sn / comp.model_sn.sum()
    out["actual_gva_share_%"] = 100 * comp.gva_meur / comp.gva_meur.sum()
    out["gap_loading_pp"] = out["model_weight_loading_%"] - out["actual_gva_share_%"]
    out["gap_sn_pp"] = out["model_weight_sn_%"] - out["actual_gva_share_%"]
    out = out.round(1)
    out.to_csv(OUT / "weights_vs_shares_best.csv")

    print("\n=== model weights vs actual GVA shares (5 scoreable sectors, 2025 GVA) ===")
    print(out.to_string())
    for tag, col in [("loadings", "model_weight_loading_%"), ("signal-to-noise", "model_weight_sn_%")]:
        mw, aw = out[col], out["actual_gva_share_%"]
        print(f"alignment ({tag}): spearman {mw.rank().corr(aw.rank()):.2f}, "
              f"pearson {np.corrcoef(mw, aw)[0,1]:.3f}, mean |gap| {np.mean(np.abs(mw-aw)):.1f}pp")

    # full weight table
    sector_of = {"gdp_qoq": "GDP (target)"}
    for b, cfg in BUCKETS.items():
        for s in cfg["series"]:
            sector_of[s] = b
    for b, ss in AUX.items():
        for s in ss:
            sector_of[s] = b + " (aux)"
    full = loads.copy()
    full["sector"] = [sector_of.get(s, "") for s in full.index]
    full["abs_loading"] = full["loading"].abs()
    print("\n=== full weight table (best model) ===")
    print(full[["sector", "abs_loading", "sn_weight"]]
          .sort_values("abs_loading", ascending=False).round(3).to_string())

    # plot
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    x = np.arange(len(out))
    w = 0.27
    for off, col, c, lab in [(-w, "model_weight_loading_%", BLUE, "Model weight (loadings)"),
                             (0, "model_weight_sn_%", AQUA, "Model weight (signal-to-noise)"),
                             (w, "actual_gva_share_%", YELLOW, "Actual GVA share 2025")]:
        bars = ax.bar(x + off, out[col], w * 0.92, color=c, edgecolor="none", label=lab)
        for b_, v in zip(bars, out[col]):
            ax.annotate(f"{v:.0f}", (b_.get_x() + b_.get_width() / 2, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, [s.replace(" (", "\n(") for s in out.index], fontsize=8.5)
    ax.set_ylabel("share of compared sectors (%)", color=MUTED, fontsize=9)
    ax.set_title("Best model (20 series, 2013+): sector weights vs actual GVA shares",
                 color=INK, fontsize=11, loc="left")
    ax.legend(fontsize=8.5, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "weights_vs_shares_best.png", dpi=150, facecolor=SURFACE)
    print("\nWrote outputs/weights_vs_shares_best.{csv,png}, outputs/loadings_best.csv")


if __name__ == "__main__":
    main()
