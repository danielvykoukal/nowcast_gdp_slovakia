"""
Diagnostics plot for the enhanced model (src/model_v2.py):
  left  - nowcast sharpening within the quarter (RMSE by information set, ex-2020)
  right - news decomposition: each release's contribution to the current-quarter nowcast

Reads outputs/horizon_rmse.csv and outputs/news_v2.csv. Writes outputs/v2_diagnostics.png.
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
COVID = ["2020Q2", "2020Q3", "2020Q4"]


def main() -> None:
    hb = pd.read_csv(OUT / "horizon_rmse.csv", index_col=0)
    ex = ~hb.index.isin(COVID)
    rmse = lambda s, mask: np.sqrt(((hb.loc[mask, s] - hb.loc[mask, "actual"]) ** 2).mean())
    hs = ["h1", "h2", "h3"]
    rmse_all = [rmse(h, slice(None)) for h in hs]
    rmse_ex = [rmse(h, ex) for h in hs]

    news = pd.read_csv(OUT / "news_v2.csv", index_col=0)["nowcast_impact_pp"]
    news = news.reindex(news.abs().sort_values(ascending=False).index).head(10)[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # left: sharpening curve
    x = np.arange(3)
    ax1.plot(x, rmse_all, "o-", color="tab:gray", label="all quarters")
    ax1.plot(x, rmse_ex, "o-", color="tab:blue", lw=2, label="excluding 2020")
    ax1.set_xticks(x); ax1.set_xticklabels(["Month 1", "Month 2", "Month 3"])
    ax1.set_ylabel("Nowcast RMSE (pp)")
    ax1.set_title("Nowcast sharpening within the quarter")
    ax1.legend(); ax1.grid(alpha=0.25)
    ax1.set_ylim(0, max(rmse_all) * 1.15)

    # right: news contributions
    colors = ["tab:green" if v >= 0 else "tab:red" for v in news.values]
    ax2.barh(news.index, news.values, color=colors, alpha=0.8)
    ax2.axvline(0, color="0.5", lw=0.8)
    ax2.set_xlabel("Contribution to current nowcast (pp)")
    ax2.set_title("News decomposition — current-quarter nowcast")
    ax2.grid(alpha=0.25, axis="x")

    fig.tight_layout()
    path = OUT / "v2_diagnostics.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Wrote {path}")
    print(f"RMSE by horizon (ex-2020): {[round(v,3) for v in rmse_ex]}")


if __name__ == "__main__":
    main()
