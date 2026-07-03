"""
Plot the pseudo-real-time backtest track (actual GDP vs nowcast) and the current-quarter
nowcast, for the explicit state-space DFM (methodology model), with DynamicFactorMQ shown
as a cross-check.

Reads outputs/backtest_ssm.csv (actual, dfm_ssm, dfm_dfmq) and outputs/nowcast_ssm.csv,
produced by src/dfm_statespace.py. Writes outputs/nowcast_backtest.png.
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


def main() -> None:
    bt = pd.read_csv(OUT / "backtest_ssm.csv")
    nc = pd.read_csv(OUT / "nowcast_ssm.csv").iloc[0]
    x = pd.PeriodIndex(bt["quarter"], freq="Q").to_timestamp()

    rmse = lambda c: np.sqrt(((bt[c] - bt["actual"]) ** 2).mean())
    r_ssm, r_dfmq = rmse("dfm_ssm"), rmse("dfm_dfmq")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                   gridspec_kw={"height_ratios": [3, 1.4]})

    # --- top: actual vs nowcasts ---
    ax1.axhline(0, color="0.7", lw=0.8)
    ax1.plot(x, bt["actual"], color="black", lw=2.2, marker="o", ms=3, label="Actual GDP (QoQ %)")
    ax1.plot(x, bt["dfm_ssm"], color="tab:blue", lw=2, marker="o", ms=3,
             label=f"Explicit state-space DFM (RMSE {r_ssm:.2f})")
    ax1.plot(x, bt["dfm_dfmq"], color="tab:orange", lw=1.2, ls="--", alpha=0.8,
             label=f"DynamicFactorMQ cross-check (RMSE {r_dfmq:.2f})")

    nc_x = (pd.Period(bt["quarter"].iloc[-1], "Q") + 1).to_timestamp()
    ax1.plot(nc_x, nc["nowcast_qoq_pct"], marker="*", ms=20, color="crimson",
             mec="black", zorder=5,
             label=f"Nowcast {nc['target_quarter']} = {nc['nowcast_qoq_pct']:+.2f}%")
    ax1.annotate(f"{nc['nowcast_qoq_pct']:+.2f}%", (nc_x, nc["nowcast_qoq_pct"]),
                 textcoords="offset points", xytext=(8, 8), color="crimson", fontweight="bold")

    ax1.set_ylabel("GDP growth, QoQ %")
    ax1.set_title("Slovak GDP: explicit MFDFM (methodology) pseudo-real-time nowcast vs. actual")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.25)
    lo = min(bt["actual"].min(), bt["dfm_ssm"].min()) - 1
    hi = max(bt["actual"].max(), bt["dfm_ssm"].max()) + 1
    ax1.set_ylim(lo, hi)

    # --- bottom: nowcast error ---
    err = bt["dfm_ssm"] - bt["actual"]
    ax2.axhline(0, color="0.6", lw=0.8)
    ax2.bar(x, err, width=60, color=np.where(err >= 0, "tab:blue", "tab:red"), alpha=0.7)
    ax2.set_ylabel("Nowcast error\n(nowcast − actual)")
    ax2.set_title("Explicit-model nowcast error by quarter", fontsize=10)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    path = OUT / "nowcast_backtest.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Wrote {path}")
    print(f"Explicit SSM RMSE {r_ssm:.3f} | DFMQ cross-check {r_dfmq:.3f} | "
          f"nowcast {nc['target_quarter']} = {nc['nowcast_qoq_pct']:+.2f}%")


if __name__ == "__main__":
    main()
