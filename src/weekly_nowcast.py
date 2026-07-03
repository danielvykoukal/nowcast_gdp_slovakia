"""
Weekly nowcast evolution + backtest.

Simulates the real workflow: every Friday, assemble the data that has actually been
*released* by that date (using each series' publication lag in days), re-estimate the
dynamic factor model, and nowcast the target quarter's GDP growth. The nowcast is tracked
week by week and should converge to the realised value as more data for the quarter arrives.

Because the inputs are monthly, the nowcast updates when releases land (a few clusters per
quarter), so the weekly path is a step function - exactly how GDPNow / NY-Fed nowcast charts
behave. The model is re-estimated whenever the data vintage changes.

Outputs:
  outputs/weekly_path_<Q>.csv + outputs/weekly_nowcast_<Q>.png   (single-quarter convergence)
  outputs/weekly_backtest_rmse.csv + outputs/weekly_backtest.png (RMSE by weeks-to-quarter-end)
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

import model_v2 as v2  # noqa: E402  (best model: blocks + AR1 idiosyncratic)

# Publication lag in DAYS: when an observation for reference month M becomes available
# (approx. release date = end of month M + lag). Surveys arrive at month-end (~0);
# hard data ~40 days; financial monthly value shortly after month-end.
LAG_DAYS = {
    "ip_total": 40, "ip_manuf": 40, "retail_vol": 40, "construction": 45,
    "exports_vol": 40, "imports_vol": 40, "unemp_rate": 30,
    "ip_de": 40, "ip_de_auto": 40, "ip_ea": 45,
    "services_iaf": 60, "hicp": 17,
    "services_H": 60, "services_J": 60, "services_N": 60, "real_wage_bill": 55,
    "esi_sk": 0, "ind_conf_sk": 0, "cons_conf_sk": 0, "esi_de": 0, "esi_ea": 30,
    "bond_10y": 5, "eur_usd": 5,
}
GDP_FLASH_DAYS = 45   # quarterly flash GDP released ~45 days after quarter end
WINDOW_AFTER = 55     # follow the nowcast this many days past quarter-end (through the flash)


# ---------------------------------------------------------------------------
def release_matrix(m: pd.DataFrame) -> pd.DataFrame:
    """Timestamp each monthly observation with its approximate release date."""
    month_end = m.index.to_timestamp(how="end").normalize()
    rel = pd.DataFrame(index=m.index, columns=m.columns, dtype="datetime64[ns]")
    for c in m.columns:
        rel[c] = month_end + pd.Timedelta(days=LAG_DAYS[c])
    return rel


def gdp_release(q: pd.Period) -> pd.Timestamp:
    return q.to_timestamp(how="end").normalize() + pd.Timedelta(days=GDP_FLASH_DAYS)


def vintage_asof(m, g, rel, d: pd.Timestamp):
    """Data available as of Friday `d`."""
    mv = m.where(rel <= d)
    gv = g.copy()
    for q in gv.index:
        if gdp_release(q) > d:
            gv[q] = np.nan
    return mv, gv


def weekly_path(m, g, rel, target_q: pd.Period, warm_params) -> pd.Series:
    """Weekly (Friday) nowcast of `target_q`; re-estimates only when the vintage changes.
    The target quarter's own GDP is always withheld (it is what we are nowcasting); the path
    runs from the start of the quarter to the flash-GDP release date."""
    start = target_q.to_timestamp(how="start")
    end = gdp_release(target_q) + pd.Timedelta(days=3)
    fridays = pd.date_range(start, end, freq="W-FRI")

    out, last_sig, last_nc = {}, None, np.nan
    for d in fridays:
        mv, gv = vintage_asof(m, g, rel, d)
        gv[target_q] = np.nan                     # never use the actual we are nowcasting
        sig = (int(mv.notna().sum().sum()), int(gv.notna().sum()))
        if sig != last_sig:                       # vintage changed -> re-estimate
            try:
                res = v2.build_model(mv, gv).fit(disp=0, maxiter=80, start_params=warm_params)
                last_nc = v2.nowcast_value(res, target_q)
            except Exception:  # noqa: BLE001
                pass
            last_sig = sig
        out[d] = last_nc
    return pd.Series(out, name="nowcast")


# ---------------------------------------------------------------------------
def plot_single(path: pd.Series, actual: float, target_q: pd.Period) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axhline(actual, color="crimson", ls="--", lw=2,
               label=f"Actual {target_q} = {actual:+.2f}%")
    ax.step(path.index, path.values, where="post", color="tab:blue", lw=1.5)
    ax.plot(path.index, path.values, "o", color="tab:blue", ms=5,
            label="Weekly nowcast (re-estimated each Friday)")
    qend = target_q.to_timestamp(how="end")
    ax.axvline(qend, color="0.5", ls=":", lw=1)
    ax.text(qend, ax.get_ylim()[1], " quarter end", va="top", fontsize=8, color="0.4")
    flash = gdp_release(target_q)
    ax.axvline(flash, color="green", ls=":", lw=1)
    ax.text(flash, ax.get_ylim()[1], " flash GDP", va="top", fontsize=8, color="green")
    ax.set_ylabel("GDP growth nowcast, QoQ %")
    ax.set_title(f"Weekly nowcast evolution for {target_q} — convergence to the actual")
    ax.legend(loc="best"); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / f"weekly_nowcast_{target_q}.png", dpi=130)
    plt.close(fig)


def backtest(m, g, rel, warm_params, start_q="2016Q1", end_q="2025Q4"):
    """Weekly nowcast path for each historical quarter; align errors by weeks-to-quarter-end."""
    actual = g.dropna()
    qs = [q for q in actual.index
          if pd.Period(start_q, "Q") <= q <= pd.Period(end_q, "Q")]
    err_by_week = {}   # weeks_rel_qend -> list of errors
    paths = {}
    for q in qs:
        p = weekly_path(m, g, rel, q, warm_params)
        a = float(actual.loc[q])
        paths[q] = p - a
        qend = q.to_timestamp(how="end")
        for d, nc in p.items():
            if np.isfinite(nc):
                wk = int(round((d - qend).days / 7.0))
                err_by_week.setdefault(wk, []).append(nc - a)
    rmse = {wk: float(np.sqrt(np.mean(np.square(v)))) for wk, v in err_by_week.items()}
    rmse_s = pd.Series(rmse).sort_index()
    rmse_s.index.name = "weeks_rel_quarter_end"
    return rmse_s, paths


def plot_backtest(rmse_s: pd.Series, paths: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    # left: individual convergence (error) paths by weeks-to-quarter-end
    for q, p in paths.items():
        qend = q.to_timestamp(how="end")
        wk = [(d - qend).days / 7.0 for d in p.index]
        ax1.plot(wk, p.values, color="0.7", lw=0.7, alpha=0.6)
    ax1.axhline(0, color="crimson", lw=1.5)
    ax1.axvline(0, color="0.4", ls=":", lw=1)
    ax1.set_xlabel("weeks relative to quarter end")
    ax1.set_ylabel("nowcast − actual (pp)")
    ax1.set_title("Every quarter's weekly error path → converges to 0")
    ax1.grid(alpha=0.25); ax1.set_ylim(-6, 6)
    # right: RMSE by week
    ax2.plot(rmse_s.index, rmse_s.values, "o-", color="tab:blue", lw=2)
    ax2.axvline(0, color="0.4", ls=":", lw=1)
    ax2.text(0, ax2.get_ylim()[1], " quarter end", va="top", fontsize=8, color="0.4")
    ax2.set_xlabel("weeks relative to quarter end")
    ax2.set_ylabel("nowcast RMSE (pp)")
    ax2.set_title("Nowcast RMSE by lead time (all quarters)")
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "weekly_backtest.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    m, g = v2.load_processed()
    rel = release_matrix(m)
    warm = v2.build_model(m, g).fit(disp=0, maxiter=200).params

    # --- single-quarter convergence (last completed quarter with an actual) ---
    target_q = g.dropna().index[-1]
    actual = float(g.dropna().iloc[-1])
    print(f"Weekly nowcast evolution for {target_q} (actual {actual:+.2f}%), re-estimating each Friday...")
    path = weekly_path(m, g, rel, target_q, warm)
    path.to_csv(OUT / f"weekly_path_{target_q}.csv")
    plot_single(path, actual, target_q)
    upd = path.dropna()
    print(f"  {path.notna().sum()} Fridays, "
          f"first nowcast {upd.iloc[0]:+.2f}% -> final {upd.iloc[-1]:+.2f}% (actual {actual:+.2f}%)")
    print(f"  Wrote outputs/weekly_nowcast_{target_q}.png")

    # --- backtest across history ---
    print("\nWeekly backtest across quarters (2016Q1-2025Q4)...")
    rmse_s, paths = backtest(m, g, rel, warm)
    rmse_s.to_frame("rmse").to_csv(OUT / "weekly_backtest_rmse.csv")
    plot_backtest(rmse_s, paths)
    for wk in [-10, -6, -2, 0, 2, 6]:
        if wk in rmse_s.index:
            print(f"  RMSE at {wk:+d} weeks vs quarter end: {rmse_s[wk]:.3f}")
    print("  Wrote outputs/weekly_backtest.png, outputs/weekly_backtest_rmse.csv")


if __name__ == "__main__":
    main()
