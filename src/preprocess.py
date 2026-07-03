"""
Part 2a - Preprocessing for the Slovak GDP nowcasting MFDFM.

Loads the raw series, applies per-series stationarity transforms (per section 8 of
the research doc), and builds two aligned, month-end-indexed frames:

  * monthly_panel  - the monthly indicators (stationary), one column per series
  * gdp_quarterly  - the quarterly GDP QoQ growth, indexed on quarter-end months

`statsmodels` DynamicFactorMQ standardizes internally, so we do NOT standardize here;
we only make each series stationary and put everything on a common monthly calendar.

Outputs data/processed/monthly_panel.csv and data/processed/gdp_quarterly.csv, plus
a transforms.json describing what was applied (used by Part 3 for interpretation).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
PROC = BASE / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

# Transform code per monthly series.
#   dlog  = 100 * log-difference (month-on-month % growth)
#   diff  = first difference (level change)
#   level = leave as is (already stationary balance/index)
TRANSFORMS: dict[str, str] = {
    "ip_total": "dlog",
    "ip_manuf": "dlog",
    "retail_vol": "dlog",
    "construction": "dlog",
    "exports_vol": "dlog",
    "imports_vol": "dlog",
    "unemp_rate": "diff",
    "esi_sk": "level",
    "ind_conf_sk": "level",
    "cons_conf_sk": "level",
    "esi_de": "level",
    "esi_ea": "level",
    "ip_de": "dlog",
    "ip_de_auto": "dlog",
    "ip_ea": "dlog",
    "services_iaf": "dlog",
    "hicp": "dlog",
    "bond_10y": "diff",
    "eur_usd": "dlog",
    # ---- panel expansion (see fetch_new_data.py) ----
    "services_H": "dlog_real",  # transport turnover
    "services_J": "dlog_real",  # ICT turnover
    "services_N": "dlog_real",  # administrative services turnover
    "real_wage_bill": "level",  # already a YoY real-growth rate (stationary) -> identity
}

# Eurostat publishes services turnover (NETTUR) only in nominal terms for SK (no VOL_SLS);
# dlog_real deflates the MoM growth by HICP inflation so these enter as real growth like
# every other activity series. services_iaf is switched to dlog_real for the same reason.
TRANSFORMS["services_iaf"] = "dlog_real"

# Order columns by economic block for readable output.
MONTHLY_ORDER = [
    "ip_total", "ip_manuf", "retail_vol", "construction",
    "exports_vol", "imports_vol", "unemp_rate",
    "ip_de", "ip_de_auto", "ip_ea",
    "services_iaf", "services_H", "services_J", "services_N", "real_wage_bill", "hicp",
    "esi_sk", "ind_conf_sk", "cons_conf_sk", "esi_de", "esi_ea",
    "bond_10y", "eur_usd",
]

SAMPLE_START = "2002-01-01"  # bound by trade volume indices


def _load(slug: str) -> pd.Series:
    df = pd.read_csv(RAW / f"{slug}.csv", parse_dates=["date"])
    return df.set_index("date")["value"].sort_index()


def _transform(s: pd.Series, code: str) -> pd.Series:
    if code == "dlog":
        return 100.0 * np.log(s).diff()
    if code == "dlog_real":  # nominal MoM growth minus HICP inflation -> real growth
        infl = 100.0 * np.log(_load("hicp")).diff()
        g = 100.0 * np.log(s).diff()
        return (g - infl.reindex(g.index)).dropna()
    if code == "diff":
        return s.diff()
    if code == "level":
        return s
    raise ValueError(f"unknown transform {code}")


def build() -> tuple[pd.DataFrame, pd.Series]:
    # --- monthly indicators ---
    cols = {}
    for slug in MONTHLY_ORDER:
        s = _transform(_load(slug), TRANSFORMS[slug])
        s.index = s.index + pd.offsets.MonthEnd(0)  # normalise to month-end
        cols[slug] = s
    monthly = pd.DataFrame(cols)
    monthly = monthly.loc[monthly.index >= SAMPLE_START]
    # Full monthly calendar so ragged edges at the end become explicit NaNs.
    full_idx = pd.date_range(monthly.index.min(), monthly.index.max(), freq="ME")
    monthly = monthly.reindex(full_idx)
    monthly.index.name = "date"

    # --- quarterly GDP target, placed on quarter-end months ---
    gdp = _load("gdp_qoq")
    gdp.index = gdp.index + pd.offsets.MonthEnd(0)
    gdp = gdp.loc[gdp.index >= SAMPLE_START]
    gdp.name = "gdp_qoq"
    gdp.index.name = "date"

    return monthly, gdp


def main() -> None:
    monthly, gdp = build()
    monthly.to_csv(PROC / "monthly_panel.csv")
    gdp.to_frame().to_csv(PROC / "gdp_quarterly.csv")
    (PROC / "transforms.json").write_text(json.dumps(TRANSFORMS, indent=2))

    print(f"monthly_panel: {monthly.shape[0]} months x {monthly.shape[1]} series "
          f"({monthly.index.min().date()} .. {monthly.index.max().date()})")
    tail_missing = monthly.tail(3).isna().sum()
    print("ragged edge (NaNs in last 3 months):")
    print(tail_missing[tail_missing > 0].to_string() or "  none")
    print(f"gdp_quarterly: {gdp.notna().sum()} quarters, "
          f"last = {gdp.dropna().index[-1].date()} ({gdp.dropna().iloc[-1]:.2f}%)")


if __name__ == "__main__":
    main()
