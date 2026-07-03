"""
Fetch the new candidate series requested for the panel expansion:

  * real_wage_bill  — INCOME-SIDE signal (the biggest gap in the panel). Built from the SU SR
    DATAcube API: average monthly nominal wage in EUR by sector (od0008ms) x employment YoY
    index by sector (od0007ms), across the 5 published sectors (industry, construction, trade,
    hotels, transport), deflated by HICP -> real household-income (wage-bill) YoY growth. This is
    exactly the domestic-demand signal that would have caught the 2025 decoupling foreign IP hid.
  * services_H / services_J / services_N — broaden services turnover beyond accommodation/food:
    transport (H), ICT (J), administrative (N) from the same Eurostat sts_setu_m as services_iaf.
    (Professional services M is not published SCA for SK; N is the available substitute.)

Not fetched (blocked without credentials / a manual file — reported, not hidden):
  * VAT receipts        — SK Financial Administration publishes XLS only, no API (manual scrape).
  * New car registrations — MoI has no clean public API.
  * Electricity          — ENTSO-E requires a free API token that cannot be registered headless.
  * German truck-toll    — Destatis serves it via GENESIS (token) only; open CSV URLs 404.

Run:  python src/fetch_new_data.py
"""
from __future__ import annotations
import json
import ssl
import urllib.request
from pathlib import Path

import certifi
import eurostat
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
_SSL = ssl.create_default_context(cafile=certifi.where())

# SU SR sector -> NACE code in od0007ms/od0008ms
SECTORS = {"industry": "NACE01", "construction": "NACE06", "trade": "NACE09",
           "hotels": "NACE10", "transport": "NACE12"}


def _get(url: str, tmo: int = 120) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, context=_SSL, timeout=tmo).read().decode("utf-8", "replace")


def _susr_pull(cube, specu, nace, unit, indic) -> pd.Series:
    """One SU SR DATAcube series (all years/months) for a fixed sector/unit/indicator."""
    url = f"https://data.statistics.sk/api/v2/dataset/{cube}/all/all/{specu}/{nace}/{unit}/{indic}?lang=en"
    js = json.loads(_get(url))
    val, dim = js["value"], js["dimension"]
    years = list(dim[f"{cube}_year"]["category"]["index"].keys())
    months = list(dim[f"{cube}_month"]["category"]["index"].keys())
    nm = len(months)
    out = {}
    for a, y in enumerate(years):
        for b, m in enumerate(months):
            md = "".join(ch for ch in m if ch.isdigit())
            if not md or not (1 <= int(md) <= 12):
                continue
            p = a * nm + b
            if p < len(val) and val[p] is not None:
                out[pd.Timestamp(f"{y}-{int(md):02d}-01") + pd.offsets.MonthEnd(0)] = val[p]
    return pd.Series(out).sort_index()


def build_wage_bill() -> pd.Series:
    wages = {s: _susr_pull("od0008ms", "SPECU_ABVAL", n, "UNIT_EUR", "U_NP_0001")
             for s, n in SECTORS.items()}
    emps = {s: _susr_pull("od0007ms", "SPECU_Y_ROMR", n, "UNIT_INDEX", "U_PR_0007")
            for s, n in SECTORS.items()}
    W, E = pd.DataFrame(wages), pd.DataFrame(emps)
    wage_yoy = 100 * np.log(W).diff(12)          # nominal wage YoY % by sector
    emp_yoy = E - 100                             # employment YoY % by sector (index base 100)
    hicp = pd.read_csv(RAW / "hicp.csv", parse_dates=["date"]).set_index("date")["value"]
    hicp_yoy = (100 * np.log(hicp).diff(12)).reindex(wage_yoy.index)
    # real wage-bill YoY growth = mean_sector(nominal wage YoY + employment YoY) - HICP YoY
    bill = (wage_yoy.mean(axis=1) + emp_yoy.mean(axis=1)).sub(hicp_yoy, axis=0).dropna()
    bill.index.name = "date"
    return bill


# Eurostat services turnover (same query as services_iaf, different NACE)
SERVICES = {"services_H": "H", "services_J": "J", "services_N": "N"}


def fetch_service(nace: str) -> pd.Series:
    df = eurostat.get_data_df("sts_setu_m", filter_pars={
        "geo": ["SK"], "freq": ["M"], "nace_r2": [nace], "s_adj": ["SCA"],
        "unit": ["I21"], "indic_bt": ["NETTUR"]})
    tc = [c for c in df.columns if isinstance(c, str) and len(c) >= 4 and c[:4].isdigit()]
    r = df.iloc[0]
    s = pd.Series({c: r[c] for c in tc}, dtype="float64").dropna()
    s.index = pd.PeriodIndex(s.index.str.replace("-", ""), freq="M").to_timestamp(how="end").normalize()
    return s.sort_index()


def main():
    print("Building real_wage_bill from SU SR DATAcube (od0007ms x od0008ms / HICP)...")
    bill = build_wage_bill()
    bill.rename("value").to_frame().to_csv(RAW / "real_wage_bill.csv")
    print(f"  real_wage_bill: {len(bill)} months {bill.index.min().date()}..{bill.index.max().date()}"
          f"  (last {bill.iloc[-1]:+.2f}% YoY real)")

    for slug, nace in SERVICES.items():
        s = fetch_service(nace)
        s.rename("value").rename_axis("date").to_frame().to_csv(RAW / f"{slug}.csv")
        print(f"  {slug}: {len(s)} months {s.index.min().date()}..{s.index.max().date()}")
    print("Done.")


if __name__ == "__main__":
    main()
