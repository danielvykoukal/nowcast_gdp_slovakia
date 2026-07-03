"""
Real-time (vintage) data for Slovakia from the OECD "Short-term economic statistics
revisions" database (DSD_STES_REVISIONS). Each EDITION is a monthly snapshot of the data
as it was published in that month, so we can reconstruct exactly what a forecaster saw at
any past date - including the GDP *flash* (first release) and first-release input values.

Fetches, for Slovakia, the revisable series that map onto our panel plus GDP:
  ip_total   <- PRVM / industry (BTE)     retail_vol <- TOVM / retail (G47)
  ip_manuf   <- PRVM / manufacturing (C)  unemp_rate <- UNEMP
  gdp_level  <- B1GQ_Q (GDP volume, for flash & final QoQ growth)

Each series is cached to data/raw/oecd_vintages/<name>.csv as long rows
[edition, time_period, value].

Run:  python src/realtime_data.py [--refresh]
"""
from __future__ import annotations
import io
import ssl
import sys
import time
from pathlib import Path

import certifi
import pandas as pd
import urllib.request

BASE = Path(__file__).resolve().parent.parent
VDIR = BASE / "data" / "raw" / "oecd_vintages"
VDIR.mkdir(parents=True, exist_ok=True)
_SSL = ssl.create_default_context(cafile=certifi.where())

BASE_URL = ("https://sdmx.oecd.org/public/rest/data/"
            "OECD.SDD.STES,DSD_STES_REVISIONS@DF_STES_REVISIONS,/")

# name -> (FREQ, MEASURE, UNIT_MEASURE, ACTIVITY)
SERIES_OECD = {
    "ip_total":   ("M", "PRVM", "IX", "BTE"),
    "ip_manuf":   ("M", "PRVM", "IX", "C"),
    "retail_vol": ("M", "TOVM", "IX", "G47"),
    "unemp_rate": ("M", "UNEMP", "PT_LF", "_T"),
    "gdp_level":  ("Q", "B1GQ_Q", "XDC", "_T"),
}


def _get(url: str, tmo: int = 150) -> str:
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/vnd.sdmx.data+csv; charset=utf-8"})
    return urllib.request.urlopen(req, context=_SSL, timeout=tmo).read().decode("utf-8", "replace")


def fetch_one(name: str) -> pd.DataFrame:
    freq, meas, unit, act = SERIES_OECD[name]
    url = BASE_URL + f"SVK.{freq}.{meas}.{unit}.{act}.?format=csvfilewithlabels"
    df = pd.read_csv(io.StringIO(_get(url)))
    out = df[["EDITION", "TIME_PERIOD", "OBS_VALUE"]].rename(
        columns={"EDITION": "edition", "TIME_PERIOD": "time_period", "OBS_VALUE": "value"})
    return out.sort_values(["edition", "time_period"]).reset_index(drop=True)


def load_vintages(name: str) -> pd.DataFrame:
    return pd.read_csv(VDIR / f"{name}.csv")


def main(refresh: bool = False) -> None:
    print(f"Fetching OECD real-time vintages for Slovakia -> {VDIR}\n")
    for name in SERIES_OECD:
        path = VDIR / f"{name}.csv"
        if path.exists() and not refresh:
            n = sum(1 for _ in path.open()) - 1
            print(f"  skip  {name:11s} (exists, {n} rows)")
            continue
        t = time.time()
        df = fetch_one(name)
        df.to_csv(path, index=False)
        print(f"  OK    {name:11s} {len(df):6d} rows, {df['edition'].nunique()} editions "
              f"({df['edition'].min()}..{df['edition'].max()}) in {time.time()-t:.0f}s")
    print("\nDone.")


if __name__ == "__main__":
    main(refresh="--refresh" in sys.argv)
