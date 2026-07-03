"""
Part 1 - Data collection for the Slovak GDP nowcasting MFDFM.

Downloads the PUBLIC subset of the panel described in
`Slovakia GDP Nowcasting Data Research.md` and writes one tidy CSV per series
to data/raw/<slug>.csv with columns [date, value].

Sources: Eurostat (via the `eurostat` package) and FRED (no-key CSV endpoint).
Non-public series named in the research doc (SIPS interbank payments, RRZ
real-time-vintage database, employer insurance-contribution administrative data)
are intentionally excluded - see DATA_CATALOGUE.md.

Run:  python src/fetch_data.py [--refresh]
"""
from __future__ import annotations
import io
import ssl
import sys
import urllib.request
from pathlib import Path

import certifi
import eurostat
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

# --- SSL context so FRED CSV works on macOS (system certs are often missing) ---
_SSL_CTX = ssl.create_default_context(cafile=certifi.where())


# ---------------------------------------------------------------------------
# Series registry.  Each entry: slug -> spec dict.
#   source: "eurostat" | "fred"
#   For eurostat: code + filter_pars (single-country query -> one data row)
#   For fred:     fred_id
#   freq: "Q" or "M" (for downstream use / metadata)
# ---------------------------------------------------------------------------
SERIES: dict[str, dict] = {
    # ---- TARGET: quarterly real GDP, chain-linked, SCA, QoQ % growth ----
    "gdp_qoq": dict(
        source="eurostat", freq="Q", code="namq_10_gdp",
        filter_pars={"geo": ["SK"], "freq": ["Q"], "unit": ["CLV_PCH_PRE"],
                     "s_adj": ["SCA"], "na_item": ["B1GQ"]},
    ),
    # ---- Monthly HARD indicators ----
    "ip_total": dict(  # production side - total industry (B-D)
        source="eurostat", freq="M", code="sts_inpr_m",
        filter_pars={"geo": ["SK"], "freq": ["M"], "nace_r2": ["B-D"],
                     "s_adj": ["SCA"], "unit": ["I21"]},
    ),
    "ip_manuf": dict(  # production side - manufacturing (C); autos not published for SK
        source="eurostat", freq="M", code="sts_inpr_m",
        filter_pars={"geo": ["SK"], "freq": ["M"], "nace_r2": ["C"],
                     "s_adj": ["SCA"], "unit": ["I21"]},
    ),
    "retail_vol": dict(  # household consumption - retail trade volume (G47)
        source="eurostat", freq="M", code="sts_trtu_m",
        filter_pars={"geo": ["SK"], "freq": ["M"], "nace_r2": ["G47"],
                     "s_adj": ["SCA"], "unit": ["I21"], "indic_bt": ["VOL_SLS"]},
    ),
    "construction": dict(  # investment / GFCF proxy - construction production (F)
        source="eurostat", freq="M", code="sts_copr_m",
        filter_pars={"geo": ["SK"], "freq": ["M"], "nace_r2": ["F"],
                     "s_adj": ["SCA"], "unit": ["I21"]},
    ),
    "exports_vol": dict(  # external demand - exports volume index, world, SA
        source="eurostat", freq="M", code="ei_eteu27_2020_m",
        filter_pars={"geo": ["SK"], "freq": ["M"], "stk_flow": ["EXP"],
                     "indic": ["ET-T"], "partner": ["WORLD"], "unit": ["IVOL-SA"]},
    ),
    "imports_vol": dict(  # external side / intermediate-input signal
        source="eurostat", freq="M", code="ei_eteu27_2020_m",
        filter_pars={"geo": ["SK"], "freq": ["M"], "stk_flow": ["IMP"],
                     "indic": ["ET-T"], "partner": ["WORLD"], "unit": ["IVOL-SA"]},
    ),
    "unemp_rate": dict(  # labour / income side - unemployment rate, SA
        source="eurostat", freq="M", code="une_rt_m",
        filter_pars={"geo": ["SK"], "freq": ["M"], "s_adj": ["SA"],
                     "age": ["TOTAL"], "sex": ["T"], "unit": ["PC_ACT"]},
    ),
    # ---- Monthly SOFT / sentiment (no revisions, released month-end) ----
    "esi_sk": dict(
        source="eurostat", freq="M", code="ei_bssi_m_r2",
        filter_pars={"geo": ["SK"], "freq": ["M"], "indic": ["BS-ESI-I"], "s_adj": ["SA"]},
    ),
    "ind_conf_sk": dict(
        source="eurostat", freq="M", code="ei_bssi_m_r2",
        filter_pars={"geo": ["SK"], "freq": ["M"], "indic": ["BS-ICI-BAL"], "s_adj": ["SA"]},
    ),
    "cons_conf_sk": dict(
        source="eurostat", freq="M", code="ei_bssi_m_r2",
        filter_pars={"geo": ["SK"], "freq": ["M"], "indic": ["BS-CCI-BAL"], "s_adj": ["SA"]},
    ),
    "esi_de": dict(  # external-demand proxy (public substitute for Ifo/ZEW)
        source="eurostat", freq="M", code="ei_bssi_m_r2",
        filter_pars={"geo": ["DE"], "freq": ["M"], "indic": ["BS-ESI-I"], "s_adj": ["SA"]},
    ),
    "esi_ea": dict(  # euro-area sentiment (external demand)
        source="eurostat", freq="M", code="ei_bssi_m_r2",
        filter_pars={"geo": ["EA20"], "freq": ["M"], "indic": ["BS-ESI-I"], "s_adj": ["SA"]},
    ),
    # ---- External real activity (Germany / euro area) - SK is a supplier economy ----
    "ip_de": dict(  # German industrial production, total
        source="eurostat", freq="M", code="sts_inpr_m",
        filter_pars={"geo": ["DE"], "freq": ["M"], "nace_r2": ["B-D"],
                     "s_adj": ["SCA"], "unit": ["I21"]},
    ),
    "ip_de_auto": dict(  # German motor-vehicle production (C29) - SK auto supply-chain proxy
        source="eurostat", freq="M", code="sts_inpr_m",
        filter_pars={"geo": ["DE"], "freq": ["M"], "nace_r2": ["C29"],
                     "s_adj": ["SCA"], "unit": ["I21"]},
    ),
    "ip_ea": dict(  # euro-area industrial production, total
        source="eurostat", freq="M", code="sts_inpr_m",
        filter_pars={"geo": ["EA20"], "freq": ["M"], "nace_r2": ["B-D"],
                     "s_adj": ["SCA"], "unit": ["I21"]},
    ),
    # ---- Domestic demand (counterweight to the industrial / foreign block) ----
    "services_iaf": dict(  # accommodation & food services turnover - discretionary consumption
        source="eurostat", freq="M", code="sts_setu_m",
        filter_pars={"geo": ["SK"], "freq": ["M"], "nace_r2": ["I"],
                     "s_adj": ["SCA"], "unit": ["I21"], "indic_bt": ["NETTUR"]},
    ),
    "hicp": dict(  # harmonised consumer prices - real-income squeeze on demand
        # prc_hicp_midx (ECOICOP v1) was frozen at 2025-12; prc_hicp_minr is the successor
        source="eurostat", freq="M", code="prc_hicp_minr",
        filter_pars={"geo": ["SK"], "freq": ["M"], "unit": ["I15"], "coicop18": ["TOTAL"]},
    ),
    # ---- Financial (monthly) ----
    "bond_10y": dict(  # SK 10Y government bond yield
        source="eurostat", freq="M", code="irt_lt_mcby_m",
        filter_pars={"geo": ["SK"], "freq": ["M"]},
    ),
    "eur_usd": dict(  # EUR/USD (daily on FRED -> aggregated to monthly here)
        source="fred", freq="M", fred_id="DEXUSEU",
    ),
}


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------
def _eurostat_to_series(df: pd.DataFrame) -> pd.Series:
    """Reshape a single-row eurostat wide frame into a (date-indexed) Series."""
    time_cols = [c for c in df.columns if isinstance(c, str) and len(c) >= 4 and c[:4].isdigit()]
    if df.shape[0] != 1:
        raise ValueError(f"expected exactly 1 data row, got {df.shape[0]}")
    row = df.iloc[0]
    s = pd.Series({c: row[c] for c in time_cols}, dtype="float64").dropna()
    # Index like '2024-Q1' or '2024-01' -> period end timestamp
    idx = pd.PeriodIndex(s.index.str.replace("-", ""), freq="Q" if "Q" in s.index[0] else "M")
    s.index = idx.to_timestamp(how="end").normalize()
    return s.sort_index()


def fetch_eurostat(spec: dict) -> pd.Series:
    df = eurostat.get_data_df(spec["code"], filter_pars=spec["filter_pars"])
    if df is None:
        raise ValueError("eurostat returned no data")
    return _eurostat_to_series(df)


def fetch_fred(spec: dict) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={spec['fred_id']}"
    with urllib.request.urlopen(url, context=_SSL_CTX, timeout=60) as r:
        raw = r.read().decode()
    d = pd.read_csv(io.StringIO(raw))
    d.columns = ["date", "value"]
    d["date"] = pd.to_datetime(d["date"])
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    s = d.dropna().set_index("date")["value"]
    if spec.get("freq") == "M":  # aggregate daily -> monthly mean, stamped at month end
        s = s.resample("ME").mean()
    return s.sort_index()


def fetch_one(slug: str, spec: dict) -> pd.Series:
    if spec["source"] == "eurostat":
        return fetch_eurostat(spec)
    if spec["source"] == "fred":
        return fetch_fred(spec)
    raise ValueError(f"unknown source {spec['source']}")


def main(refresh: bool = False) -> None:
    print(f"Fetching {len(SERIES)} public series -> {RAW}\n")
    ok, fail = 0, 0
    for slug, spec in SERIES.items():
        path = RAW / f"{slug}.csv"
        if path.exists() and not refresh:
            n = sum(1 for _ in path.open()) - 1
            print(f"  skip  {slug:14s} (exists, {n} rows) - use --refresh to redownload")
            ok += 1
            continue
        try:
            s = fetch_one(slug, spec)
            frame = s.rename("value").rename_axis("date").to_frame()
            frame.to_csv(path)
            frame.to_excel(RAW / f"{slug}.xlsx")
            print(f"  OK    {slug:14s} {len(s):4d} rows  {s.index.min().date()} .. {s.index.max().date()}")
            ok += 1
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"  FAIL  {slug:14s} {type(e).__name__}: {str(e)[:90]}")
            fail += 1
    print(f"\nDone: {ok} ok, {fail} failed.")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main(refresh="--refresh" in sys.argv)
