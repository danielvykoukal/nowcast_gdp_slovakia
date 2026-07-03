# Data Catalogue — Slovak GDP Nowcasting MFDFM

Public data panel for the Mixed-Frequency Dynamic Factor Model. Every series is
downloaded by [`src/fetch_data.py`](src/fetch_data.py) into `data/raw/<slug>.csv`.

**Scope decision:** public sources only. Series named in the research document that
are *not* publicly downloadable are listed in the [Excluded](#excluded-non-public-series)
section with the reason and the public substitute used instead.

Sources:
- **Eurostat** — via the `eurostat` Python package (SDMX dissemination API).
- **FRED** — St. Louis Fed, no-key CSV endpoint (`fredgraph.csv?id=...`).

All Eurostat activity series requested are **seasonally & calendar adjusted (SCA/SA)** and
expressed as **volume / real** measures (index 2021=100 or volume indices), so no separate
deflation step is needed — the research doc's deflation requirement is satisfied at source.

---

## Target variable

| Slug | Variable | GDP approach | Source | Dataset / filter | Freq | Pub. lag | Unit | Transform |
|---|---|---|---|---|---|---|---|---|
| `gdp_qoq` | Real GDP, chain-linked, SCA | — (target) | Eurostat | [`namq_10_gdp`](https://ec.europa.eu/eurostat/databrowser/view/namq_10_gdp/default/table) · `B1GQ`/`CLV_PCH_PRE`/`SCA`/`SK` | Q | ~45–70 d | QoQ % | none (already QoQ growth) |

Cross-check series (not modelled): FRED [`CLVMNACSCAB1GQSK`](https://fred.stlouisfed.org/series/CLVMNACSCAB1GQSK).

## Monthly hard indicators (production / expenditure / trade / labour)

| Slug | Variable | GDP approach / sector | Source | Dataset / filter | Freq | Pub. lag | Unit | Transform |
|---|---|---|---|---|---|---|---|---|
| `ip_total` | Industrial production, total industry (B–D) | Production — industry | Eurostat | [`sts_inpr_m`](https://ec.europa.eu/eurostat/databrowser/view/sts_inpr_m/default/table) · `B-D`/`SCA`/`I21` | M | ~40 d | Index 2021=100 | Δlog ×100 (MoM %) |
| `ip_manuf` | Industrial production, manufacturing (C) | Production — manufacturing/autos proxy | Eurostat | `sts_inpr_m` · `C`/`SCA`/`I21` | M | ~40 d | Index 2021=100 | Δlog ×100 |
| `retail_vol` | Retail trade sales volume (G47) | Expenditure — household consumption | Eurostat | [`sts_trtu_m`](https://ec.europa.eu/eurostat/databrowser/view/sts_trtu_m/default/table) · `G47`/`VOL_SLS`/`SCA`/`I21` | M | ~40 d | Index 2021=100 | Δlog ×100 |
| `construction` | Construction production (F) | Expenditure — GFCF / investment | Eurostat | [`sts_copr_m`](https://ec.europa.eu/eurostat/databrowser/view/sts_copr_m/default/table) · `F`/`SCA`/`I21` | M | ~45 d | Index 2021=100 | Δlog ×100 |
| `exports_vol` | Exports of goods, volume, world | Expenditure — net exports | Eurostat | [`ei_eteu27_2020_m`](https://ec.europa.eu/eurostat/databrowser/view/ei_eteu27_2020_m/default/table) · `EXP`/`ET-T`/`WORLD`/`IVOL-SA` | M | ~40 d | Volume index | Δlog ×100 |
| `imports_vol` | Imports of goods, volume, world | Expenditure — net exports / input signal | Eurostat | `ei_eteu27_2020_m` · `IMP`/`ET-T`/`WORLD`/`IVOL-SA` | M | ~40 d | Volume index | Δlog ×100 |
| `unemp_rate` | Unemployment rate, SA | Income — labour market | Eurostat | [`une_rt_m`](https://ec.europa.eu/eurostat/databrowser/view/une_rt_m/default/table) · `TOTAL`/`T`/`PC_ACT`/`SA` | M | ~30 d | % of labour force | first difference (pp) |

## Monthly soft / sentiment indicators (no revisions, released month-end)

| Slug | Variable | Sector | Source | Dataset / filter | Freq | Pub. lag | Unit | Transform |
|---|---|---|---|---|---|---|---|---|
| `esi_sk` | Economic Sentiment Indicator, Slovakia | Sentiment (composite) | Eurostat | [`ei_bssi_m_r2`](https://ec.europa.eu/eurostat/databrowser/view/ei_bssi_m_r2/default/table) · `BS-ESI-I`/`SK`/`SA` | M | ~0 d | Index | level |
| `ind_conf_sk` | Industrial confidence balance, Slovakia | Sentiment — industry | Eurostat | `ei_bssi_m_r2` · `BS-ICI-BAL`/`SK`/`SA` | M | ~0 d | Balance | level |
| `cons_conf_sk` | Consumer confidence balance, Slovakia | Sentiment — households | Eurostat | `ei_bssi_m_r2` · `BS-CCI-BAL`/`SK`/`SA` | M | ~0 d | Balance | level |
| `esi_de` | Economic Sentiment Indicator, Germany | External demand proxy | Eurostat | `ei_bssi_m_r2` · `BS-ESI-I`/`DE`/`SA` | M | ~0 d | Index | level |
| `esi_ea` | Economic Sentiment Indicator, euro area | External demand proxy | Eurostat | `ei_bssi_m_r2` · `BS-ESI-I`/`EA20`/`SA` | M | ~1 mo | Index | level |

## External real-activity indicators (Germany / euro area)

Slovakia is a supplier to the German/euro-area industrial chain, so foreign production leads
domestic activity by 1–3 months. Added in the enhanced model ([`src/model_v2.py`](src/model_v2.py)).

| Slug | Variable | Sector | Source | Dataset / filter | Freq | Pub. lag | Unit | Transform |
|---|---|---|---|---|---|---|---|---|
| `ip_de` | Industrial production, Germany, total (B–D) | External activity | Eurostat | `sts_inpr_m` · `DE`/`B-D`/`SCA`/`I21` | M | ~40 d | Index 2021=100 | Δlog ×100 |
| `ip_de_auto` | Motor-vehicle production, Germany (C29) | External — auto supply chain | Eurostat | `sts_inpr_m` · `DE`/`C29`/`SCA`/`I21` | M | ~40 d | Index 2021=100 | Δlog ×100 |
| `ip_ea` | Industrial production, euro area, total (B–D) | External activity | Eurostat | `sts_inpr_m` · `EA20`/`B-D`/`SCA`/`I21` | M | ~45 d | Index 2021=100 | Δlog ×100 |

## Domestic-demand indicators

Added to counterweight the industrial/foreign block (see [`BIAS.md`](BIAS.md)). Note: in a
single common factor these do not by themselves remove the 2025 level bias, but they broaden the
domestic-demand signal in the panel.

| Slug | Variable | Sector | Source | Dataset / filter | Freq | Pub. lag | Unit | Transform |
|---|---|---|---|---|---|---|---|---|
| `services_iaf` | Accommodation & food services turnover (NACE I) | Domestic — discretionary consumption | Eurostat | `sts_setu_m` · `I`/`SCA`/`I21`/`NETTUR` | M | ~60 d | Index 2021=100 (nominal) | Δlog ×100 |
| `hicp` | Harmonised index of consumer prices, all-items | Domestic — real-income control | Eurostat | `prc_hicp_midx` · `CP00`/`I15` | M | ~17 d | Index 2015=100 | Δlog ×100 (monthly inflation) |

## Financial indicators (monthly)

| Slug | Variable | Sector | Source | Dataset / filter | Freq | Pub. lag | Unit | Transform |
|---|---|---|---|---|---|---|---|---|
| `bond_10y` | Slovak 10Y government bond yield | Financial | Eurostat | [`irt_lt_mcby_m`](https://ec.europa.eu/eurostat/databrowser/view/irt_lt_mcby_m/default/table) · `SK` | M | ~10 d | % p.a. | first difference |
| `eur_usd` | EUR/USD exchange rate (monthly mean of daily) | Financial | FRED | [`DEXUSEU`](https://fred.stlouisfed.org/series/DEXUSEU) | D→M | ~0 d | USD per EUR | Δlog ×100 |

---

## Excluded (non-public) series

These are described in the research document but are **not publicly downloadable**, so they are
omitted. The nearest public substitute already in the panel is noted.

| Series (doc §) | Why excluded | Public substitute used |
|---|---|---|
| SIPS interbank payment volumes (§7.2) | NBS-internal settlement data, not published | Financial block (`bond_10y`, `eur_usd`) + retail volume |
| RRZ real-time-vintage database (§3.2) | Restricted; used for true real-time backtests | Latest vintage + **pseudo-real-time** replay via publication lags (see `src/model.py`) |
| Employer health/social insurance contributions (§5.2) | Administrative micro-data, not public | `unemp_rate` as the labour-market indicator |
| German Ifo / ZEW indices (§6.2) | Licensed, not on Eurostat | `esi_de` (Eurostat ESI for Germany) as external-demand proxy |
| Motor-vehicle IPP breakdown, NACE C29 (§4.1) | Not published for SK at monthly SCA in `sts_inpr_m` | `ip_manuf` (total manufacturing C) |
| SAX / DAX equity indices (§7.1) | No stable no-key public endpoint | `bond_10y`, `eur_usd`, `esi_de` cover the financial/external block |
| Electronic truck-toll (mýto) volumes | Operator data, no open API | `ip_total`, `exports_vol` (real-activity block) |
| Electricity consumption / load | ENTSO-E API needs a token | External IP block (`ip_de`, `ip_ea`) |
| VAT receipts | SK Financial Administration, no open API | `retail_vol` (consumption) |
| New car registrations (ACEA, monthly) | Licensed; no clean monthly API for SK | `ip_de_auto` (German auto output) |

These are high-value real-time proxies (the "administrative data" edge from the research doc) and
are natural next additions if credentialed/manual access is arranged.

## Notes

- Panel start is effectively **2002-01** (limited by the trade volume indices); GDP target runs
  1995Q1–latest. The model uses the common overlapping sample.
- Publication lags in the table drive the ragged-edge / pseudo-real-time reconstruction in Part 2.
- Re-download everything with `python src/fetch_data.py --refresh`.
