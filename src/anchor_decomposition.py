"""
Deep dig: why does the nowcast sit persistently above flash GDP (~+0.3pp) while GDP
stays low? Decompose every 2024Q1-2026Q2 nowcast into

    nowcast = sample-mean anchor (mu)  +  common-factor signal  +  GDP idiosyncratic AR(1)

using the smoothed states of the full-sample model, plus:
  * the "no data" nowcast (long-horizon forecast) -> the model's resting point,
  * indicator z-scores over 2024-2025 vs the GDP z-score -> the wedge that keeps
    the factor from reading the slowdown.

Run:  python3 src/anchor_decomposition.py
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
import model as base

W = np.array([1.0, 2.0, 3.0, 2.0, 1.0])  # Mariano-Murasawa quarterly aggregation weights


def main():
    m, g = base.load_processed()
    res = base.build_model(m, g).fit(disp=0, maxiter=200)
    p = res.params

    gd = g.dropna()
    mu, sd = gd.mean(), gd.std()
    print("=== GDP QoQ growth means (the anchor candidates) ===")
    for tag, s in [("full 2002+", gd), ("2013+", gd[gd.index >= "2013Q1"]),
                   ("2021+", gd[gd.index >= "2021Q1"]), ("2024+", gd[gd.index >= "2024Q1"])]:
        print(f"  {tag:<12} mean {s.mean():+.3f}  (n={len(s)})")
    print(f"  model standardizes GDP with full-sample mean {mu:+.3f}, std {sd:.3f}\n")

    print("=== key parameters ===")
    for k in p.index:
        if "gdp" in k or "factor" in k.lower():
            print(f"  {k:<28} {p[k]:+.4f}")
    lam = p["loading.0->gdp_qoq"]
    rho = p.get("L1.eps_Q.gdp_qoq", np.nan)
    print(f"  -> GDP idio AR(1) rho = {rho:+.3f}\n")

    # --- smoothed states ---
    st = res.states.smoothed
    fcol = [c for c in st.columns if "eps" not in str(c)][0]
    ecol = [c for c in st.columns if "eps" in str(c) and "gdp" in str(c)][0]
    f, eps = st[fcol], st[ecol]
    pred = res.predict()["gdp_qoq"]

    print("=== nowcast decomposition at quarter-end months ===")
    print(f"{'quarter':<8}{'pred':>7}{'mu':>7}{'factor':>9}{'gdp idio':>10}{'check':>8}{'actual':>8}")
    for q in pd.period_range("2024Q1", "2026Q2", freq="Q"):
        t = q.asfreq("M", how="end")
        if t not in pred.index:
            continue
        idx = [t - i for i in range(4, -1, -1)]
        wf = float(W[::-1] @ f.loc[idx].values[::-1])   # w0*f_t + w1*f_{t-1} + ...
        we = float(W[::-1] @ eps.loc[idx].values[::-1])
        fac_c, idio_c = sd * lam * wf, sd * we
        chk = mu + fac_c + idio_c
        act = g.get(q, np.nan)
        print(f"{str(q):<8}{pred.loc[t]:>+7.2f}{mu:>+7.2f}{fac_c:>+9.2f}{we*sd:>+10.2f}"
              f"{chk:>+8.2f}{act:>+8.2f}")
    print("  (check == pred confirms the decomposition; actual = latest-vintage GDP)\n")

    # --- resting point: forecast with no new data ---
    fc = res.forecast("2028-12")["gdp_qoq"].dropna()
    print("=== model's resting point (pure forecast, no data) ===")
    for t in ["2026-09", "2027-06", "2028-12"]:
        te = pd.Period(t, "M")
        if te in fc.index:
            print(f"  {te.asfreq('Q')}: {fc.loc[te]:+.3f}")
    print()

    # --- indicator z-scores 2024-2025 vs GDP z-score ---
    zm = (m - m.mean()) / m.std()
    zwin = zm.loc[(zm.index >= "2024-01") & (zm.index <= "2025-12")].mean().sort_values(ascending=False)
    zg = float((gd[(gd.index >= "2024Q1") & (gd.index <= "2025Q4")].mean() - mu) / sd)
    print("=== average z-score 2024-2025 (vs full-sample mean) ===")
    print(f"  GDP QoQ growth: z = {zg:+.2f}   <- what the factor SHOULD read")
    for k, v in zwin.items():
        print(f"  {k:<16} {v:+.2f}")


if __name__ == "__main__":
    main()
