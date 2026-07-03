"""
Decompose the *nowcast itself* (GDP unobserved, vintage ragged edge) into
    mu (sample-mean anchor) + common-factor signal + GDP idiosyncratic carry-over
for 2024Q1-2026Q1. Fixed full-sample parameters, smoothed states per vintage.

Run:  python3 src/anchor_decomposition_vintage.py
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

W = np.array([1.0, 2.0, 3.0, 2.0, 1.0])


def main():
    m, g = base.load_processed()
    res = base.build_model(m, g).fit(disp=0, maxiter=200)
    lam = res.params["loading.0->gdp_qoq"]
    gd = g.dropna()
    mu, sd = gd.mean(), gd.std()

    print(f"anchor mu = {mu:+.3f}, sd = {sd:.3f}, lambda = {lam:+.4f}\n")
    print(f"{'quarter':<8}{'nowcast':>9}{'mu':>7}{'factor':>9}{'gdp idio':>10}{'flash':>8}")
    for q in pd.period_range("2024Q1", "2026Q1", freq="Q"):
        mv, gv = base.make_vintage(m, g, q)
        r = base.build_model(mv, gv).smooth(res.params)
        st = r.states.smoothed
        fcol = [c for c in st.columns if "eps" not in str(c)][0]
        ecol = [c for c in st.columns if "eps" in str(c) and "gdp" in str(c)][0]
        t = q.asfreq("M", how="end")
        idx = [t - i for i in range(4, -1, -1)]
        wf = float(W @ st.loc[idx, fcol].values)   # idx ascending, W symmetric
        we = float(W @ st.loc[idx, ecol].values)
        fac_c, idio_c = sd * lam * wf, sd * we
        print(f"{str(q):<8}{mu + fac_c + idio_c:>+9.2f}{mu:>+7.2f}{fac_c:>+9.2f}"
              f"{idio_c:>+10.2f}{'':>8}")


if __name__ == "__main__":
    main()
