"""
Fix 1 from the anchor investigation (see anchor_decomposition.py): a mixed-frequency
DFM whose GDP equation carries a LOCAL LEVEL - a random-walk trend state mu_t - so the
time-varying mean of GDP growth is part of the state vector and the Kalman filter can
learn regime shifts instead of anchoring on the fixed sample mean.

State (k = 10):   s_t = [f_t..f_{t-4}, mu_t..mu_{t-4}]'
Transition:       f_t  = phi1 f_{t-1} + phi2 f_{t-2} + w_t,  w_t ~ N(0,1)
                  mu_t = mu_{t-1} + eta_t,                   eta_t ~ N(0, sigma2_trend)
Measurement:      monthly indicator i:  y_it = lambda_i f_t + e_it
                  quarterly GDP:        y_gt = MM'(lambda_g f + mu) + e_gt
                  MM = Mariano-Murasawa weights [1/3, 2/3, 1, 2/3, 1/3]

mu enters only the GDP row (it is GDP's own trend, not the panel's); its scale is
identified by the fixed MM weights + sigma2_trend. Factor block initialized stationary,
trend block diffuse. Everything else follows src/dfm_statespace.py.

Run:  python3 src/dfm_locallevel.py   (full-sample fit, trend path, current nowcast)
"""
from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.initialization import Initialization

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
PROC = BASE / "data" / "processed"
OUT = BASE / "outputs"

MM_WEIGHTS = np.array([1 / 3, 2 / 3, 1.0, 2 / 3, 1 / 3])
K = 5  # lags per block


class MMDynamicFactorLL(sm.tsa.statespace.MLEModel):
    """Single-factor MF-DFM + random-walk local level on the GDP equation."""

    def __init__(self, endog: pd.DataFrame, n_monthly: int):
        init = Initialization(2 * K)
        init.set((0, K), "stationary")
        init.set((K, 2 * K), "approximate_diffuse")
        super().__init__(endog, k_states=2 * K, k_posdef=2, initialization=init)
        self.n_monthly = n_monthly
        self._var0 = 2 + self.k_endog          # idio variances start here
        # lag shifts for both blocks
        for i in range(1, K):
            self["transition", i, i - 1] = 1.0
            self["transition", K + i, K + i - 1] = 1.0
        self["transition", K, K] = 1.0          # mu random walk
        self["selection", 0, 0] = 1.0
        self["selection", K, 1] = 1.0
        self["state_cov", 0, 0] = 1.0           # factor shock var fixed (identification)

    @property
    def param_names(self):
        n = self.endog_names
        return (["phi1", "phi2"] + [f"loading.{x}" for x in n]
                + [f"sigma2.{x}" for x in n] + ["sigma2.trend"])

    @property
    def start_params(self):
        import dfm_statespace as ss
        base_start = ss._start_params(self)      # phi(2), loadings, idio vars
        return np.append(base_start, 1e-4)       # small trend-innovation variance

    def transform_params(self, unconstrained):
        c = unconstrained.copy()
        c[self._var0:] = unconstrained[self._var0:] ** 2
        return c

    def untransform_params(self, constrained):
        u = constrained.copy()
        u[self._var0:] = np.sqrt(np.maximum(constrained[self._var0:], 1e-12))
        return u

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)
        self["transition", 0, 0] = params[0]
        self["transition", 0, 1] = params[1]
        load = params[2:2 + self.k_endog]
        Z = np.zeros((self.k_endog, 2 * K))
        Z[:self.n_monthly, 0] = load[:self.n_monthly]
        Z[self.n_monthly, :K] = load[self.n_monthly] * MM_WEIGHTS
        Z[self.n_monthly, K:] = MM_WEIGHTS       # trend enters GDP row, fixed weights
        self["design"] = Z
        self["obs_cov"] = np.diag(params[self._var0:-1])
        self["state_cov", 1, 1] = params[-1]


# ---------------------------------------------------------------------------
def make_endog(m: pd.DataFrame, g: pd.Series, mom) -> pd.DataFrame:
    ms = (m - mom["m_mean"]) / mom["m_std"]
    gs = (g - mom["g_mean"]) / mom["g_std"]
    endog = ms.copy()
    endog["gdp"] = gs.reindex(endog.index)
    endog.index = pd.DatetimeIndex(endog.index, freq="ME")
    return endog


def moments(m: pd.DataFrame, g: pd.Series):
    return {"m_mean": m.mean(), "m_std": m.std(),
            "g_mean": float(g.mean()), "g_std": float(g.std())}


def gdp_signal(res, month: pd.Timestamp, mom) -> float:
    Z = np.asarray(res.model["design"])
    row = Z[res.model.n_monthly, :]
    idx = res.model._index.get_loc(month)
    return mom["g_mean"] + mom["g_std"] * float(row @ res.smoothed_state[:, idx])


def trend_path(res, mom) -> pd.Series:
    """Implied local mean of QoQ GDP growth: MM-aggregated trend, de-standardised."""
    mu = res.smoothed_state[K:2 * K, :]          # mu_t .. mu_{t-4}
    agg = MM_WEIGHTS @ mu
    return pd.Series(mom["g_mean"] + mom["g_std"] * agg, index=res.model._index)


def fit_ll(m: pd.DataFrame, g: pd.Series, start_params=None, maxiter=300):
    mom = moments(m, g)
    mod = MMDynamicFactorLL(make_endog(m, g, mom), n_monthly=m.shape[1])
    res = mod.fit(start_params=start_params, disp=False, maxiter=maxiter)
    return res, mom


# ---------------------------------------------------------------------------
def main():
    m = pd.read_csv(PROC / "monthly_panel.csv", parse_dates=["date"]).set_index("date")
    g = pd.read_csv(PROC / "gdp_quarterly.csv", parse_dates=["date"]).set_index("date")["gdp_qoq"]

    res, mom = fit_ll(m, g)
    print(f"log-likelihood = {res.llf:.1f}")
    print(f"sigma2.trend = {res.params[-1]:.6f} "
          f"(monthly trend-innovation std, standardized units: {np.sqrt(res.params[-1]):.4f})")

    tp = trend_path(res, mom)
    print("\nImplied local mean of QoQ GDP growth (smoothed trend):")
    for y in ["2005-12-31", "2010-12-31", "2015-12-31", "2019-12-31",
              "2021-12-31", "2023-12-31", "2025-12-31"]:
        t = pd.Timestamp(y)
        if t in tp.index:
            print(f"  {t.date()}: {tp.loc[t]:+.3f}")
    tp.rename("trend_qoq").to_frame().to_csv(OUT / "ll_trend_path.csv")

    target = m.index.max() + pd.offsets.MonthEnd(0)
    nc = gdp_signal(res, target, mom)
    print(f"\nNOWCAST {pd.Period(target, 'Q')} (local-level DFM): {nc:+.2f}% QoQ")
    print("Wrote outputs/ll_trend_path.csv")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
