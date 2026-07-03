"""Plot the winning config (drop_foreign, 2013+) vs the full model against actual GDP, 2022+."""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
OUT = BASE / "outputs"
import model as base, regime_experiment as R, dfm_paper as P

m, g = base.load_processed()
actual = g.dropna()
flash = P.build_flash()

configs = {
    "full  (all data, 2002+)": (R.COMBOS["full"][0], R.COMBOS["full"][1], None, "tab:blue"),
    "winner: drop foreign IP, 2013+": (R.COMBOS["drop_foreign"][0], R.COMBOS["drop_foreign"][1],
                                       pd.Period("2013Q1"), "tab:orange"),
}
ncs = {name: R.run_config(m, g, cf, bd, st) for name, (cf, bd, st, _) in configs.items()}

qs = [pd.Period(f"{y}Q{i}") for y in (2022, 2023, 2024, 2025) for i in (1, 2, 3, 4)] + [pd.Period("2026Q1")]
x = np.arange(len(qs))
fig, ax = plt.subplots(figsize=(13, 6))
ax.bar(x, [actual.get(q, np.nan) for q in qs], color="0.8", label="actual (final GDP)", width=0.6)
for name, (_, _, _, col) in configs.items():
    ax.plot(x, [ncs[name].get(q, np.nan) for q in qs], "o-", color=col, lw=1.8, ms=5, label=name)
fx = [i for i, q in enumerate(qs) if q in flash.index]
ax.plot(fx, [flash[qs[i]] for i in fx], "D", color="crimson", ms=6, label="flash (first release)")
ax.axvspan(15.5, len(qs) - 0.5, color="gold", alpha=0.12)
ax.text(17.5, ax.get_ylim()[1] * 0.95, "2025 (the biased window)", fontsize=9, color="0.4")
ax.set_xticks(x); ax.set_xticklabels([str(q) for q in qs], rotation=45, ha="right")
ax.set_ylabel("GDP QoQ %"); ax.axhline(0, color="0.5", lw=0.8)
ax.set_title("Fixing the 2025 bias: winner (drop foreign IP + estimate 2013+) tracks actual;\n"
             "the full 2002+ model floats ~1pp high in 2025")
ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
fig.savefig(OUT / "regime_winner_2025.png", dpi=130)
print("Wrote outputs/regime_winner_2025.png")
print("\n2025 nowcasts:")
for q in [pd.Period(f"2025Q{i}") for i in (1, 2, 3, 4)]:
    print(f"  {q}: full {ncs[list(configs)[0]].get(q):+.2f}  winner {ncs[list(configs)[1]].get(q):+.2f}"
          f"  actual {actual.get(q):+.2f}  flash {float(flash.get(q, np.nan)):+.2f}")
