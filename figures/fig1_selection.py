"""Fig. 1 of the paper: test accuracy of the checkpoint chosen on the test set (released rule)
and on a held-out validation set, for the same training runs (condition R1, three seeds).

Reads   results/diffmic/phase5_per_run.csv, results/diffmic/phase5_selection_inflation.csv,
        results/nnmobilenet/nnmb_per_run.csv, results/nnmobilenet/nnmb_selection_inflation.csv
Writes  figures/fig1_selection.png (600 dpi) and figures/fig1_selection.pdf

Sized for one IEEE column (3.5 in) with 8 pt Times labels. Run from anywhere:  python figures/fig1_selection.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
dm = pd.read_csv(ROOT / "results/diffmic/phase5_per_run.csv")
nn = pd.read_csv(ROOT / "results/nnmobilenet/nnmb_per_run.csv")
dmS = pd.read_csv(ROOT / "results/diffmic/phase5_selection_inflation.csv")
nnS = pd.read_csv(ROOT / "results/nnmobilenet/nnmb_selection_inflation.csv")

BLUE, ORANGE, INK, MUTED, GRID, GREY = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1", "#9a9994"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "Tinos", "TeX Gyre Termes", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def pairs(df, weights=None):
    """Test accuracy (%) of the test-selected and the validation-selected checkpoint of each R1 run."""
    d = df[df.run.str.startswith("R1")]
    if weights:
        d = d[d.weights == weights]
    a = d[d.rule.str.startswith("as_released")].set_index("run").acc * 100
    v = d[d.rule == "val_selected"].set_index("run").acc * 100
    return a, v.loc[a.index]


def num(x):
    """One decimal, with a true minus sign."""
    return f"{x:.1f}".replace("-", "−")


panels = [
    ("DiffMIC", *pairs(dm), dmS[dmS.metric == "acc"].iloc[0]),
    ("nnMobileNet\n(model weights)", *pairs(nn, "model"), nnS[(nnS.weights == "model") & (nnS.metric == "acc")].iloc[0]),
    ("nnMobileNet\n(EMA weights)", *pairs(nn, "ema"), nnS[(nnS.weights == "ema") & (nnS.metric == "acc")].iloc[0]),
]

fig, axs = plt.subplots(1, 3, figsize=(3.5, 2.12), sharey=True)
for ax, (name, a, v, s) in zip(axs, panels):
    for r in a.index:
        ax.plot([0, 1], [a[r], v[r]], color=GREY, lw=0.9, zorder=1, solid_capstyle="round")
    ax.scatter(np.zeros(len(a)), a, s=20, color=ORANGE, edgecolor="white", linewidth=0.7, zorder=3)
    ax.scatter(np.ones(len(v)), v, s=20, color=BLUE, edgecolor="white", linewidth=0.7, zorder=3)
    ax.set_xticks([0, 1], ["Test", "Validation"])
    ax.set_xlim(-0.5, 1.5)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    diff = f"+{num(s.inflation_mean * 100)} ({num(s.inflation_ci_lo * 100)} to {num(s.inflation_ci_hi * 100)})"
    ax.set_title(f"{name}\n{diff}", loc="left", color=INK, pad=4, linespacing=1.15)
axs[0].set_ylabel("Test accuracy (%)")
axs[0].set_ylim(80, 88)
axs[0].set_yticks([80, 82, 84, 86, 88])
for ax in axs[1:]:
    ax.tick_params(axis="y", length=0)
axs[1].set_xlabel("Set used to choose the checkpoint", labelpad=3)
fig.subplots_adjust(left=0.125, right=0.975, top=0.775, bottom=0.2, wspace=0.1)
fig.savefig(OUT / "fig1_selection.png", dpi=600)
fig.savefig(OUT / "fig1_selection.pdf")
print("wrote", OUT / "fig1_selection.png")
