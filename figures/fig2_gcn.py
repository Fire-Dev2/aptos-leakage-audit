"""Fig. 2 of the paper: test accuracy of the GCN model under condition A (copies made before the split, as
described in the paper) and condition B (duplicate-aware split, training-only oversampling), for three
test orders. Bars are means over three seeds, points are single seeds, the dashed line is the published
accuracy, and the blue bar is the corrected protocol with one image at a time.

Reads   results/gcn/gcn_per_run.csv
Writes  figures/fig2_gcn.png (600 dpi) and figures/fig2_gcn.pdf

Sized for one IEEE column (3.5 in) with 8 pt Times labels. Run from anywhere:  python figures/fig2_gcn.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "results/gcn/gcn_per_run.csv")
df = df[df.model == "final_released"]

PUBLISHED = 98.45          # accuracy reported in the GCN paper
YLIM = (70, 104)           # set to (0, 104) for a zero-based axis
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
DARK, GREY, BLUE, ORANGE = "#4a4945", "#9a9994", "#2a78d6", "#eb6834"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "Tinos", "TeX Gyre Termes", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 0, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})

ORDERS = [("class-sorted (released)", "Grade-\nsorted"), ("random order (mean of 5)", "Random\norder"),
          ("one image at a time", "One\nimage")]
bars = [(c, o, lab) for c in "AB" for o, lab in ORDERS]
x = np.array([0, 1, 2, 3.45, 4.45, 5.45])
colors = [DARK, GREY, GREY, GREY, GREY, BLUE]   # published protocol, context, corrected one-image protocol

fig, ax = plt.subplots(figsize=(3.5, 2.3))
for xi, (c, o, _), col in zip(x, bars, colors):
    v = df[(df.cond == c) & (df.order == o)].acc.values * 100
    assert len(v) == 3, (c, o, len(v))
    ax.bar(xi, v.mean() - YLIM[0], bottom=YLIM[0], width=0.74, color=col, zorder=2, linewidth=0)
    ax.scatter(xi + np.linspace(-0.2, 0.2, len(v)), v, s=7, color=INK, edgecolor="white", linewidth=0.35, zorder=3)
    ax.text(xi, max(v.max(), PUBLISHED if abs(v.mean() - PUBLISHED) < 1.5 else 0) + 0.9, f"{v.mean():.2f}",
            ha="center", va="bottom", color=INK)
ax.axhline(PUBLISHED, color=ORANGE, lw=0.9, ls=(0, (4, 2.2)), zorder=1)
ax.text(x[-1] + 0.45, PUBLISHED + 0.7, f"Published: {PUBLISHED:.2f}", ha="right", va="bottom", color=INK)
ax.set_xticks(x, [b[2] for b in bars], linespacing=1.0)
ax.tick_params(axis="x", pad=2)
ax.set_xlim(-0.6, 6.05)
ax.set_ylim(*YLIM)
ax.set_yticks(np.arange(YLIM[0] if YLIM[0] else 0, 101, 10 if YLIM[0] else 20))
ax.set_ylabel("Test accuracy (%)")
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_axisbelow(True)
for xm, name in ((x[:3].mean(), "Condition A"), (x[3:].mean(), "Condition B")):
    ax.annotate(name, (xm, 0), xycoords=("data", "axes fraction"), xytext=(0, -27), textcoords="offset points",
                ha="center", va="top", color=INK)
fig.subplots_adjust(left=0.125, right=0.995, top=0.985, bottom=0.235)
fig.savefig(OUT / "fig2_gcn.png", dpi=600)
fig.savefig(OUT / "fig2_gcn.pdf")
print("wrote", OUT / "fig2_gcn.png")
