"""Accuracy by duplicate status (DiffMIC and nnMobileNet, released rule) and the headline table.
Supplementary: this figure is not in the paper; the numbers are the ones quoted in its Section III-D.

Reads   results/diffmic/phase5_per_run.csv, results/diffmic/phase5_selection_inflation.csv,
        results/nnmobilenet/nnmb_per_run.csv, results/nnmobilenet/nnmb_selection_inflation.csv
Writes  figures/fig_duplicates_both.png and results/table_phase5_headline.csv

Run from anywhere:  python figures/fig_duplicates_both.py
(The paired test/validation figure this script used to draw is now figures/fig1_selection.py.)
"""
from pathlib import Path

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
dm = pd.read_csv(ROOT / "results/diffmic/phase5_per_run.csv"); nn = pd.read_csv(ROOT / "results/nnmobilenet/nnmb_per_run.csv")
dmS = pd.read_csv(ROOT / "results/diffmic/phase5_selection_inflation.csv"); nnS = pd.read_csv(ROOT / "results/nnmobilenet/nnmb_selection_inflation.csv")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})

cats = [("Clean\n(no train copy)", "acc_clean_imgs"), ("Train copy,\nsame grade", "acc_dup_same_grade"), ("Train copy,\ndifferent grade", "acc_dup_conflict_grade")]
n = dict(acc_clean_imgs=1048, acc_dup_same_grade=32, acc_dup_conflict_grade=18)
rel = {"DiffMIC": dm[dm.rule.str.startswith("as_released")], "nnMobileNet": nn[nn.rule.str.startswith("as_released") & (nn.weights == "model")]}
fig, ax = plt.subplots(figsize=(4.6, 2.9)); wd = 0.36
for j, (paper, d) in enumerate(rel.items()):
    for i, (lab, k) in enumerate(cats):
        vals = d[k].values * 100; x = i + (j - 0.5) * wd
        ax.bar(x, vals.mean(), width=wd * 0.92, color=["#4a4945", "#b9b8b2"][j], zorder=2, label=paper if i == 0 else None)
        ax.scatter(np.full(len(vals), x) + np.linspace(-0.07, 0.07, len(vals)), vals, s=9, color=INK, zorder=3)
        ax.text(x, vals.max() + 3, f"{vals.mean():.0f}", ha="center", va="bottom", fontsize=7.5, color=INK)
ax.set_xticks(range(3), [f"{c[0]}\nn={n[c[1]]}" for c in cats], fontsize=8); ax.set_ylim(0, 115); ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel("Test accuracy (%)"); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2, bbox_to_anchor=(1, 1.08))
ax.set_title("Accuracy by duplicate status (released rule, 6 runs each)", loc="left", fontsize=9, color=INK, pad=14)
fig.tight_layout(); fig.savefig(OUT / "fig_duplicates_both.png", dpi=300)

# headline table
def row(paper, d, S, w=None):
    q = d if w is None else d[d.weights == w]
    r0 = q[q.run.str.startswith("R0") & q.rule.str.startswith("as_released")]
    r0l = q[q.run.str.startswith("R0") & (q.rule == "last_epoch")]
    s = S[S.metric == "acc"] if w is None else S[(S.metric == "acc") & (S.weights == w)]
    s = s.iloc[0]
    return dict(paper=paper, R0_as_released_acc=r0.acc.mean() * 100, R0_as_released_sd=r0.acc.std(ddof=1) * 100,
                R0_last_epoch_acc=r0l.acc.mean() * 100, R0_dedup_acc=r0.dedup_acc.mean() * 100,
                R1_as_released_acc=s.as_released_mean * 100, R1_val_selected_acc=s.val_selected_mean * 100,
                R1_last_epoch_acc=s.last_epoch_mean * 100, inflation_pts=s.inflation_mean * 100,
                ci_lo=s.inflation_ci_lo * 100, ci_hi=s.inflation_ci_hi * 100,
                same_grade_dup_acc=r0.acc_dup_same_grade.mean() * 100, conflict_dup_acc=r0.acc_dup_conflict_grade.mean() * 100,
                clean_acc=r0.acc_clean_imgs.mean() * 100)
T = pd.DataFrame([row("DiffMIC (MICCAI 2023)", dm, dmS), row("nnMobileNet (CVPRW 2024), model", nn, nnS, "model"),
                  row("nnMobileNet (CVPRW 2024), EMA", nn, nnS, "ema")])
T.to_csv(ROOT / "results/table_phase5_headline.csv", index=False); print(T.round(2).T.to_string())
