import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
df = pd.read_csv("gcn_per_run.csv"); df = df[df.model == "final_released"]
INK, MUTED, GRID, GREY = "#0b0b0b", "#52514e", "#e6e5e1", "#9a9994"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
bars = [("A", "class-sorted (released)", "Copies made before split,\ntest batches sorted by grade\n(as described)"),
        ("A", "random order (mean of 5)", "Copies made before split,\ntest batches in random order"),
        ("B", "class-sorted (released)", "Split first (duplicate-aware),\ntest batches sorted by grade"),
        ("B", "random order (mean of 5)", "Split first (duplicate-aware),\ntest batches in random order")]
cols = ["#4a4945", "#8a8983", "#8a8983", "#2a78d6"]
fig, ax = plt.subplots(figsize=(7.2, 3.3))
for i, (c, o, lab) in enumerate(bars):
    v = df[(df.cond == c) & (df.order == o)].acc.values * 100
    ax.bar(i, v.mean(), width=0.62, color=cols[i], zorder=2)
    ax.scatter(np.full(len(v), i) + np.linspace(-0.1, 0.1, len(v)), v, s=12, color=INK, zorder=3)
    ax.text(i, v.max() + 0.8, f"{v.mean():.1f}%", ha="center", va="bottom", fontsize=9, color=INK)
ax.axhline(98.45, color="#eb6834", lw=1.2, ls="--", zorder=1)
ax.text(3.45, 98.45 + 0.4, "Reported: 98.45%", ha="right", va="bottom", fontsize=8, color="#eb6834")
ax.set_xticks(range(4), [b[2] for b in bars], fontsize=7.5); ax.set_ylim(70, 102); ax.set_ylabel("Test accuracy (%)")
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_title("Graph-enhanced DR classifier (PLOS Comput Biol 2025), MobileViT, 3 seeds each", loc="left", fontsize=9, color=INK)
fig.tight_layout(); fig.savefig("fig_gcn.png", dpi=300); fig.savefig("fig_gcn.pdf")
