"""Study-selection flow (PRISMA). Supplementary: not a figure of the paper. The counts are those of
data/screening/prisma_counts.csv. Writes figures/prisma_flow.png; run from anywhere: python figures/prisma.py"""
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
INK, MUTED, LINE, FILL, ACC = "#0b0b0b", "#52514e", "#8a8983", "#f3f2ee", "#2a78d6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
fig, ax = plt.subplots(figsize=(3.5, 4.4)); ax.set_xlim(0, 100); ax.set_ylim(0, 132); ax.axis("off")
def box(x, y, w, h, t, bold=False, fc=FILL, ec=LINE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=0.7))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=5.4, color=INK, fontweight="bold" if bold else "normal", linespacing=1.25)
def arr(x1, y1, x2, y2): ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.7, color=MUTED, shrinkA=0, shrinkB=0))
box(1, 116, 47, 14, "Original searches\n(arXiv, journals, conferences,\n352 GitHub repos)\nn = 104")
box(52, 116, 47, 14, "Europe PMC database search\n(2020–2025, 25 Sep 2026)\nn = 268")
arr(24.5, 116, 40, 106); arr(75.5, 116, 60, 106)
box(20, 96, 60, 10, "Records after removing 28 duplicates\nn = 344")
arr(50, 96, 50, 88)
box(20, 76, 60, 12, "Screened on title / record\nn = 344")
box(61, 58, 38, 16, "Excluded at title screen\nn = 95\n(review/survey, not\nclassification, off-topic)", ec=LINE)
arr(80, 82, 88, 74)
arr(50, 76, 50, 68)
box(1, 52, 56, 16, "Assessed against full inclusion criteria\n(full text, code repository)\nn = 249")
box(61, 20, 38, 34, "Excluded, n = 237\nE1 year: 13\nE2 APTOS not dev+test: 46\nE3 not classification: 4\nE4 no APTOS test metric: 5\nE5 no public code: 162\nE6 incomplete code: 3\nE8 no identifiable paper: 4")
arr(57, 58, 61, 45)
arr(29, 52, 29, 44)
box(1, 30, 56, 14, "Included studies\nn = 12", bold=True, fc="#e3eefb", ec=ACC)
arr(29, 30, 29, 22)
box(1, 4, 56, 18, "Leakage coding of all 12\n(independent second rater on 6)\nCode reproduction, 3 seeds × 2 conditions\n(DiffMIC, nnMobileNet, GCN)")
fig.savefig(Path(__file__).with_name("prisma_flow.png"), dpi=400, bbox_inches="tight")
