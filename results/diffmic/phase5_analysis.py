"""Phase 5 analysis: DiffMIC (MICCAI 2023) on APTOS 2019 - as released vs leakage-fixed selection."""
import glob, os, pickle, sys
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, cohen_kappa_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

RUNS, OUT, AUD = sys.argv[1], sys.argv[2], sys.argv[3]   # runs dir, output dir, audit folder (manifest + released train pkl)
os.makedirs(OUT, exist_ok=True)
meta = pd.read_csv(os.path.join(os.path.dirname(RUNS.rstrip("/")), "test_meta.csv"))
m = pd.read_csv(f"{AUD}/audit_aptos_v1/manifest_groups.csv"); m["id"] = m.path.str.extract(r"([0-9a-f]{12})\.png")[0]
tr = pd.DataFrame([(os.path.splitext(os.path.basename(d["img_root"]))[0], int(d["label"]))
                   for d in pickle.load(open(f"{AUD}/phase3/splits/diffmic_aptos_train.pkl", "rb"))], columns=["id", "label"])
tr["g"] = tr.id.map(dict(zip(m.id, m.group_id)))
glabels = tr.groupby("g").label.apply(set)
meta["train_copy_grades"] = meta.group_id.map(glabels)
dup = meta.has_train_duplicate.values
same = np.array([d and (s == {l}) for d, s, l in zip(dup, meta.train_copy_grades, meta.label)])
conf = dup & ~same
clean = ~dup
y = meta.label.values

def load(run, split):
    out = {}
    for f in sorted(glob.glob(f"{run}/logs/aptos/split_0/evals/e*_{split}.npz")):
        e = int(os.path.basename(f)[1:5]); z = np.load(f); out[e] = z["prob"].argmax(1)
    return out
def metr(p, mask=None):
    yy, pp = (y, p) if mask is None else (y[mask], p[mask])
    return dict(acc=(pp == yy).mean(), f1=f1_score(yy, pp, average="macro"), qwk=cohen_kappa_score(yy, pp, weights="quadratic"))

rows, curves = [], {}
for run in sorted(glob.glob(f"{RUNS}/*/")):
    name = os.path.basename(run.rstrip("/")); T = load(run, "test"); V = load(run, "val")
    ep = sorted(T); acc = np.array([(T[e] == y).mean() for e in ep]); curves[name] = (np.array(ep), acc)
    picks = {"as_released (max test acc)": ep[int(acc.argmax())], "last_epoch": ep[-1]}
    if V:
        vy = None
        vz = np.load(sorted(glob.glob(f"{run}/logs/aptos/split_0/evals/e*_val.npz"))[0])["y"]
        vacc = np.array([(V[e] == vz).mean() for e in ep]); picks["val_selected"] = ep[int(vacc.argmax())]
        curves[name + "_val"] = (np.array(ep), vacc)
    for rule, e in picks.items():
        p = T[e]; r = dict(run=name, condition="R0 released split" if name.startswith("R0") else "R1 train/val/test", rule=rule, epoch=e)
        r.update({f"{k}": v for k, v in metr(p).items()})
        r.update({f"dedup_{k}": v for k, v in metr(p, ~dup).items()})
        r["acc_clean_imgs"] = (p[clean] == y[clean]).mean()
        r["acc_dup_same_grade"] = (p[same] == y[same]).mean(); r["acc_dup_conflict_grade"] = (p[conf] == y[conf]).mean()
        r["n_dup_same"] = int(same.sum()); r["n_dup_conflict"] = int(conf.sum())
        rows.append(r)
    rows.append(dict(run=name, rule="all_evals_mean", acc=acc.mean(), acc_max_minus_mean=acc.max() - acc.mean(),
                     acc_sd_last10=acc[-10:].std(ddof=1)))
df = pd.DataFrame(rows); df.to_csv(f"{OUT}/phase5_per_run.csv", index=False)

def ci(x):
    x = np.asarray(x, float); h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); return x.mean(), x.mean() - h, x.mean() + h
R1 = df[df.run.str.startswith("R1")]
a = R1[R1.rule.str.startswith("as_released")].set_index("run"); v = R1[R1.rule == "val_selected"].set_index("run"); l = R1[R1.rule == "last_epoch"].set_index("run")
summ = []
for k in ["acc", "f1", "qwk", "dedup_acc"]:
    d = a[k] - v.loc[a.index, k]
    summ.append(dict(metric=k, as_released_mean=a[k].mean(), val_selected_mean=v[k].mean(), last_epoch_mean=l[k].mean(),
                     inflation_mean=ci(d)[0], inflation_ci_lo=ci(d)[1], inflation_ci_hi=ci(d)[2], n_seeds=len(d)))
S = pd.DataFrame(summ); S.to_csv(f"{OUT}/phase5_selection_inflation.csv", index=False)
print(S.round(4).to_string()); print(df[df.rule != "all_evals_mean"][["run", "rule", "epoch", "acc", "f1", "qwk", "dedup_acc", "acc_clean_imgs", "acc_dup_same_grade", "acc_dup_conflict_grade"]].round(4).to_string())

# ---------- figures (static, print) ----------
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[2.2, 1]))
for name in sorted(k for k in curves if k.startswith("R1") and not k.endswith("_val")):
    e, acc = curves[name]; ax.plot(e, acc * 100, color="#9a9994", lw=1.2, zorder=1)
    ia = acc.argmax(); iv = list(e).index(int(v.loc[name, "epoch"]))
    ax.scatter(e[ia], acc[ia] * 100, s=34, color=ORANGE, edgecolor="white", linewidth=1.2, zorder=3)
    ax.scatter(e[iv], acc[iv] * 100, s=34, color=BLUE, edgecolor="white", linewidth=1.2, zorder=3)
ax.set_xlabel("Training epoch"); ax.set_ylabel("Test accuracy (%)"); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_ylim(70, 87); ax.set_title("A  Test accuracy at each of 101 evaluations (3 seeds)", loc="left", fontsize=9, color=INK)
ax.scatter([], [], s=34, color=ORANGE, label="Best on test set (as released)"); ax.scatter([], [], s=34, color=BLUE, label="Best on validation set")
ax.legend(frameon=False, loc="lower right", fontsize=8)
for name in a.index:
    bx.plot([0, 1], [a.loc[name, "acc"] * 100, v.loc[name, "acc"] * 100], color="#9a9994", lw=1.2, zorder=1)
    bx.scatter([0], [a.loc[name, "acc"] * 100], s=34, color=ORANGE, edgecolor="white", linewidth=1.2, zorder=3)
    bx.scatter([1], [v.loc[name, "acc"] * 100], s=34, color=BLUE, edgecolor="white", linewidth=1.2, zorder=3)
bx.set_xticks([0, 1], ["As released", "Validation-\nselected"]); bx.set_xlim(-0.4, 1.4); bx.set_ylim(80, 87)
bx.grid(axis="y", color=GRID, lw=0.8); bx.set_axisbelow(True)
m0, lo, hi = S.loc[S.metric == "acc", ["inflation_mean", "inflation_ci_lo", "inflation_ci_hi"]].values[0] * 100
bx.set_title(f"B  Inflation {m0:.1f} pts\n    (95% CI {lo:.1f}–{hi:.1f})", loc="left", fontsize=9, color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_selection_inflation.png", dpi=300); fig.savefig(f"{OUT}/fig_selection_inflation.pdf")

fig, ax = plt.subplots(figsize=(3.6, 2.8))
cats = [("Clean\n(no train copy)", "acc_clean_imgs", int(clean.sum())), ("Train copy,\nsame grade", "acc_dup_same_grade", int(same.sum())),
        ("Train copy,\ndifferent grade", "acc_dup_conflict_grade", int(conf.sum()))]
rel = df[df.rule.str.startswith("as_released")]
for i, (lab, k, n) in enumerate(cats):
    vals = rel[k].values * 100
    ax.bar(i, vals.mean(), width=0.6, color=BLUE if i else "#9a9994", zorder=2)
    ax.scatter(np.full(len(vals), i) + np.linspace(-0.12, 0.12, len(vals)), vals, s=12, color=INK, zorder=3)
    ax.text(i, vals.max() + 5, f"{vals.mean():.0f}%\nn={n}", ha="center", va="bottom", fontsize=8, color=INK)
ax.set_xticks(range(3), [c[0] for c in cats], fontsize=8); ax.set_ylim(0, 118); ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel("Test accuracy (%)"); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_title(f"Accuracy by duplicate status ({len(rel)} runs)", loc="left", fontsize=9, color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_duplicates.png", dpi=300); fig.savefig(f"{OUT}/fig_duplicates.pdf")
