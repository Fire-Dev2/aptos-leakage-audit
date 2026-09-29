"""Phase 5 analysis, paper 2: nnMobileNet (CVPRW 2024) on APTOS 2019 - released test-set checkpoint rule vs validation selection.
Usage: python nnmb_analysis.py <nnmb results dir (has runs/, test_meta.csv, csv/)> <out dir> <audit folder>
Metrics mirror the released engine.py: accuracy (= its 'f1' and 'spec', both micro-averaged), quadratic kappa,
AUC on the flattened one-hot matrix (micro AUC, as released); macro-F1 added for comparability with DiffMIC.
Selection: the released code keeps the first epoch with the best TEST score (strict >), separately for the model and its EMA."""
import glob, os, pickle, sys
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score

D, OUT, AUD = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
meta = pd.read_csv(f"{D}/test_meta.csv"); y = meta.label.values
m = pd.read_csv(f"{AUD}/audit_aptos_v1/manifest_groups.csv"); m["id"] = m.path.str.extract(r"([0-9a-f]{12})\.png")[0]
tr = pd.DataFrame([(os.path.splitext(os.path.basename(d["img_root"]))[0], int(d["label"]))
                   for d in pickle.load(open(f"{AUD}/phase3/splits/diffmic_aptos_train.pkl", "rb"))], columns=["id", "label"])
assert set(tr.id) == set(pd.read_csv(f"{D}/csv/r0_train.csv").id), "nnMobileNet R0 train split differs from DiffMIC's"
tr["g"] = tr.id.map(dict(zip(m.id, m.group_id)))
glabels = tr.groupby("g").label.apply(set)
dup = meta.has_train_duplicate.values
same = np.array([d and (glabels.get(g) == {l}) for d, g, l in zip(dup, meta.group_id, meta.label)])
conf = dup & ~same; clean = ~dup
Y1 = np.eye(5)[y]

def load(run, split, w):
    fs = sorted(glob.glob(f"{run}/evals/e*_{split}_{w}.npz"))
    ep = np.array([int(os.path.basename(f)[1:5]) for f in fs])
    if not fs: return ep, None, None
    P = np.stack([np.load(f)["prob"] for f in fs]); yy = np.load(fs[0])["y"] if fs else None
    return ep, P, yy
def metr(p):
    pr = p.argmax(1)
    return dict(acc=(pr == y).mean(), auc_micro=roc_auc_score(Y1.ravel(), p.ravel()), qwk=cohen_kappa_score(y, pr, weights="quadratic"),
                f1_macro=f1_score(y, pr, average="macro"), dedup_acc=(pr[~dup] == y[~dup]).mean(),
                acc_clean_imgs=(pr[clean] == y[clean]).mean(), acc_dup_same_grade=(pr[same] == y[same]).mean(),
                acc_dup_conflict_grade=(pr[conf] == y[conf]).mean(),
                conflict_pred_equals_train_copy=np.mean([pr[i] in glabels[meta.group_id[i]] for i in np.where(conf)[0]]))

rows, curves = [], {}
for run in sorted(glob.glob(f"{D}/runs/*/")):
    name = os.path.basename(run.rstrip("/"))
    for w in ["model", "ema"]:
        ep, P, _ = load(run, "test", w); assert len(ep) == 1000 and (ep == np.arange(1000)).all(), (name, w, len(ep))
        acc = (P.argmax(2) == y).mean(1); curves[(name, w)] = acc
        picks = {"as_released (best test acc)": int(acc.argmax()), "last_epoch": 999}
        vep, VP, vy = load(run, "val", w)
        if len(vep):
            vacc = (VP.argmax(2) == vy).mean(1); curves[(name, w, "val")] = vacc; picks["val_selected"] = int(vacc.argmax())
        for rule, e in picks.items():
            r = dict(run=name, weights=w, rule=rule, epoch=e); r.update(metr(P[e])); rows.append(r)
        rows.append(dict(run=name, weights=w, rule="summary", acc=acc.mean(), acc_last100_mean=acc[-100:].mean(),
                         acc_last100_sd=acc[-100:].std(ddof=1), acc_max=acc.max()))
df = pd.DataFrame(rows); df.to_csv(f"{OUT}/nnmb_per_run.csv", index=False)

def ci(x):
    x = np.asarray(x, float); h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); return x.mean(), x.mean() - h, x.mean() + h
summ = []
for w in ["model", "ema"]:
    R1 = df[df.run.str.startswith("R1") & (df.weights == w)]
    a = R1[R1.rule.str.startswith("as_released")].set_index("run"); v = R1[R1.rule == "val_selected"].set_index("run")
    l = R1[R1.rule == "last_epoch"].set_index("run")
    for k in ["acc", "f1_macro", "qwk", "auc_micro", "dedup_acc"]:
        d = a[k] - v.loc[a.index, k]; c = ci(d)
        summ.append(dict(weights=w, metric=k, as_released_mean=a[k].mean(), val_selected_mean=v[k].mean(), last_epoch_mean=l[k].mean(),
                         inflation_mean=c[0], inflation_ci_lo=c[1], inflation_ci_hi=c[2], n_seeds=len(d)))
    R0 = df[df.run.str.startswith("R0") & (df.weights == w) & df.rule.str.startswith("as_released")]
    for k in ["acc", "auc_micro", "qwk", "f1_macro"]:
        c = ci(R0[k]); summ.append(dict(weights=w, metric="R0_as_released_" + k, as_released_mean=c[0], inflation_ci_lo=c[1], inflation_ci_hi=c[2], n_seeds=len(R0)))
S = pd.DataFrame(summ); S.to_csv(f"{OUT}/nnmb_selection_inflation.csv", index=False)
pd.set_option("display.width", 250)
print(S.round(4).to_string())
print(df[df.rule != "summary"][["run", "weights", "rule", "epoch", "acc", "auc_micro", "qwk", "f1_macro", "dedup_acc", "acc_clean_imgs",
                                 "acc_dup_same_grade", "acc_dup_conflict_grade", "conflict_pred_equals_train_copy"]].round(4).to_string())
print(df[df.rule == "summary"][["run", "weights", "acc", "acc_last100_mean", "acc_last100_sd", "acc_max"]].round(4).to_string())
np.savez_compressed(f"{OUT}/nnmb_curves.npz", **{"|".join(k): v for k, v in curves.items()})
