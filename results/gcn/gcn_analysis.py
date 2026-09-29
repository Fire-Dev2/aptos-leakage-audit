import glob, json, os, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score
def met(z, perm=None):
    p, y = z["prob"], z["y"]
    if perm is not None:  # back to sorted order (not needed for metrics, kept for per-image checks)
        inv = np.argsort(perm); p, y = p[inv], y[inv]
    pr = p.argmax(1)
    return dict(acc=accuracy_score(y, pr), f1=f1_score(y, pr, average="macro"), kappa=cohen_kappa_score(y, pr),
                qwk=cohen_kappa_score(y, pr, weights="quadratic"), auc=roc_auc_score(y, p, multi_class="ovr")), pr, y
rows = []
for d in sorted(glob.glob("runs/*_seed*/")):
    name = os.path.basename(d.rstrip("/")); meta = json.load(open(d + "leak_meta.json"))
    for tag in ["final_released", "true_bestval"]:
        m, _, _ = met(np.load(f"{d}{tag}_sorted.npz")); rows.append(dict(run=name, model=tag, order="class-sorted (released)", **m))
        pm = [met(np.load(f"{d}{tag}_perm{k}.npz"))[0] for k in range(5)]
        rows.append(dict(run=name, model=tag, order="random order (mean of 5)", **{k: np.mean([x[k] for x in pm]) for k in pm[0]}))
        m, _, _ = met(np.load(f"{d}{tag}_bs1.npz")); rows.append(dict(run=name, model=tag, order="one image at a time", **m))
    rows[-6]["stop_epoch"] = meta["stop_epoch"]; rows[-6]["true_best_val_epoch"] = meta["true_best_val_epoch"]
df = pd.DataFrame(rows); df["cond"] = df.run.str[0]
df.to_csv("gcn_per_run.csv", index=False)
pd.set_option("display.width", 200)
S = df.groupby(["cond", "model", "order"])[["acc", "f1", "kappa", "qwk", "auc"]].agg(["mean", "std"]).round(4)
print(S.to_string())
S.to_csv("gcn_summary.csv")
