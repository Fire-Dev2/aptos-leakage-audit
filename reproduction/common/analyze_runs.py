"""
analyze_runs.py — turn the per-epoch prediction dumps into the Phase 5 numbers.

For every run folder (…/evals/eXXXX_test.npz [+ eXXXX_val.npz]) it reports:
  as_released   test accuracy of the epoch with the best TEST accuracy (what DiffMIC's ckpt_best does)
  val_selected  test accuracy of the epoch with the best VALIDATION accuracy (R1 runs only)
  last_epoch    test accuracy at the final evaluation
  … each also on the de-duplicated test subset, plus accuracy on the duplicated vs clean test images.
Then aggregates over seeds with mean and 95% t-interval.

Accuracy is computed per image (DiffMIC's own log averages per batch; both are reported).

Usage: python analyze_runs.py --runs RUN_DIR [RUN_DIR ...] --meta test_meta.csv --out results.csv
"""
import argparse, glob, os, re
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, cohen_kappa_score

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--meta", required=True)
ap.add_argument("--out", default="phase5_results.csv"); ap.add_argument("--batch", type=int, default=32)
a = ap.parse_args()
meta = pd.read_csv(a.meta); dup = meta.has_train_duplicate.values.astype(bool)

def load(run, split):
    out = {}
    for f in sorted(glob.glob(os.path.join(run, "**", "evals", f"e*_{split}.npz"), recursive=True)):
        e = int(re.search(r"e(\d+)_", os.path.basename(f)).group(1)); z = np.load(f)
        out[e] = (z["prob"].argmax(1), z["y"].astype(int))
    return out

def m(pred, y, mask=None):
    if mask is not None: pred, y = pred[mask], y[mask]
    return dict(acc=(pred == y).mean(), f1_macro=f1_score(y, pred, average="macro"),
                qwk=cohen_kappa_score(y, pred, weights="quadratic"), n=len(y))

def batch_acc(pred, y, b):  # DiffMIC's reported accuracy: mean over batches
    return np.mean([(pred[i:i+b] == y[i:i+b]).mean() for i in range(0, len(y), b)])

rows = []
for run in a.runs:
    T, V = load(run, "test"), load(run, "val")
    if not T: print("no evals in", run); continue
    ep = sorted(T)
    y0 = T[ep[0]][1]; assert len(y0) == len(meta) and (y0 == meta.label.values).all(), "test order/labels mismatch"
    acc = {e: (T[e][0] == T[e][1]).mean() for e in ep}
    picks = {"as_released": max(ep, key=lambda e: acc[e]), "last_epoch": ep[-1]}
    if V:
        vacc = {e: (V[e][0] == V[e][1]).mean() for e in V}
        picks["val_selected"] = max(sorted(vacc), key=lambda e: vacc[e])
    for rule, e in picks.items():
        p, y = T[e]
        r = dict(run=os.path.basename(os.path.normpath(run)), rule=rule, epoch=e, n_evals=len(ep),
                 batch_acc_as_logged=batch_acc(p, y, a.batch))
        r.update({f"all_{k}": v for k, v in m(p, y).items()})
        r.update({f"dedup_{k}": v for k, v in m(p, y, ~dup).items()})
        r["acc_on_duplicated_imgs"] = (p[dup] == y[dup]).mean(); r["acc_on_clean_imgs"] = (p[~dup] == y[~dup]).mean()
        rows.append(r)
    # how noisy is a single evaluation? spread of test accuracy over the last 10 evaluations
    last = [acc[e] for e in ep[-10:]]
    rows.append(dict(run=os.path.basename(os.path.normpath(run)), rule="noise_last10_evals",
                     all_acc=np.mean(last), all_n=len(last), acc_sd_between_evals=np.std(last, ddof=1) if len(last) > 1 else np.nan,
                     acc_max_minus_mean=max(acc.values()) - np.mean(list(acc.values()))))

df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
if df.empty: print("no evaluations yet - nothing to summarise"); raise SystemExit(0)

def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2: return (np.mean(x), np.nan, np.nan)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); return (x.mean(), x.mean() - h, x.mean() + h)

print("\nper-seed rows written to", a.out)
for rule in ["as_released", "val_selected", "last_epoch"]:
    d = df[df.rule == rule]
    if d.empty: continue
    print(f"\n== {rule}  (n runs = {len(d)})")
    for col in ["all_acc", "all_f1_macro", "all_qwk", "dedup_acc", "acc_on_duplicated_imgs", "acc_on_clean_imgs"]:
        mu, lo, hi = ci(d[col]); print(f"  {col:24s} {mu:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
