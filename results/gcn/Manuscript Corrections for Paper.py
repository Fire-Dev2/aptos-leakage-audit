#!/usr/bin/env python
"""
gcn_referable.py - referable-DR sensitivity and specificity for the GCN reruns.

Run from the folder that contains runs/ (unpacked results_gcn.tgz):  python gcn_referable.py
Referable DR = grade >= 2. A test image counts as flagged when the predicted grade (argmax) is >= 2.
No threshold is tuned and nothing is retrained: the numbers come from the saved test predictions.

Outputs
  gcn_referable_per_run.csv   one row per run x model x test order
  gcn_referable_summary.csv   mean and SD (ddof=1) over the three seeds
"""
import glob, os, numpy as np, pandas as pd

REF = 2
ORDERS = [("class-sorted (released)", ["{tag}_sorted.npz"]),
          ("random order (mean of 5)", ["{tag}_perm%d.npz" % k for k in range(5)]),
          ("one image at a time", ["{tag}_bs1.npz"])]

rows = []
for d in sorted(glob.glob("runs/*_seed*")):
    name = os.path.basename(d)
    files = pd.read_csv(f"{d}/test_files.csv")
    is_orig = ~files.path.map(os.path.basename).str.contains("__").values   # augmented copies are named src__tagNNNNN.png
    for tag in ["final_released", "true_bestval"]:
        for order, pats in ORDERS:
            acc = []; sens = []; spec = []; sens_o = []; missed = []; as0 = []; as1 = []; false_ref = []
            for pat in pats:
                z = np.load(f"{d}/{pat.format(tag=tag)}")
                y = z["y"].astype(int); pred = z["prob"].argmax(1)
                orig = is_orig[z["perm"]] if "perm" in z.files else is_orig
                t, q = y >= REF, pred >= REF
                acc.append((pred == y).mean())
                sens.append((q & t).sum() / t.sum()); spec.append((~q & ~t).sum() / (~t).sum())
                sens_o.append((q & t & orig).sum() / (t & orig).sum())
                missed.append((~q & t).sum()); false_ref.append((q & ~t).sum())
                as0.append((t & (pred == 0)).sum()); as1.append((t & (pred == 1)).sum())
            rows.append(dict(run=name, cond=name[0], model=tag, order=order, n_test=len(y), n_referable=int(t.sum()),
                             acc=np.mean(acc), sensitivity=np.mean(sens), specificity=np.mean(spec),
                             sensitivity_original_images_only=np.mean(sens_o), missed_referable=np.mean(missed),
                             missed_predicted_grade0=np.mean(as0), missed_predicted_grade1=np.mean(as1),
                             false_referrals=np.mean(false_ref)))
df = pd.DataFrame(rows)
df.to_csv("gcn_referable_per_run.csv", index=False)
cols = ["acc", "sensitivity", "specificity", "sensitivity_original_images_only", "missed_referable", "false_referrals"]
S = df.groupby(["cond", "model", "order"])[cols].agg(["mean", "std"]).round(4)
S.to_csv("gcn_referable_summary.csv")
print(S.loc[(slice(None), "final_released"), ["acc", "sensitivity", "specificity"]].mul(100).round(2).to_string())
