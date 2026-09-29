"""
make_splits.py — build the Phase 5 split files for DiffMIC from its released split + the Phase 2 audit.

Inputs : DiffMIC's released aptos_train.pkl / aptos_test.pkl, Phase 2 manifest_groups.csv
Outputs (ids only, no images — safe to share under the APTOS rules):
  r0_train_ids.csv  r0_test_ids.csv   released split, unchanged              (condition R0)
  r1_train_ids.csv  r1_val_ids.csv    released train minus a 15% validation  (condition R1)
  test_meta.csv                       per test image: label, has_train_duplicate, grade_conflict, group
"""
import argparse, os, pickle
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ap = argparse.ArgumentParser()
ap.add_argument("--train_pkl", required=True); ap.add_argument("--test_pkl", required=True)
ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--val_folds", type=int, default=7, help="1/val_folds of train goes to validation (~14%)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--batch", type=int, default=32, help="DiffMIC training batch size")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

stem = lambda p: os.path.splitext(os.path.basename(p))[0]
tr = [(stem(d["img_root"]), int(d["label"])) for d in pickle.load(open(a.train_pkl, "rb"))]
te = [(stem(d["img_root"]), int(d["label"])) for d in pickle.load(open(a.test_pkl, "rb"))]
m = pd.read_csv(a.manifest); m["id"] = m.path.str.extract(r"([0-9a-f]{12})\.png")[0]
gid = dict(zip(m.id, m.group_id)); gsz = dict(zip(m.id, m.group_size)); conf = dict(zip(m.id, m.group_label_conflict))

pd.DataFrame(tr, columns=["id", "label"]).to_csv(f"{a.out}/r0_train_ids.csv", index=False)
pd.DataFrame(te, columns=["id", "label"]).to_csv(f"{a.out}/r0_test_ids.csv", index=False)

train_groups = {gid[i] for i, _ in tr}
test_groups = {gid[i] for i, _ in te}
meta = pd.DataFrame(te, columns=["id", "label"])
meta["group_id"] = meta.id.map(gid)
meta["has_train_duplicate"] = [(gsz[i] > 1) and (gid[i] in train_groups) for i in meta.id]
meta["grade_conflict_group"] = meta.id.map(conf).fillna(False)
meta.to_csv(f"{a.out}/test_meta.csv", index=False)

# validation carve-out: groups touching the test set are never eligible for validation
T = pd.DataFrame(tr, columns=["id", "label"]); T["g"] = T.id.map(gid)
eligible = T[~T.g.isin(test_groups)].reset_index(drop=True)
sgkf = StratifiedGroupKFold(n_splits=a.val_folds, shuffle=True, random_state=a.seed)
_, vi = next(sgkf.split(eligible, eligible.label, eligible.g))
val_ids = set(eligible.id.iloc[vi])
# DiffMIC's train loader keeps the last partial batch (a batch of 1 crashes BatchNorm). Give R1 the same
# last-batch size as the released R0 split by moving singleton (non-duplicated) grade-0 images, in id order,
# from validation back to train. NOTE: StratifiedGroupKFold output differs across scikit-learn versions;
# the released splits/*.csv files are the canonical split (made with this rule; 3 images moved, val = 352).
lab_of = T.set_index("id").label
while (len(T) - len(val_ids)) % a.batch != len(tr) % a.batch:
    single = sorted(i for i in val_ids if gsz[i] == 1 and lab_of[i] == 0)
    val_ids.discard(single[0]); print("moved", single[0], "val -> train (match last-batch size)")
T[T.id.isin(val_ids)][["id", "label"]].to_csv(f"{a.out}/r1_val_ids.csv", index=False)
T[~T.id.isin(val_ids)][["id", "label"]].to_csv(f"{a.out}/r1_train_ids.csv", index=False)

v = T[T.id.isin(val_ids)]; r = T[~T.id.isin(val_ids)]
assert not (set(v.g) & set(r.g)), "val and train share a duplicate group"
assert not (set(v.g) & test_groups), "val and test share a duplicate group"
print(f"R0 train {len(tr)}  test {len(te)}  | test with train duplicate: {int(meta.has_train_duplicate.sum())}")
print(f"R1 train {len(r)}  val {len(v)}  | val grade counts {v.label.value_counts().sort_index().to_dict()}")
print(f"   train grade counts {r.label.value_counts().sort_index().to_dict()}")
