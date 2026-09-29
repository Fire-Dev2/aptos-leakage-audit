"""Group-aware stratified 70/15/15 split of APTOS 2019 for the GCN 'fixed' condition.
No duplicate group (Phase 2 audit) crosses train/val/test. Stratified on each group's grade
(for the few grade-conflict groups: the most common grade, ties -> lowest). ids only."""
import sys, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
m = pd.read_csv(sys.argv[1]); out = sys.argv[2]
m["id"] = m.path.str.extract(r"([0-9a-f]{12})\.png")[0]
glab = m.groupby("group_id").label.agg(lambda s: s.value_counts().sort_index().idxmax())
m["strat"] = m.group_id.map(glab)
sg = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=0)
fold = np.zeros(len(m), int)
for f, (_, te) in enumerate(sg.split(m, m.strat, m.group_id)): fold[te] = f
m["split"] = np.where(fold < 3, "test", np.where(fold < 6, "val", "train"))
for s in ["train", "val", "test"]:
    m[m.split == s][["id", "label"]].to_csv(f"{out}/g_{s}_ids.csv", index=False)
g = {s: set(m[m.split == s].group_id) for s in ["train", "val", "test"]}
assert not (g["train"] & g["val"]) and not (g["train"] & g["test"]) and not (g["val"] & g["test"])
print(m.groupby(["split", "label"]).size().unstack().to_string()); print(m.split.value_counts().to_dict())
