"""Score audit output (pairs.csv) against planted ground truth (truth.json)."""
import json
import sys
from pathlib import Path

import pandas as pd

out, truth_file = Path(sys.argv[1]), Path(sys.argv[2])
pairs = pd.read_csv(out / "pairs.csv")
truth = json.load(open(truth_file))


def key(p):
    return Path(p).stem


pairs["key"] = [frozenset((key(a), key(b))) for a, b in zip(pairs.path_i, pairs.path_j)]
T = {frozenset((key(t["a"]), key(t["b"]))): t["kind"] for t in truth}

g = pairs.generators.fillna("")
det = {
    "tier: exact": pairs.tier == "exact",
    "tier: exact+near_verified": pairs.tier.isin(["exact", "near_verified"]),
    "any candidate (all tiers)": pairs.tier.notna(),
    "gen: phash": g.str.contains("phash"),
    "gen: dhash": g.str.contains("dhash"),
    "gen: ncc": g.str.contains("ncc"),
    "gen: embed": g.str.contains("embed"),
}
print(f"{'detector':26s} {'TP':>4s} {'FP':>6s} {'FN':>4s}  missed kinds")
for name, mask in det.items():
    found = set(pairs.key[mask])
    tp = found & set(T)
    fp = found - set(T)
    fn = set(T) - found
    missed = sorted({T[k] for k in fn})
    print(f"{name:26s} {len(tp):4d} {len(fp):6d} {len(fn):4d}  {missed}")

s = json.load(open(out / "summary.json"))
if "official_split" in s:
    print("\nofficial split:", json.dumps(s["official_split"]))
print("groups:", json.dumps({k: v for k, v in s["groups"].items() if k != "size_hist"}))
