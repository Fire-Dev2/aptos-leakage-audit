#!/usr/bin/env python3
"""
leakage_audit.py — duplicate / near-duplicate / leakage audit for image-classification datasets.

Works on
  * <root>/<split>/<class>/<image>   --layout split-class   (brain tumor MRI, Kermany chest X-ray)
  * <root>/<class>/<image>           --layout class         (one pool, no official split)
  * <img-dir>/<id><ext> + CSV        --layout csv           (APTOS 2019: train.csv id_code,diagnosis)

How pairs are found and judged
  Candidate generators (loose, high recall):
    phash / dhash   perceptual hashes of the border-cropped grayscale image
    ncc             pixel correlation of border-cropped 64x64 grayscale thumbnails, ALL pairs
    embed           cosine similarity of pretrained CNN features (catches crops/shifts)
  Tiers (what the numbers in the paper are based on):
    exact           identical file bytes (md5) or identical decoded pixels
    near_verified   ncc >= --ncc-thresh AND pHash distance <= --phash-verify
                    (re-encodes, resizes, brightness changes; both are needed because photos from
                    one camera look alike at thumbnail size — seen on APTOS 2019)
    candidate       flagged by some generator but not verified -> manual review
  Groups (connected components) use exact + near_verified edges, plus any pairs you confirm
  by hand (--confirmed review_sheet.csv). Everything downstream (contamination, clean split)
  is computed on groups.

Subcommands
  audit          run the audit
  review-stats   estimate false-positive rates from a filled-in review_sheet.csv (Wilson 95% CI)

Examples
  python leakage_audit.py audit --layout csv --img-dir aptos/train_images \
      --csv aptos/train.csv --id-col id_code --label-col diagnosis --ext .png --out out_aptos
  python leakage_audit.py audit --layout split-class --root brain_mri --out out_brain
  python leakage_audit.py review-stats --sheet out_aptos/review_sheet.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

try:
    import imagehash
except ImportError:  # pragma: no cover
    sys.exit("Missing dependency: pip install imagehash")

__version__ = "0.3.0"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SPLIT_ALIASES = {
    "train": {"train", "training", "trn"},
    "test": {"test", "testing", "tst"},
    "val": {"val", "valid", "validation", "dev"},
}


# ----------------------------------------------------------------------------- manifest
def canon_split(s) -> str:
    s = str(s).strip().lower()
    for k, names in SPLIT_ALIASES.items():
        if s in names:
            return k
    return s


def build_manifest(args) -> pd.DataFrame:
    rows = []
    if args.layout == "split-class":
        root = Path(args.root)
        for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                for f in sorted(cls_dir.rglob("*")):
                    if f.suffix.lower() in IMG_EXTS:
                        rows.append((str(f), canon_split(split_dir.name), cls_dir.name))
    elif args.layout == "class":
        root = Path(args.root)
        for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for f in sorted(cls_dir.rglob("*")):
                if f.suffix.lower() in IMG_EXTS:
                    rows.append((str(f), "pool", cls_dir.name))
    elif args.layout == "csv":
        df = pd.read_csv(args.csv)
        img_dir = Path(args.img_dir)
        missing = 0
        for _, r in df.iterrows():
            ident = str(r[args.id_col])
            p = img_dir / (ident if Path(ident).suffix else ident + args.ext)
            if not p.exists():
                missing += 1
                continue
            split = canon_split(r[args.split_col]) if args.split_col else "pool"
            rows.append((str(p), split, str(r[args.label_col])))
        if missing:
            print(f"[warn] {missing} CSV rows had no image file", file=sys.stderr)
    if not rows:
        sys.exit("No images found — check --layout / --root / --img-dir.")
    m = pd.DataFrame(rows, columns=["path", "split", "label"])
    m.insert(0, "idx", range(len(m)))
    return m


# ----------------------------------------------------------------------------- per-image features
def load_rgb(path: str) -> Image.Image:
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    return im.convert("RGB")


def autocrop(im: Image.Image, thr: int) -> Image.Image:
    """Crop away the dark background (fundus black border, MRI air). No-op if thr < 0."""
    if thr < 0:
        return im
    g = np.asarray(im.convert("L"))
    ys, xs = np.nonzero(g > thr)
    if len(xs) < 0.05 * g.size:  # almost all dark: leave as is
        return im
    return im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def ncc_vector(g: Image.Image, side: int) -> np.ndarray:
    a = np.asarray(g.resize((side, side), Image.BILINEAR), np.float32).ravel()
    a -= a.mean()
    return a / (np.linalg.norm(a) + 1e-8)


def _features_one(task):
    p, crop_thr, hash_size, ncc_side = task
    with open(p, "rb") as fh:
        md5 = hashlib.md5(fh.read()).hexdigest()
    im = load_rgb(p)
    pix = hashlib.md5(np.asarray(im).tobytes() + str(im.size).encode()).hexdigest()
    g = autocrop(im, crop_thr).convert("L")
    ph = int(str(imagehash.phash(g, hash_size=hash_size)), 16)
    dh = int(str(imagehash.dhash(g, hash_size=hash_size)), 16)
    return md5, pix, im.width, im.height, ph, dh, ncc_vector(g, ncc_side)


def compute_features(m: pd.DataFrame, args):
    from multiprocessing import Pool

    tasks = [(p, args.crop_thr, args.hash_size, args.ncc_side) for p in m["path"]]
    res, t0 = [], time.time()
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap(_features_one, tasks, chunksize=8)):
            res.append(r)
            if (i + 1) % 250 == 0:
                print(f"      {i+1}/{len(m)} ({time.time()-t0:.0f}s)", flush=True)
    m = m.copy()
    m["md5"] = [r[0] for r in res]
    m["pixel_md5"] = [r[1] for r in res]
    m["width"] = [r[2] for r in res]
    m["height"] = [r[3] for r in res]
    V = np.stack([r[6] for r in res]).astype(np.float32)
    return m, [r[4] for r in res], [r[5] for r in res], V


def to_words(ints: list[int], nbits: int) -> np.ndarray:
    nwords = math.ceil(nbits / 64)
    out = np.zeros((len(ints), nwords), dtype=np.uint64)
    mask = (1 << 64) - 1
    for i, v in enumerate(ints):
        for k in range(nwords):
            out[i, k] = (v >> (64 * k)) & mask
    return out


def popcount(a: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(a)
    return np.unpackbits(a.view(np.uint8), axis=-1).reshape(*a.shape, 64).sum(-1)


def hamming_pairs(W: np.ndarray, thresh: int, chunk: int = 256):
    n = len(W)
    for s in range(0, n, chunk):
        d = popcount(W[s : s + chunk, None, :] ^ W[None, :, :]).sum(-1)
        for a, b in zip(*np.nonzero(d <= thresh)):
            if s + a < b:
                yield int(s + a), int(b)


def sim_pairs(X: np.ndarray, thresh: float, chunk: int = 512):
    """All pairs i<j with X_i·X_j >= thresh; also returns each row's best non-self match."""
    n = len(X)
    best = np.full(n, -1.0, np.float32)
    best_j = np.full(n, -1, np.int64)
    hits = []
    for s in range(0, n, chunk):
        S = X[s : s + chunk] @ X.T
        rows = np.arange(S.shape[0])
        S[rows, s + rows] = -np.inf
        bj = S.argmax(1)
        best[s : s + chunk] = S[rows, bj]
        best_j[s : s + chunk] = bj
        for a, b in zip(*np.nonzero(S >= thresh)):
            if s + a < b:
                hits.append((int(s + a), int(b)))
    return hits, best, best_j


def compute_embeddings(m: pd.DataFrame, args) -> np.ndarray:
    import torch
    import torchvision
    from torchvision import transforms as T

    if args.weights == "none":
        print("      [warn] --weights none: untrained network, cosine values are NOT meaningful")
        model = getattr(torchvision.models, args.arch)(weights=None)
    elif args.weights_file:  # offline: a downloaded torchvision .pth file
        model = getattr(torchvision.models, args.arch)(weights=None)
        model.load_state_dict(torch.load(args.weights_file, map_location="cpu"))
    else:
        model = getattr(torchvision.models, args.arch)(weights="DEFAULT")
    for head in ("fc", "classifier", "heads"):
        if hasattr(model, head):
            setattr(model, head, torch.nn.Identity())
            break
    model.eval().to(args.device)
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    feats, paths, t0 = [], list(m["path"]), time.time()
    with torch.no_grad():
        for s in range(0, len(paths), args.batch):
            x = torch.stack([tf(autocrop(load_rgb(p), args.crop_thr)) for p in paths[s : s + args.batch]])
            f = torch.nn.functional.normalize(model(x.to(args.device)).flatten(1), dim=1)
            feats.append(f.cpu().numpy().astype(np.float32))
            if (s // args.batch) % 25 == 0:
                print(f"      {min(s+args.batch, len(paths))}/{len(paths)} ({time.time()-t0:.0f}s)")
    return np.concatenate(feats)


# ----------------------------------------------------------------------------- grouping + stats
class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def bucket_pairs(values):
    buckets = defaultdict(list)
    for i, v in enumerate(values):
        buckets[v].append(i)
    for idxs in buckets.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                yield idxs[a], idxs[b]


def contaminated_fraction(group, gsize, test_idx):
    te = np.zeros(len(group), bool)
    te[test_idx] = True
    train_groups = set(group[~te])
    return sum(1 for i in test_idx if gsize[group[i]] > 1 and group[i] in train_groups) / len(test_idx)


def simulate_random_splits(m, group, gsize, test_frac, seeds):
    """Pooled datasets (APTOS): share of the test set that has a duplicate in train under the
    random stratified image-level split most papers use."""
    from sklearn.model_selection import train_test_split

    idx, y = np.arange(len(m)), m["label"].values
    fr = []
    for s in range(seeds):
        try:
            _, te = train_test_split(idx, test_size=test_frac, random_state=s, stratify=y)
        except ValueError:
            _, te = train_test_split(idx, test_size=test_frac, random_state=s)
        fr.append(contaminated_fraction(group, gsize, te))
    fr = np.array(fr)
    n_te = round(len(m) * test_frac)
    return dict(test_frac=test_frac, seeds=seeds, expected_test_images=n_te,
                mean_frac_contaminated=round(float(fr.mean()), 4),
                mean_images_contaminated=round(float(fr.mean()) * n_te, 1),
                range95=[round(float(np.percentile(fr, 2.5)), 4), round(float(np.percentile(fr, 97.5)), 4)])


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - hw), min(1.0, c + hw)


# ----------------------------------------------------------------------------- review material
def side_by_side(a, b, H=320):
    ia, ib = load_rgb(a), load_rgb(b)
    ia = ia.resize((max(1, int(ia.width * H / ia.height)), H))
    ib = ib.resize((max(1, int(ib.width * H / ib.height)), H))
    c = Image.new("RGB", (ia.width + ib.width + 10, H), "white")
    c.paste(ia, (0, 0))
    c.paste(ib, (ia.width + 10, 0))
    return c


def make_review(pairs, out: Path, n_per_tier: int, seed: int) -> int:
    rev = out / "review"
    rev.mkdir(exist_ok=True)
    cand = pairs[pairs.tier == "candidate"]
    strata = {
        "near_verified": pairs[pairs.tier == "near_verified"],
        "candidate_multi": cand[cand.n_generators >= 2],
        "candidate_single": cand[cand.n_generators == 1],
    }
    rows = []
    for stratum, df in strata.items():
        if df.empty:
            continue
        for _, r in df.sample(n=min(n_per_tier, len(df)), random_state=seed).iterrows():
            fn = f"pair_{len(rows)+1:04d}.jpg"
            side_by_side(r.path_i, r.path_j).save(rev / fn, quality=88)
            rows.append(dict(review_id=len(rows) + 1, stratum=stratum, stratum_population=len(df),
                             image=f"review/{fn}", path_i=r.path_i, path_j=r.path_j,
                             label_i=r.label_i, label_j=r.label_j, split_i=r.split_i, split_j=r.split_j,
                             ncc=r.ncc, phash_dist=r.phash_dist, dhash_dist=r.dhash_dist, cosine=r.cosine,
                             is_duplicate="", notes=""))
    pd.DataFrame(rows).to_csv(out / "review_sheet.csv", index=False)
    html = ["<html><head><meta charset='utf-8'><title>Duplicate review</title></head>"
            "<body style='font-family:sans-serif;max-width:1100px;margin:auto'>",
            "<h2>Near-duplicate review sample</h2><p>For each pair, fill <b>is_duplicate</b> in "
            "review_sheet.csv: 1 = same image (any re-save/resize/crop), 0 = different image. "
            "Use notes for 'same eye, different photo' etc.</p>"]
    for r in rows:
        html.append(f"<h4>#{r['review_id']} · {r['stratum']} · labels {r['label_i']} / {r['label_j']} · "
                    f"ncc {r['ncc']} · pHash {r['phash_dist']} · cos {r['cosine']}</h4>"
                    f"<img src='{r['image']}' style='max-width:100%'>")
    (out / "review.html").write_text("\n".join(html) + "</body></html>")
    return len(rows)


def load_confirmed(path, m):
    s = pd.read_csv(path)
    s = s[s.is_duplicate.astype(str).str.strip().isin(["1", "1.0"])]
    pos = {p: i for i, p in enumerate(m.path)}
    return [(pos[a], pos[b]) for a, b in zip(s.path_i, s.path_j) if a in pos and b in pos]


def save_histogram(best_ncc, thresh, out: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=150)
    ax.hist(best_ncc, bins=100, color="#4a6fa5")
    ax.axvline(thresh, color="#c0392b", ls="--", lw=1)
    ax.set_xlabel("Nearest-neighbour pixel correlation (NCC)")
    ax.set_ylabel("Images")
    ax.set_yscale("log")
    ax.set_title("Each image's most similar other image")
    fig.tight_layout()
    fig.savefig(out / "nn_similarity_hist.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- audit
def cmd_audit(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("[1/6] manifest")
    m = build_manifest(args)
    print(f"      {len(m)} images · splits {dict(Counter(m.split))} · labels {dict(Counter(m.label))}")

    print("[2/6] per-image features (md5, pixels, pHash, dHash, thumbnails)")
    cache = out / "cache_features.npz"
    key = hashlib.md5(("|".join(m.path) + f"{args.crop_thr},{args.hash_size},{args.ncc_side}").encode()).hexdigest()
    if cache.exists() and str(np.load(cache, allow_pickle=True)["key"]) == key:
        z = np.load(cache, allow_pickle=True)
        m = m.copy()
        for c in ("md5", "pixel_md5", "width", "height"):
            m[c] = z[c]
        ph, dh, V = [int(x, 16) for x in z["ph"]], [int(x, 16) for x in z["dh"]], z["V"]
        print("      loaded from cache")
    else:
        m, ph, dh, V = compute_features(m, args)
        np.savez(cache, key=key, md5=m.md5.values, pixel_md5=m.pixel_md5.values, width=m.width.values,
                 height=m.height.values, ph=np.array([format(x, "x") for x in ph]),
                 dh=np.array([format(x, "x") for x in dh]), V=V)
    nbits = args.hash_size ** 2
    PH, DH = to_words(ph, nbits), to_words(dh, nbits)

    print("[3/6] candidate pairs")
    gen: dict[tuple[int, int], set] = defaultdict(set)
    for i, j in bucket_pairs(m.md5):
        gen[(i, j)].add("md5")
    for i, j in bucket_pairs(m.pixel_md5):
        gen[(i, j)].add("pixel")
    for i, j in hamming_pairs(PH, args.phash_thresh):
        gen[(i, j)].add("phash")
    for i, j in hamming_pairs(DH, args.dhash_thresh):
        gen[(i, j)].add("dhash")
    ncc_hits, best_ncc, best_ncc_j = sim_pairs(V, args.ncc_cand)
    for i, j in ncc_hits:
        gen[(i, j)].add("ncc")
    E = None
    if not args.no_embed:
        print(f"[4/6] CNN embeddings ({args.arch}, weights={args.weights}, device={args.device})")
        ef = out / "embeddings.npy"
        if ef.exists() and np.load(ef).shape[0] == len(m):
            E = np.load(ef)
            print("      loaded from cache")
        else:
            E = compute_embeddings(m, args)
            np.save(ef, E)
        emb_hits, _, _ = sim_pairs(E, args.cos_thresh)
        for i, j in emb_hits:
            gen[(i, j)].add("embed")
    else:
        print("[4/6] embeddings skipped (--no-embed)")
    print(f"      {len(gen)} candidate pairs")

    print("[5/6] scoring pairs, building groups")
    recs = []
    for (i, j), g in gen.items():
        ncc = float(V[i] @ V[j])
        exact = bool(g & {"md5", "pixel"})
        phd = int(popcount(PH[i] ^ PH[j]).sum())
        verified = ncc >= args.ncc_thresh and phd <= args.phash_verify
        tier = "exact" if exact else ("near_verified" if verified else "candidate")
        recs.append(dict(
            i=i, j=j, tier=tier, generators="+".join(sorted(g)),
            n_generators=len(g & {"phash", "dhash", "embed", "ncc"}),
            md5_same="md5" in g, pixel_same="pixel" in g,
            ncc=round(ncc, 4),
            phash_dist=phd, dhash_dist=int(popcount(DH[i] ^ DH[j]).sum()),
            cosine=None if E is None else round(float(E[i] @ E[j]), 4),
        ))
    pairs = pd.DataFrame(recs)
    if pairs.empty:
        pairs = pd.DataFrame(columns=["i", "j", "tier", "generators", "n_generators", "md5_same",
                                      "pixel_same", "ncc", "phash_dist", "dhash_dist", "cosine"])
    for c in ("path", "split", "label"):
        pairs[f"{c}_i"] = m[c].values[pairs.i.astype(int)] if len(pairs) else []
        pairs[f"{c}_j"] = m[c].values[pairs.j.astype(int)] if len(pairs) else []
    pairs["label_conflict"] = pairs.label_i != pairs.label_j
    pairs["cross_split"] = pairs.split_i != pairs.split_j
    pairs = pairs.sort_values(["tier", "ncc"], ascending=[True, False])
    pairs.to_csv(out / "pairs.csv", index=False)

    dsu = DSU(len(m))
    for r in pairs.itertuples():
        if r.tier in ("exact", "near_verified"):
            dsu.union(int(r.i), int(r.j))
    n_confirmed = 0
    if args.confirmed:
        for i, j in load_confirmed(args.confirmed, m):
            dsu.union(i, j)
            n_confirmed += 1
    group = np.array([dsu.find(i) for i in range(len(m))])
    gsize = Counter(group)
    m["group_id"] = group
    m["group_size"] = [gsize[g] for g in group]
    n_lab = m.groupby("group_id")["label"].nunique()
    m["group_label_conflict"] = m.group_id.map(n_lab > 1)
    m["nn_ncc"] = best_ncc.round(4)
    m["nn_path"] = m.path.values[best_ncc_j]
    m.to_csv(out / "manifest_groups.csv", index=False)
    save_histogram(best_ncc, args.ncc_thresh, out)

    print("[6/6] summary")
    multi = {g: k for g, k in gsize.items() if k > 1}
    tiers = pairs.tier.value_counts().to_dict()
    S = {
        "tool_version": __version__,
        "n_images": len(m), "splits": dict(Counter(m.split)), "labels": dict(Counter(m.label)),
        "settings": {k: getattr(args, k) for k in ("hash_size", "phash_thresh", "dhash_thresh", "ncc_cand",
                                                   "ncc_thresh", "phash_verify", "ncc_side", "cos_thresh", "crop_thr")}
                    | {"embed": None if args.no_embed else f"{args.arch}/{args.weights}",
                       "confirmed_pairs_added": n_confirmed},
        "pairs": {
            "exact": int(tiers.get("exact", 0)),
            "near_verified": int(tiers.get("near_verified", 0)),
            "candidate_unverified": int(tiers.get("candidate", 0)),
            "exact_with_label_conflict": int(((pairs.tier == "exact") & pairs.label_conflict).sum()),
            "near_verified_with_label_conflict": int(((pairs.tier == "near_verified") & pairs.label_conflict).sum()),
        },
        "groups": {
            "n_dup_groups": len(multi),
            "images_in_dup_groups": int(sum(multi.values())),
            "frac_images_in_dup_groups": round(sum(multi.values()) / len(m), 4),
            "redundant_images": int(sum(k - 1 for k in multi.values())),
            "largest_group": int(max(multi.values())) if multi else 1,
            "size_hist": {int(k): int(v) for k, v in sorted(Counter(multi.values()).items())},
            "groups_with_label_conflict": int(sum(1 for g in multi if n_lab[g] > 1)),
            "images_in_label_conflict_groups": int(m.group_label_conflict.sum()),
            "dup_images_by_class": {str(k): int(v) for k, v in
                                    m[m.group_size > 1].label.value_counts().sort_index().items()},
        },
    }
    splits = set(m.split)
    if {"train", "test"} <= splits:
        tr_groups = set(m.loc[m.split == "train", "group_id"])
        te = m[m.split == "test"]
        cont = te[te.group_id.isin(tr_groups) & (te.group_size > 1)]
        tr_pix = set(m.loc[m.split == "train", "pixel_md5"])
        S["official_split"] = {
            "test_images": int(len(te)),
            "test_with_exact_train_copy": int(te.pixel_md5.isin(tr_pix).sum()),
            "test_with_train_duplicate_exact_or_verified": int(len(cont)),
            "frac_test_contaminated": round(len(cont) / max(1, len(te)), 4),
            "contaminated_by_class": {str(k): int(v) for k, v in cont.label.value_counts().sort_index().items()},
            "contaminated_with_conflicting_label": int(cont.group_label_conflict.sum()),
            "within_test_redundant": int(len(te) - te.group_id.nunique()),
        }
        if "val" in splits:
            va = m[m.split == "val"]
            S["official_split"]["val_with_train_duplicate"] = int(
                (va.group_id.isin(tr_groups) & (va.group_size > 1)).sum())
    else:
        S["random_split_expectation"] = {
            str(p): simulate_random_splits(m, group, gsize, p, args.sim_seeds) for p in args.test_fracs}

    S["runtime_s"] = round(time.time() - t_start, 1)
    (out / "summary.json").write_text(json.dumps(S, indent=2, default=str))
    print(json.dumps(S, indent=2, default=str))
    if not args.skip_review and len(pairs):
        print("      writing review sample ...", flush=True)
        n = make_review(pairs, out, args.review_per_tier, args.seed)
        S["review_pairs_written"] = n
        (out / "summary.json").write_text(json.dumps(S, indent=2, default=str))
    print(f"\nOutputs in {out}/")


def cmd_review_stats(args):
    s = pd.read_csv(args.sheet)
    s = s[s.is_duplicate.astype(str).str.strip().isin(["0", "1", "0.0", "1.0"])].copy()
    if s.empty:
        sys.exit("No rows with is_duplicate filled in (use 1 or 0).")
    s["is_duplicate"] = s.is_duplicate.astype(float).astype(int)
    res = {}
    for stratum, df in s.groupby("stratum"):
        fp = int((df.is_duplicate == 0).sum())
        p, lo, hi = wilson(fp, len(df))
        pop = int(df.stratum_population.iloc[0])
        res[stratum] = dict(reviewed=len(df), not_duplicates=fp, fp_rate=round(p, 3),
                            fp_ci95=[round(lo, 3), round(hi, 3)], population=pop,
                            est_true_duplicates=round(pop * (1 - p)),
                            est_true_duplicates_ci95=[round(pop * (1 - hi)), round(pop * (1 - lo))])
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit", help="run the duplicate/leakage audit")
    a.add_argument("--layout", choices=["split-class", "class", "csv"], required=True)
    a.add_argument("--root", help="dataset root for folder layouts")
    a.add_argument("--img-dir", help="image folder for --layout csv")
    a.add_argument("--csv", help="label CSV for --layout csv")
    a.add_argument("--id-col", default="id_code")
    a.add_argument("--label-col", default="diagnosis")
    a.add_argument("--split-col", default=None, help="optional split column in the CSV")
    a.add_argument("--ext", default=".png")
    a.add_argument("--out", required=True)
    a.add_argument("--crop-thr", type=int, default=15, help="dark-border crop threshold (-1 = off)")
    a.add_argument("--hash-size", type=int, default=16, help="16 -> 256-bit hashes")
    a.add_argument("--phash-thresh", type=int, default=32)
    a.add_argument("--dhash-thresh", type=int, default=32)
    a.add_argument("--ncc-side", type=int, default=64)
    a.add_argument("--ncc-cand", type=float, default=0.95, help="NCC level that makes a pair a candidate")
    a.add_argument("--ncc-thresh", type=float, default=0.98, help="NCC level needed to verify a near-duplicate")
    a.add_argument("--phash-verify", type=int, default=12,
                   help="pHash distance (of hash_size^2 bits) also needed to verify a near-duplicate")
    a.add_argument("--cos-thresh", type=float, default=0.95)
    a.add_argument("--no-embed", action="store_true")
    a.add_argument("--arch", default="resnet50")
    a.add_argument("--weights", default="DEFAULT", help="'DEFAULT' (ImageNet) or 'none' (testing only)")
    a.add_argument("--weights-file", default=None, help="local torchvision .pth (offline use)")
    a.add_argument("--workers", type=int, default=4, help="processes for per-image features")
    a.add_argument("--batch", type=int, default=32)
    a.add_argument("--device", default="auto", help="auto, cpu, cuda, mps")
    a.add_argument("--confirmed", help="filled review_sheet.csv; rows with is_duplicate=1 join groups")
    a.add_argument("--test-fracs", type=float, nargs="+", default=[0.2, 0.3])
    a.add_argument("--sim-seeds", type=int, default=200)
    a.add_argument("--review-per-tier", type=int, default=50)
    a.add_argument("--skip-review", action="store_true", help="don't render review images")
    a.add_argument("--seed", type=int, default=0)
    a.set_defaults(func=cmd_audit)

    r = sub.add_parser("review-stats", help="false-positive rates from a filled review sheet")
    r.add_argument("--sheet", required=True)
    r.set_defaults(func=cmd_review_stats)

    args = ap.parse_args()
    if getattr(args, "device", None) == "auto" and not getattr(args, "no_embed", True):
        import torch
        args.device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    args.func(args)


if __name__ == "__main__":
    main()
