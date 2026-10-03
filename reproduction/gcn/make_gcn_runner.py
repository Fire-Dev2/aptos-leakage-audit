import base64, gzip, pathlib
K = pathlib.Path("/mnt/user-data/outputs/phase5_kit/gcn")
files = {"patch_gcn.py": K / "patch_gcn.py", "make_gcn_split.py": K / "make_gcn_split.py",
         "splits/g_train_ids.csv": K / "splits/g_train_ids.csv", "splits/g_val_ids.csv": K / "splits/g_val_ids.csv",
         "splits/g_test_ids.csv": K / "splits/g_test_ids.csv"}
payload = {k: base64.b64encode(gzip.compress(v.read_bytes())).decode() for k, v in files.items()}
TEMPLATE = r'''#!/usr/bin/env python
"""
run_gcn.py - Phase 5, paper 3: graph-enhanced DR classifier (PLOS Comput Biol 2025) on APTOS 2019.

  One-time setup (login node):  python run_gcn.py --setup      (~10-20 min: packages, ImageNet weights, image caches)
  Train (inside the Slurm job):  python run_gcn.py
  Re-running is safe: finished runs are skipped, unfinished runs restart.

Conditions (backbone MobileViT-S, the paper's best model on APTOS; everything else = released defaults):
  A  "as described": minority grades oversampled to 1805 images each by duplicating originals with random
     rotation / flips / blur / brightness-contrast, THEN a stratified 70/15/15 train/val/test split (new split per seed).
  B  "fixed": group-aware 70/15/15 split of the 3,662 originals (no duplicate group crosses splits), THEN the
     training set alone is oversampled the same way; validation and test are untouched originals.
Seeds 1234, 1, 2 for each. The paper's balancing parameters were not released; ours are documented in this file.
"""
import os, sys, glob, time, json, subprocess, shutil, pathlib, base64, gzip, tarfile
HERE = pathlib.Path.cwd(); HOME = HERE / "gcn"; HOME.mkdir(exist_ok=True)
SETUP = "--setup" in sys.argv
BACKBONE, OUTDIM = os.environ.get("BACKBONE", "mobilevit_s"), os.environ.get("OUTDIM", "640")
JOBS = [("A", 1234), ("B", 1234), ("A", 1), ("B", 1), ("A", 2), ("B", 2)]
REPO_URL, COMMIT = "https://github.com/mfar201/diabetic_retinopathy_classification_gcn.git", "fb123aba9d0d4e5bce8f9bcf3950895d758a1e29"
def sh(c, **k):
    print("+", c, flush=True); return subprocess.run(c, shell=True, **k)
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

KIT = HOME / "kit"
PAYLOAD = __PAYLOAD__
for name, b in PAYLOAD.items():
    p = KIT / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(gzip.decompress(base64.b64decode(b)))

def find_data():
    if os.path.exists(f"{HERE}/aptos/train.csv") and len(os.listdir(f"{HERE}/aptos/train_images")) >= 3662: return f"{HERE}/aptos"
    for p in glob.glob(f"{HERE}/**/train.csv", recursive=True):
        d = os.path.dirname(p)
        if os.path.isdir(f"{d}/train_images") and len(os.listdir(f"{d}/train_images")) >= 3662: return d
    return None
DATA = os.environ.get("APTOS_DIR") or find_data(); assert DATA, "APTOS not found under this folder (run setup.sh first)"
log("APTOS:", DATA)

REPO = HOME / "repo"
if not (REPO / "train.py").exists():
    sh(f"git clone -q {REPO_URL} {REPO} && cd {REPO} && git checkout -q {COMMIT}", check=True)
sh(f"python {KIT}/patch_gcn.py {REPO}", check=True)

if SETUP:
    sh("pip -q install torch_geometric==2.6.1 seaborn tqdm 2>&1 | tail -2")
    sh(f"cd {REPO} && python -c \"import timm; timm.create_model('{BACKBONE}', pretrained=True); print('weights OK: {BACKBONE}')\"", check=True)
    sh("python -c \"import torch_geometric; from torch_geometric.nn import GCNConv; print('torch_geometric', torch_geometric.__version__)\"", check=True)

import numpy as np, pandas as pd
from PIL import Image, ImageFilter, ImageEnhance

# ---------- image caches (exactly what the released loader sees: Resize((224,224)) of the full image) ----------
lab = pd.read_csv(f"{DATA}/train.csv").set_index("id_code")["diagnosis"].astype(int)
C = HOME / "cache"; (C / "orig").mkdir(parents=True, exist_ok=True)
AUG_CODE = r"""
import sys, os, json, numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms as T
from multiprocessing import Pool
src, jobs_file = sys.argv[1], sys.argv[2]
jobs = json.load(open(jobs_file)); R = T.Resize((224, 224))
def augment(im, seed):
    # 'random rotations, flips, blur operations, and brightness/contrast adjustments' (paper); parameters are ours
    g = np.random.RandomState(seed)
    if g.rand() < 0.5: im = im.transpose(Image.FLIP_LEFT_RIGHT)
    if g.rand() < 0.5: im = im.transpose(Image.FLIP_TOP_BOTTOM)
    if g.rand() < 0.5: im = im.rotate(g.uniform(-30, 30), resample=Image.BILINEAR)
    if g.rand() < 0.3: im = im.filter(ImageFilter.GaussianBlur(radius=g.uniform(1.0, 3.0)))
    if g.rand() < 0.5: im = ImageEnhance.Brightness(im).enhance(g.uniform(0.8, 1.2))
    if g.rand() < 0.5: im = ImageEnhance.Contrast(im).enhance(g.uniform(0.8, 1.2))
    return im
def one(j):
    out = j["out"]
    if os.path.exists(out): return
    im = Image.open(os.path.join(src, j["src"] + ".png")).convert("RGB")
    if j.get("seed") is not None: im = augment(im, j["seed"])
    R(im).save(out + ".tmp.png"); os.replace(out + ".tmp.png", out)
with Pool(min(os.cpu_count(), 16)) as p: p.map(one, jobs, chunksize=8)
"""
(KIT / "make_cache.py").write_text(AUG_CODE)

def oversample_jobs(ids, tag, seed0):
    """copies needed to bring every grade up to the largest grade; sources cycle through a shuffled list"""
    ids = list(ids); y = lab.loc[ids]; target = y.value_counts().max(); g = np.random.RandomState(seed0); jobs = []
    for c in sorted(y.unique()):
        pool = sorted(y[y == c].index); need = target - len(pool); k = 0
        order = list(g.permutation(pool))
        while need > 0:
            s = order[k % len(order)]; k += 1; need -= 1
            jobs.append(dict(src=s, seed=int(g.randint(2**31 - 1)), out=str(C / tag / f"{s}__{tag}{k:05d}.png"), label=int(c)))
    return jobs, int(target)

jobs = [dict(src=i, seed=None, out=str(C / "orig" / f"{i}.png"), label=int(lab[i])) for i in lab.index]
gtr = pd.read_csv(KIT / "splits/g_train_ids.csv").id
jA, tA = oversample_jobs(lab.index, "augA", 0)        # condition A: balance the whole dataset first
jB, tB = oversample_jobs(gtr, "augB", 0)              # condition B: balance the training split only
for t in ["augA", "augB"]: (C / t).mkdir(exist_ok=True)
todo = [j for j in jobs + jA + jB if not os.path.exists(j["out"])]
if todo:
    if not SETUP: log("building image caches for", len(todo), "files (normally done in --setup)")
    json.dump(todo, open(KIT / "cache_jobs.json", "w"))
    sh(f"python {KIT}/make_cache.py {DATA}/train_images {KIT}/cache_jobs.json", check=True)
log(f"cache: {len(glob.glob(str(C / 'orig' / '*.png')))} originals, {len(glob.glob(str(C / 'augA' / '*.png')))} A-copies "
    f"(balanced to {tA}/grade), {len(glob.glob(str(C / 'augB' / '*.png')))} B-copies (train balanced to {tB}/grade)")

# ---------- ImageFolder trees (symlinks) ----------
from sklearn.model_selection import train_test_split
def build_tree(root, parts):
    if (root / "READY").exists(): return
    shutil.rmtree(root, ignore_errors=True)
    for split, items in parts.items():
        for c in range(5): (root / split / str(c)).mkdir(parents=True, exist_ok=True)
        for out, c in items: os.symlink(out, root / split / str(c) / os.path.basename(out))
    pd.DataFrame([(s, os.path.basename(o), c) for s, it in parts.items() for o, c in it], columns=["split", "file", "label"]).to_csv(root / "files.csv", index=False)
    (root / "READY").write_text("ok")
def no_single_last_batch(tr, va, bs=32):
    # the released training loop keeps a last partial batch; a batch of 1 crashes BatchNorm. Move one image to val if needed.
    while len(tr) % bs == 1: va.append(tr.pop())
    return tr, va
TREES = {}
allA = [(j["out"], j["label"]) for j in jobs + jA]
for s in sorted({s for c, s in JOBS if c == "A"}):
    tr, rest = train_test_split(allA, test_size=0.30, stratify=[c for _, c in allA], random_state=s)
    va, te = train_test_split(rest, test_size=0.50, stratify=[c for _, c in rest], random_state=s)
    tr, va = no_single_last_batch(list(tr), list(va))
    TREES[("A", s)] = HOME / "trees" / f"A_split{s}"; build_tree(TREES[("A", s)], dict(train=tr, val=va, test=te))
orig = {i: (str(C / "orig" / f"{i}.png"), int(lab[i])) for i in lab.index}
trB = [orig[i] for i in gtr] + [(j["out"], j["label"]) for j in jB]
vaB = [orig[i] for i in pd.read_csv(KIT / "splits/g_val_ids.csv").id]; teB = [orig[i] for i in pd.read_csv(KIT / "splits/g_test_ids.csv").id]
trB, vaB = no_single_last_batch(trB, vaB)
TB = HOME / "trees" / "B"; build_tree(TB, dict(train=trB, val=vaB, test=teB))
for s in sorted({s for c, s in JOBS if c == "B"}): TREES[("B", s)] = TB
for k, t in sorted(TREES.items()):
    f = pd.read_csv(t / "files.csv"); log(k, f.split.value_counts().to_dict())
if SETUP:
    log("SETUP OK - now run:  sbatch gcn.sbatch"); sys.exit(0)

# ---------- runs ----------
import torch
NGPU = torch.cuda.device_count() or (1 if os.environ.get("ALLOW_CPU") else 0); assert NGPU >= 1, "no GPU"
PER_GPU = int(os.environ.get("PER_GPU", 3))
log(f"GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]} -> {PER_GPU} runs per GPU; backbone {BACKBONE}")
RUNS = [dict(name=f"{c}_seed{s}", cond=c, seed=s, dir=HOME / "runs" / f"{c}_seed{s}", tree=TREES[(c, s)]) for c, s in JOBS]
(HOME / "runs").mkdir(exist_ok=True)
done = lambda r: (r["dir"] / "LEAK_DONE").exists() and (r["dir"] / "true_bestval_bs1.npz").exists()
nep = lambda r: len(glob.glob(f"{r['dir']}/evals/e*_test.npz"))
queue = [r for r in RUNS if not done(r)]; slots = {g: 0 for g in range(NGPU)}; running = []; fails = {}
t0 = time.time(); last = 0
while queue or running:
    for r, p, g in list(running):
        if p.poll() is not None:
            running.remove((r, p, g)); slots[g] -= 1
            if done(r): log(r["name"], "FINISHED")
            else:
                fails[r["name"]] = fails.get(r["name"], 0) + 1
                log(r["name"], f"stopped early (exit {p.returncode}):"); sh(f"grep -v -i warn {r['dir']}/run.log | tail -15")
                if fails[r["name"]] < 3: queue.append(r)
    while queue and min(slots.values()) < PER_GPU:
        g = min(slots, key=slots.get); r = queue.pop(0)
        shutil.rmtree(r["dir"], ignore_errors=True); r["dir"].mkdir(parents=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONPATH=":".join([str(REPO)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])), MPLBACKEND="Agg",
                   LEAK_BACKBONE=BACKBONE, LEAK_OUTDIM=OUTDIM, LEAK_SEED=str(r["seed"]), LEAK_OUT=str(r["dir"]),
                   LEAK_TRAIN=str(r["tree"] / "train"), LEAK_VAL=str(r["tree"] / "val"), LEAK_TEST=str(r["tree"] / "test"))
        p = subprocess.Popen(f"python {REPO}/train.py", shell=True, cwd=r["dir"], env=env,
                             stdout=open(r["dir"] / "run.log", "w"), stderr=subprocess.STDOUT)
        running.append((r, p, g)); slots[g] += 1; log("started", r["name"], "on GPU", g)
    time.sleep(30)
    if time.time() - last > 1200:
        last = time.time(); log(f"{(time.time()-t0)/3600:.1f} h |", " | ".join(f"{r['name']}: {nep(r)}/50" for r in RUNS))

for f in glob.glob(f"{HOME}/runs/**/*.pth", recursive=True): os.remove(f)
for k, t in TREES.items(): shutil.copy(t / "files.csv", HOME / "runs" / f"tree_{k[0]}_{k[1]}_files.csv")
out = HERE / "results_gcn.tgz"
with tarfile.open(str(out) + ".tmp", "w:gz") as tf:
    tf.add(HOME / "runs", arcname="runs"); tf.add(KIT / "splits", arcname="splits")
os.replace(str(out) + ".tmp", out)
log(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")
bad = [r["name"] for r in RUNS if not done(r)]
print("\n" + "=" * 70); print("ALL DONE. Download results_gcn.tgz and send it to Claude." if not bad else
      f"Some runs did not finish: {bad}. Download results_gcn.tgz (has the logs) and send it to Claude."); print("=" * 70)
'''
out = K / "run_gcn.py"; out.write_text(TEMPLATE.replace("__PAYLOAD__", repr(payload))); print("wrote", out, out.stat().st_size)
(K / "gcn.sbatch").write_text("""#!/bin/bash
#SBATCH --job-name=gcn
#SBATCH --account=YOUR_PROJECT
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=24
#SBATCH --time=12:00:00
#SBATCH --output=gcn_%j.log
# GCN (PLOS Comput Biol 2025) reruns: 6 runs, 3 per A100. If the job ends early, `sbatch gcn.sbatch` again.
cd "$HOME/leakage"
module load miniconda3/24.1.2-py310
source activate diffmic
nvidia-smi
python run_gcn.py
""")
