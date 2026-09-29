import base64, gzip, pathlib
K = pathlib.Path("/mnt/user-data/outputs/phase5_kit")
files = {"patch_nnmb.py": K / "nnmb/patch_nnmb.py",
         "splits/r0_train_ids.csv": K / "splits/r0_train_ids.csv", "splits/r0_test_ids.csv": K / "splits/r0_test_ids.csv",
         "splits/r1_train_ids.csv": K / "splits/r1_train_ids.csv", "splits/r1_val_ids.csv": K / "splits/r1_val_ids.csv",
         "splits/test_meta_nolabels.csv": K / "splits/test_meta_nolabels.csv"}
payload = {k: base64.b64encode(gzip.compress(v.read_bytes())).decode() for k, v in files.items()}
TEMPLATE = r'''#!/usr/bin/env python
"""
run_nnmb.py - Phase 5, paper 2: nnMobileNet (CVPRW 2024) on APTOS 2019, released test-set checkpoint rule vs validation selection.

  One-time setup (login node):  python run_nnmb.py --setup
  Train (inside the Slurm job):  python run_nnmb.py
  Re-running either is safe: finished runs are skipped, unfinished runs resume from their last checkpoint (every 10 epochs).

Runs: R0 (released-style: train 2564 / test 1098, best epoch picked on TEST) seeds 1234, 1, 2
      R1 (train 2212 / val 352 / test 1098, same code; validation predictions also saved) seeds 1234, 1, 2
Same APTOS split as the DiffMIC reruns (nnMobileNet did not release its APTOS split files or its cropped images).
Settings: the repository's documented single-GPU command (--input_size 224 --drop_path 0.2) with the paper's AdamP optimiser;
everything else at the released defaults (1000 epochs, batch 32, lr 1e-3, cosine, 20 warm-up epochs, wd 0.05, mixup/cutmix, EMA).
"""
import os, sys, glob, time, subprocess, shutil, pathlib, base64, gzip
HERE = pathlib.Path.cwd(); HOME = HERE / "nnmb"; HOME.mkdir(exist_ok=True)
SETUP = "--setup" in sys.argv
JOBS = [("R0", 1234), ("R1", 1234), ("R0", 1), ("R1", 1), ("R0", 2), ("R1", 2)]
LAST = 999
def sh(c, **k):
    print("+", c, flush=True); return subprocess.run(c, shell=True, **k)
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

KIT = HOME / "kit"
PAYLOAD = __PAYLOAD__
for name, b in PAYLOAD.items():
    p = KIT / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(gzip.decompress(base64.b64decode(b)))

def find_data():
    for p in glob.glob(f"{HERE}/**/train.csv", recursive=True):
        d = os.path.dirname(p)
        if os.path.isdir(f"{d}/train_images") and len(os.listdir(f"{d}/train_images")) >= 3662: return d
    return None
DATA = find_data(); assert DATA, "APTOS not found under this folder (run osc_setup.sh first)"
log("APTOS:", DATA)

REPO = HOME / "NN-MOBILENET"
if not (REPO / "main.py").exists():
    sh(f"git clone -q https://github.com/Retinal-Research/NN-MOBILENET.git {REPO} && cd {REPO} && git checkout -q 920acd3", check=True)
sh(f"python {KIT}/patch_nnmb.py {REPO}", check=True)

import numpy as np, pandas as pd
WEIGHTS = HOME / "rexnet_3.0.pth"
if SETUP:
    sh("pip -q install tensorboardX gdown timm==0.9.16 2>/dev/null || true")
    # ImageNet ReXNet-3.0 weights: the file linked in the nnMobileNet README, else the official ReXNet release
    for fid in ["1COB7eKY4VAS9QOnpBLTg4wxW27U3RFSy", "1iXAsr8gs3pRz0QyHKomdj5SGVzPWbIs2"]:
        if WEIGHTS.exists() and WEIGHTS.stat().st_size > 10_000_000: break
        sh(f"gdown -q 'https://drive.google.com/uc?id={fid}' -O {WEIGHTS}")
    assert WEIGHTS.exists() and WEIGHTS.stat().st_size > 10_000_000, "could not download rexnet_3.0.pth"
    import torch; sys.path.insert(0, str(REPO))
    from model.resnext import ReXNetV1
    own = ReXNetV1(width_mult=3.0, classes=5).state_dict(); sd = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd); ok = [k for k in sd if k in own and own[k].shape == sd[k].shape]
    nf = sum(1 for k in own if k.startswith("features"))
    log(f"pretrained check: {len(ok)} of {nf} backbone tensors match")

# ---------- data: CSVs + fundus-cropped images (the released 'crop' step was not published) ----------
lab = pd.read_csv(f"{DATA}/train.csv").set_index("id_code")["diagnosis"]
CSV = HOME / "csv"; CSV.mkdir(exist_ok=True)
for k in ["r0_train", "r0_test", "r1_train", "r1_val"]:
    ids = pd.read_csv(KIT / "splits" / f"{k}_ids.csv").id
    pd.DataFrame({"id": ids, "label": ids.map(lab).astype(int)}).to_csv(CSV / f"{k}.csv", index=False)
meta = pd.read_csv(KIT / "splits/test_meta_nolabels.csv"); meta["label"] = meta.id.map(lab); meta.to_csv(HOME / "test_meta.csv", index=False)
IMG = HOME / "crop512"
if len(glob.glob(f"{IMG}/*.png")) < 3662:
    IMG.mkdir(exist_ok=True)
    code = r"""
import sys, os, numpy as np
from PIL import Image
from multiprocessing import Pool
src, dst = sys.argv[1], sys.argv[2]
def one(f):
    out = os.path.join(dst, f)
    if os.path.exists(out): return
    im = Image.open(os.path.join(src, f)).convert("RGB"); g = np.asarray(im.convert("L"))
    ys, xs = np.where(g > 15)
    if len(xs): im = im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))   # tight box around the fundus
    s = 512 / max(im.size); im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.BICUBIC)
    im.save(out + ".tmp.png"); os.replace(out + ".tmp.png", out)
with Pool(os.cpu_count()) as p: p.map(one, sorted(os.listdir(src)), chunksize=16)
"""
    (KIT / "crop_images.py").write_text(code)
    sh(f"python {KIT}/crop_images.py {DATA}/train_images {IMG}", check=True)
log("cropped images:", len(glob.glob(f"{IMG}/*.png")))
if SETUP:
    log("SETUP OK - now run:  sbatch nnmb.sbatch"); sys.exit(0)

# ---------- runs ----------
import torch
NGPU = torch.cuda.device_count(); assert NGPU >= 1, "no GPU"
PER_GPU = int(os.environ.get("PER_GPU", 3))
log(f"GPUs: {[torch.cuda.get_device_name(i) for i in range(NGPU)]} -> {PER_GPU} runs per GPU")
RUNS = [dict(name=f"{c}_seed{s}", cond=c, seed=s, dir=HOME / "runs" / f"{c}_seed{s}") for c, s in JOBS]
(HOME / "runs").mkdir(exist_ok=True)
done = lambda r: (r["dir"] / "evals" / f"e{LAST:04d}_test_model.npz").exists()
has_ckpt = lambda r: any(p.split("-")[-1].split(".")[0].isdigit() for p in glob.glob(f"{r['dir']}/checkpoint-*.pth"))
nevals = lambda r: len(glob.glob(f"{r['dir']}/evals/e*_test_model.npz"))
for r in RUNS:
    if r["dir"].exists() and not done(r) and not has_ckpt(r):
        log(r["name"], "incomplete with no checkpoint -> restart"); shutil.rmtree(r["dir"])
queue = [r for r in RUNS if not done(r)]; slots = {g: 0 for g in range(NGPU)}; running = []; fails = {}
clean_env = {k: v for k, v in os.environ.items() if not (k.startswith("SLURM_") or k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))}
t0 = time.time(); last = 0
while queue or running:
    for r, p, g in list(running):
        if p.poll() is not None:
            running.remove((r, p, g)); slots[g] -= 1
            if done(r): log(r["name"], "FINISHED")
            else:
                fails[r["name"]] = fails.get(r["name"], 0) + 1
                log(r["name"], f"stopped early (exit {p.returncode}):"); sh(f"grep -v -i warn {r['dir']}.log | tail -15")
                if fails[r["name"]] < 3: queue.append(r)
    while queue and min(slots.values()) < PER_GPU:
        g = min(slots, key=slots.get); r = queue.pop(0); r["dir"].mkdir(parents=True, exist_ok=True)
        env = dict(clean_env, CUDA_VISIBLE_DEVICES=str(g), LEAK_IMG_DIR=str(IMG), LEAK_PRETRAIN=str(WEIGHTS),
                   LEAK_TRAIN_CSV=str(CSV / f"{r['cond'].lower()}_train.csv"), LEAK_TEST_CSV=str(CSV / "r0_test.csv"),
                   LEAK_VAL_CSV=str(CSV / "r1_val.csv") if r["cond"] == "R1" else "")
        cmd = (f"python main.py --data_set apots --nb_classes 5 --input_size 224 --drop_path 0.2 --opt adamp "
               f"--epochs 1000 --batch_size 32 --num_workers 4 --save_ckpt_freq 10 --seed {r['seed']} "
               f"--output_dir {r['dir']} --log_dir {r['dir']}/log")
        p = subprocess.Popen(cmd, shell=True, cwd=REPO, env=env, stdout=open(f"{r['dir']}.log", "a"), stderr=subprocess.STDOUT)
        running.append((r, p, g)); slots[g] += 1; log("started", r["name"], "on GPU", g, "(resuming)" if has_ckpt(r) else "(new)")
    time.sleep(60)
    if time.time() - last > 1800:
        last = time.time(); log(f"{(time.time()-t0)/3600:.1f} h |", " | ".join(f"{r['name']}: {nevals(r)}/1000" for r in RUNS))

for f in glob.glob(f"{HOME}/runs/**/*.pth", recursive=True): os.remove(f)
sh(f"cd {HOME} && tar czf {HERE}/results_nnmb.tgz --exclude='*.pth' runs test_meta.csv csv")
bad = [r["name"] for r in RUNS if not done(r)]
print("\n" + "=" * 70); print("ALL DONE. Download results_nnmb.tgz and send it to Claude." if not bad else
      f"Some runs did not finish: {bad}. Download results_nnmb.tgz (has the logs) and send it to Claude."); print("=" * 70)
'''
out = K / "nnmb/run_nnmb.py"; out.write_text(TEMPLATE.replace("__PAYLOAD__", repr(payload))); print("wrote", out, out.stat().st_size)
(K / "nnmb/nnmb.sbatch").write_text("""#!/bin/bash
#SBATCH --job-name=nnmb
#SBATCH --account=YOUR_OSC_PROJECT
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=24
#SBATCH --time=36:00:00
#SBATCH --output=nnmb_%j.log
# nnMobileNet reruns: 6 runs, 3 per A100. If the job ends early, `sbatch nnmb.sbatch` again - it resumes.
cd "$HOME/leakage"
module load miniconda3/24.1.2-py310
source activate diffmic
nvidia-smi
python run_nnmb.py
""")
