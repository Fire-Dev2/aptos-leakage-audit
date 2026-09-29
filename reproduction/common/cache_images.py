"""
cache_images.py — speed-up with identical results: pre-apply DiffMIC's deterministic first two transforms.

DiffMIC's APTOS loader opens each full-size PNG (up to 3216x2136) every epoch, then applies
CropCenterSquare() -> Resize((224,224)) before any random augmentation. We apply exactly those two
transforms once (using DiffMIC's own classes) and save lossless 224x224 PNGs. When the loader later
re-applies them to a 224x224 image, both are no-ops (a full-size centre crop and a same-size resize
return the pixels unchanged), so every tensor the model sees is identical. verify() checks this.

Usage: python cache_images.py DIFFMIC_DIR SRC_DIR DST_DIR PKL [PKL ...]
       writes DST_DIR/<id>.png and a copy of each PKL with suffix _c224.pkl pointing at the cache
"""
import os, sys, pickle, random
from multiprocessing import Pool

DIFF, SRC, DST, PKLS = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
sys.path.insert(0, DIFF)
from PIL import Image
import numpy as np, torch
from torchvision import transforms
from dataloader import transforms as trans
from dataloader.loading import APTOSDataset

PREFIX = transforms.Compose([trans.CropCenterSquare(), transforms.Resize((224, 224))])

def one(name):
    out = os.path.join(DST, name)
    if not os.path.exists(out):
        PREFIX(Image.open(os.path.join(SRC, name)).convert("RGB")).save(out + ".tmp.png")
        os.replace(out + ".tmp.png", out)
    return name

def verify(pkl_orig, pkl_new, n=24):
    for train in (False, True):
        a, b = APTOSDataset(pkl_orig, train=train), APTOSDataset(pkl_new, train=train)
        for i in random.Random(0).sample(range(len(a)), min(n, len(a))):
            outs = []
            for ds in (a, b):
                random.seed(i); np.random.seed(i); torch.manual_seed(i)
                outs.append(ds[i])
            assert outs[0][1] == outs[1][1] and torch.equal(outs[0][0], outs[1][0]), f"mismatch train={train} idx={i}"
    print("verified identical tensors (eval and train transforms):", os.path.basename(pkl_orig))

if __name__ == "__main__":
    os.makedirs(DST, exist_ok=True)
    names = sorted({os.path.basename(d["img_root"]) for p in PKLS for d in pickle.load(open(p, "rb"))})
    with Pool(max(1, os.cpu_count())) as pool:
        for k, _ in enumerate(pool.imap_unordered(one, names, chunksize=8)):
            if k % 500 == 0: print("cached", k, "/", len(names), flush=True)
    for p in PKLS:
        new = p.replace(".pkl", "_c224.pkl")
        pickle.dump([dict(d, img_root=os.path.join(DST, os.path.basename(d["img_root"]))) for d in pickle.load(open(p, "rb"))],
                    open(new, "wb"))
        verify(p, new)
