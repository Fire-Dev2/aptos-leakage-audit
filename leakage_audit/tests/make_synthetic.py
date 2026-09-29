"""Build two synthetic datasets with planted duplicates and a ground-truth file.

A) APTOS-like: pooled PNGs + train.csv (id_code, diagnosis)
B) Brain-MRI-like: Training/<class>/ and Testing/<class>/ folders
"""
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

rng = random.Random(7)
nprng = np.random.default_rng(7)


def fundus(seed, size=512):
    r = random.Random(seed)
    im = Image.new("RGB", (size, size), (0, 0, 0))
    d = ImageDraw.Draw(im)
    base = (r.randint(150, 220), r.randint(60, 110), r.randint(20, 50))
    d.ellipse([20, 20, size - 20, size - 20], fill=base)
    # optic disc
    cx, cy = r.randint(120, 390), r.randint(150, 360)
    d.ellipse([cx - 40, cy - 40, cx + 40, cy + 40], fill=(250, 220, 150))
    # vessels
    for _ in range(r.randint(8, 14)):
        pts = [(cx, cy)]
        for _ in range(6):
            pts.append((pts[-1][0] + r.randint(-70, 70), pts[-1][1] + r.randint(-70, 70)))
        d.line(pts, fill=(120, 20, 20), width=r.randint(2, 6))
    # lesions
    for _ in range(r.randint(0, 40)):
        x, y = r.randint(60, size - 60), r.randint(60, size - 60)
        s = r.randint(2, 9)
        d.ellipse([x - s, y - s, x + s, y + s], fill=r.choice([(255, 255, 180), (90, 0, 0)]))
    arr = np.asarray(im).astype(np.int16)
    arr = arr + nprng.integers(-6, 7, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))


def mri(seed, size=256):
    r = random.Random(seed)
    im = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(im)
    d.ellipse([30, 20, size - 30, size - 20], fill=r.randint(70, 110))
    for _ in range(r.randint(10, 20)):
        x, y = r.randint(50, size - 50), r.randint(40, size - 40)
        s = r.randint(4, 25)
        d.ellipse([x - s, y - s, x + s, y + s], fill=r.randint(40, 200))
    return im.filter(ImageFilter.GaussianBlur(1.5)).convert("RGB")


def jpeg_roundtrip(im, q=80):
    b = io.BytesIO()
    im.save(b, "JPEG", quality=q)
    b.seek(0)
    return Image.open(b).convert("RGB")


def build_aptos(root: Path):
    img = root / "train_images"
    img.mkdir(parents=True, exist_ok=True)
    rows, truth = [], []
    base_ids = []
    for k in range(150):
        iid = f"base{k:03d}"
        fundus(1000 + k).save(img / f"{iid}.png")
        rows.append((iid, k % 5))
        base_ids.append((iid, k % 5, 1000 + k))

    def plant(kind, src, im, label):
        iid = f"{kind}_{src}"
        if kind == "exact":
            (img / f"{iid}.png").write_bytes((img / f"{src}.png").read_bytes())
        else:
            im.save(img / f"{iid}.png")
        rows.append((iid, label))
        truth.append(dict(a=src, b=iid, kind=kind, label_conflict=label != dict(rows)[src]))

    picks = rng.sample(base_ids, 30)
    for n, (iid, lab, seed) in enumerate(picks):
        orig = Image.open(img / f"{iid}.png").convert("RGB")
        if n < 5:
            plant("exact", iid, None, lab)
        elif n < 10:
            plant("jpeg", iid, jpeg_roundtrip(orig), lab)
        elif n < 15:
            plant("resize", iid, orig.resize((358, 358), Image.BILINEAR), lab)
        elif n < 20:
            plant("bright", iid, ImageEnhance.Brightness(orig).enhance(1.08), lab)
        elif n < 23:
            plant("crop", iid, orig.crop((15, 15, 497, 497)), lab)
        elif n < 26:  # exact copy, different grade -> label noise
            plant("exact", iid, None, (lab + 2) % 5)
        # n 26..29 untouched (controls)
    # triple group: one image copied twice
    src = picks[27][0]
    for t in ("tripA", "tripB"):
        iid = f"{t}_{src}"
        (img / f"{iid}.png").write_bytes((img / f"{src}.png").read_bytes())
        rows.append((iid, picks[27][1]))
        truth.append(dict(a=src, b=iid, kind="exact", label_conflict=False))
    truth.append(dict(a=f"tripA_{src}", b=f"tripB_{src}", kind="exact", label_conflict=False))
    pd.DataFrame(rows, columns=["id_code", "diagnosis"]).to_csv(root / "train.csv", index=False)
    json.dump(truth, open(root / "truth.json", "w"), indent=1)
    print(f"APTOS-like: {len(rows)} images, {len(truth)} true duplicate pairs")


def build_brain(root: Path):
    classes = ["glioma", "meningioma", "notumor", "pituitary"]
    truth = []
    k = 0
    made = {}
    for split, n in (("Training", 40), ("Testing", 10)):
        for c in classes:
            d = root / split / c
            d.mkdir(parents=True, exist_ok=True)
            for _ in range(n):
                p = d / f"img{k:04d}.jpg"
                mri(5000 + k).save(p, quality=92)
                made[p] = c
                k += 1
    train_imgs = [p for p in made if "Training" in str(p)]
    # 8 test images that are re-saved copies of train images (cross-split leak)
    for n, src in enumerate(rng.sample(train_imgs, 8)):
        c = made[src]
        dst = root / "Testing" / c / f"leak{n}.jpg"
        Image.open(src).convert("RGB").save(dst, quality=85)
        truth.append(dict(a=str(src), b=str(dst), kind="cross_split_jpeg"))
    # 4 within-train exact copies
    for n, src in enumerate(rng.sample(train_imgs, 4)):
        dst = src.parent / f"dup{n}.jpg"
        dst.write_bytes(src.read_bytes())
        truth.append(dict(a=str(src), b=str(dst), kind="within_train_exact"))
    json.dump(truth, open(root / "truth.json", "w"), indent=1)
    print(f"Brain-like: {len(list(root.rglob('*.jpg')))} images, {len(truth)} true duplicate pairs "
          f"(8 cross-split test leaks)")


if __name__ == "__main__":
    out = Path(sys.argv[1])
    build_aptos(out / "aptos_like")
    build_brain(out / "brain_like")
