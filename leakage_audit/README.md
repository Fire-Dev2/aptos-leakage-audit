# leakage-audit

Duplicate and train/test-leakage audit for medical image classification datasets (Phase 2).

## Install (Mac, Apple Silicon)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Get the data
- **APTOS 2019**: kaggle.com/competitions/aptos2019-blindness-detection → accept rules → download.
  You need `train.csv` and `train_images/` (3,662 PNGs). The competition test set has no public labels; ignore it.
- **Brain tumor MRI** (calibration check): kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset.
  Get the **original V1** release if you can (the current release is already deduplicated).

## Run
```bash
# APTOS (pooled: reports expected leakage under random 80/20 and 70/30 splits)
python leakage_audit.py audit --layout csv --img-dir aptos/train_images \
    --csv aptos/train.csv --id-col id_code --label-col diagnosis --ext .png --out out_aptos

# Brain MRI (official Training/Testing split)
python leakage_audit.py audit --layout split-class --root brain_mri --out out_brain
```
The first run downloads ImageNet ResNet-50 weights (~100 MB). On an M-series Mac it uses the GPU automatically.

## Outputs
| file | what it is |
|---|---|
| `summary.json` | headline numbers for the paper |
| `pairs.csv` | every flagged pair with tier, pHash/dHash distance, pixel correlation (ncc), cosine, labels, splits |
| `manifest_groups.csv` | one row per image with its duplicate-group id (use this for a group-aware clean split) |
| `nn_similarity_hist.png` | each image's nearest-neighbour similarity (candidate paper figure) |
| `review.html`, `review_sheet.csv` | stratified sample of pairs for manual checking |

## Tiers
- **exact**: identical bytes or identical pixels
- **near_verified**: pixel correlation of border-cropped thumbnails ≥ 0.98 (re-saves, resizes, brightness changes)
- **candidate**: flagged by pHash / dHash / pixel correlation ≥ 0.95 / CNN cosine ≥ 0.95, but not verified. These go to manual review. Crops and "same eye, different photo" land here.

Headline counts use **exact + near_verified**, plus any pairs you confirm by hand.

## Manual verification (plan step 7)
1. Open `review.html` and fill `is_duplicate` (1/0) in `review_sheet.csv`.
2. `python leakage_audit.py review-stats --sheet out_aptos/review_sheet.csv` gives the false-positive rate per stratum with Wilson 95% CIs.
3. Rerun the audit with `--confirmed out_aptos/review_sheet.csv` to fold confirmed pairs into the groups.

## Validation
`tests/make_synthetic.py` plants exact copies, JPEG re-saves, resizes, brightness shifts, crops, label conflicts, and cross-split leaks.
On it, exact + near_verified found 27/29 planted pairs (both misses were crops) and 12/12 pairs on the split-folder set, with 0 false positives, and recovered the exact test contamination (8/48).
```bash
python tests/make_synthetic.py /tmp/synth
python leakage_audit.py audit --layout csv --img-dir /tmp/synth/aptos_like/train_images --csv /tmp/synth/aptos_like/train.csv --out /tmp/o --no-embed
python tests/score_against_truth.py /tmp/o /tmp/synth/aptos_like/truth.json
```
