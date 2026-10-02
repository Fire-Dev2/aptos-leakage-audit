# APTOS 2019 leakage audit

Code and data for the paper **"Data Leakage in Diabetic Retinopathy Classifiers for Tele-Ophthalmology Screening: A Code-Level Audit and Controlled Reproduction on APTOS 2019"** (T. Nambi, E. Kharat, N. Bhimireddy; under review).

Archived on Zenodo: [doi.org/10.5281/zenodo.23047712](https://doi.org/10.5281/zenodo.23047712) (this DOI always resolves to the latest version).

The repository contains:

1. **`leakage_audit/`**: the duplicate-audit tool (v0.3.0). It compares every image pair using exact hashes, perceptual and difference hashes, and pixel correlation of border-cropped thumbnails. It also includes a synthetic validation set.
2. **`data/duplicates/`**: the duplicate groups found in the 3,662 labeled APTOS 2019 images, with the verified pairs and the review of 94 sampled pairs. The first rating was AI-assisted; 20 of the pairs were also rated by two people, who agreed on all of them.
3. **`data/splits/`**: split files that keep every duplicate group on one side of the split. Use these to train and test on APTOS 2019 without cross-split duplicates.
4. **`data/screening/`**: the selection protocol, the PRISMA counts, the Europe PMC export and screening sheets, and the leakage coding of the 12 included studies.
5. **`reproduction/`**: instrumentation patches and run scripts for the three reproduced studies (DiffMIC, nnMobileNet and the graph-enhanced GCN classifier).
6. **`results/`** and **`figures/`**: per-run results, summary tables and the scripts that produce the paper's figures (see [Results and figures](#results-and-figures)).

**No images are included.** Every file refers to images by their Kaggle `id_code`. Download the data from the [APTOS 2019 Blindness Detection competition](https://www.kaggle.com/competitions/aptos2019-blindness-detection) under the competition rules.

## Headline numbers

- **Duplicates:** 134 exact and 14 near-duplicate pairs form 131 groups covering 270 images (7.4%). Of these groups, 37 contain conflicting grades.
- **Random splits:** these place a copy of a training image in the test set for 6.0% of test images at 80/20 and 5.3% at 70/30.
- **Checkpoint selection:** choosing the checkpoint on the test set added 2.2 accuracy points for DiffMIC, and 1.2 (model weights) or 0.8 (EMA weights) for nnMobileNet, over validation-based selection.
- **GCN classifier:** oversampling before the split plus grade-sorted test batches gave 98.2% accuracy. A duplicate-aware split and training-only oversampling gave 81.6% with randomly ordered test batches and 83.2% one image at a time.
- **Referable DR (grade 2 or higher):** read as a screen, the GCN model's sensitivity fell from 99.0% under the published protocol to 92.2% with the duplicate-aware split and one image at a time. Specificity fell from 99.4% to 93.3%.

## Using the duplicate groups

`data/duplicates/aptos2019_duplicate_groups.csv` has one row per image, with these columns:

| column | meaning |
|---|---|
| `id_code` | Kaggle image id |
| `grade` | label in `train.csv` (0–4) |
| `group_id` | duplicate group; images that share a `group_id` are copies of the same photograph |
| `group_size` | number of images in the group (1 = no duplicate) |
| `group_grade_conflict` | `True` if the group's images do not all carry the same grade |

Example: a group-aware stratified split.

```python
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
g = pd.read_csv("data/duplicates/aptos2019_duplicate_groups.csv")
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
train_idx, test_idx = next(sgkf.split(g, g.grade, g.group_id))
```

`data/splits/groupaware_70_15_15/` is a ready-made 70/15/15 split (2,564 / 550 / 548 images) built this way. It is the split used for condition B of the GCN rerun.

## Running the audit tool

```bash
cd leakage_audit
pip install -r requirements.txt
python leakage_audit.py audit --layout csv --img-dir /path/to/train_images \
    --csv /path/to/train.csv --id-col id_code --label-col diagnosis --ext .png --out out_aptos \
    --no-embed --sim-seeds 500 --review-per-tier 40
```

The paper's run used the settings recorded in `data/duplicates/audit_summary.json`: hash size 16 (256-bit hashes), 64 px thumbnails, dark-border threshold 15, no CNN embeddings, and 500 simulated random splits per ratio. A pair is a candidate if its pHash or dHash distance is at most 32 bits or its thumbnail correlation is at least 0.95. A candidate is a verified near-duplicate only if its correlation is at least 0.98 **and** its pHash distance is at most 12 bits; exact duplicates have identical bytes or identical pixels. `leakage_audit/tests/` builds the synthetic validation set and scores it against the truth file.

## Reproduction experiments

Each study is run from the authors' code at a fixed commit. The patches add instrumentation only: prediction dumps at every evaluation, data paths and seed from environment variables, and a validation pass. Training, models and the released checkpoint rules are not changed.

| study | upstream repository | commit | conditions |
|---|---|---|---|
| DiffMIC (MICCAI 2023) | github.com/scott-yjyang/DiffMIC | `8a094f7` | R0 released split and rule; R1 with a 352-image validation set |
| nnMobileNet (CVPRW 2024) | github.com/Retinal-Research/NN-MOBILENET | `920acd3` | same R0/R1 splits (the authors' split was not released) |
| GCN classifier (PLoS Comput. Biol. 2025) | github.com/mfar201/diabetic_retinopathy_classification_gcn | `fb123ab` | A: oversample, then split; B: duplicate-aware split, then oversample the training set |

The runs used the Ohio Supercomputer Center (Slurm, NVIDIA A100). To run elsewhere, replace `YOUR_OSC_PROJECT` in the `.sbatch` files, or run the `run_*.py` scripts directly. Each script clones the upstream repository, applies the patch, downloads APTOS 2019 with the Kaggle API (you need your own `~/.kaggle/kaggle.json`) and runs three seeds (1234, 1 and 2).

`reproduction/common/analyze_runs.py` selects checkpoints from the saved predictions (test-selected, validation-selected, last epoch) and computes paired differences with 95% t-intervals. The per-run outputs used in the paper are in `results/`.

## Results and figures

| file | what it holds | paper |
|---|---|---|
| `results/diffmic/phase5_per_run.csv`, `phase5_selection_inflation.csv` | DiffMIC: every run and rule; paired test-minus-validation differences | Section III-C, III-D |
| `results/nnmobilenet/nnmb_per_run.csv`, `nnmb_selection_inflation.csv` | nnMobileNet (model and EMA weights): the same | Section III-C, III-D |
| `results/table_phase5_headline.csv` | one row per model, from `figures/fig_duplicates_both.py` | Section III-C, III-D |
| `results/gcn/gcn_per_run.csv`, `gcn_summary.csv` | GCN: five-grade metrics for both conditions and three test orders, from `gcn_analysis.py` | Section III-E, Fig. 2 |
| `results/gcn/gcn_referable_per_run.csv`, `gcn_referable_summary.csv` | GCN: referable-DR sensitivity and specificity, from `gcn_referable.py` | Table III |
| `results/gcn/results_gcn.tgz` | GCN: per-image test predictions of every run (per-epoch evaluations, sorted, five random orders, one image at a time), run logs and split lists | input of the two GCN scripts |
| `figures/fig1_selection.py` → `fig1_selection.png` | test-selected and validation-selected checkpoints, same runs | Fig. 1 |
| `figures/fig2_gcn.py` → `fig2_gcn.png` | GCN accuracy, conditions A and B, three test orders | Fig. 2 |
| `figures/fig_duplicates_both.py` → `fig_duplicates_both.png` | accuracy by duplicate status | not in the paper |
| `figures/prisma.py` → `prisma_flow.png` | study-selection flow | not in the paper |
| `figures/fig_selection_inflation.png` | DiffMIC test accuracy at each evaluation, from `results/diffmic/phase5_analysis.py` | not in the paper |

To recompute the GCN tables, unpack the archive and run the two scripts from the folder that then contains `runs/`:

```bash
cd results/gcn && tar xzf results_gcn.tgz
python gcn_analysis.py      # writes gcn_per_run.csv, gcn_summary.csv
python gcn_referable.py     # writes gcn_referable_per_run.csv, gcn_referable_summary.csv
```

The figure scripts read the tables above and can be run from any folder, for example `python figures/fig2_gcn.py`.

`results/diffmic/phase5_analysis.py` and `results/nnmobilenet/nnmb_analysis.py` produced the DiffMIC and nnMobileNet tables. They read the per-epoch prediction files written by the run scripts and the audit tool's output folder, which are not in this repository, so they are included to document the computation; rerun the experiments to regenerate their inputs.

## Licenses

- Code: MIT (see `LICENSE`).
- Derived data files (duplicate groups, pairs, splits, screening and coding tables): CC BY 4.0.
- APTOS 2019 images are not redistributed here. The duplicate and split files carry each image's `id_code` and its grade from the competition's `train.csv`; that information remains subject to the Kaggle competition rules.

## Citation

See `CITATION.cff`. The paper is under review; this section will be updated with its DOI.
