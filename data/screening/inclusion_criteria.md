# Phase 3 — Paper selection protocol (fixed 2026-09-22, before screening)

## Inclusion (all must hold)
1. **Published 2020–2025**: peer-reviewed journal or conference paper, or an arXiv/medRxiv preprint.
2. **Uses the APTOS 2019 Blindness Detection training set** (3,662 images) as the main development and evaluation data, either alone or pooled with other sets, as long as APTOS images appear in the test split.
3. **Image-level DR classification**: binary (DR / no DR, or referable / non-referable) or 5-grade.
4. **Reports test performance** (accuracy, kappa, AUC, F1, sensitivity or specificity) on a split the authors made from APTOS.
5. **Public code**, linked from the paper, or a repository whose README explicitly cites that paper.
6. **Code is runnable in principle**: it contains the training and evaluation scripts, not only a demo or inference script, in Python (PyTorch, TensorFlow or Keras).

## Exclusion reasons (logged per paper, first reason that applies)
- E1 Outside 2020–2025
- E2 APTOS used only as an external test set, or not used at all
- E3 Not image-level classification (segmentation, lesion detection, image quality, generation only)
- E4 No test metric on an APTOS split
- E5 No public code
- E6 Code incomplete (no training or evaluation script)
- E7 Duplicate record (same paper found twice); counted once
- E8 Not in English / full text unavailable

## Sources searched
- GitHub repository search (queries: `aptos 2019`, `aptos2019`, `aptos blindness`, `aptos diabetic retinopathy`), keeping repositories that link a paper
- Web and scholarly search (Google Scholar, arXiv, publisher sites) for "APTOS 2019" + diabetic retinopathy + classification, plus "code available" / "github"

## What gets recorded for each included paper
Title, year, venue, DOI/arXiv ID, code URL, framework, task (binary / 5-class), split described in the paper, headline reported metrics, and preprocessing or augmentation notes relevant to leakage.
