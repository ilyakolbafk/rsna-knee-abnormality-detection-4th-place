# Winning model documentation: RSNA Knee Abnormality Detection 2026

## A1. Competition and team

- Competition: RSNA Knee Abnormality Detection (Kaggle, `rsna-knee-abnormality-detection`)
- Team: Ilya Kolb (one person)
- Private leaderboard: 0.958, 4th place (selected submissions 57532611 and 57498146, both 0.967 public / 0.958 private)
- Name: Ilya Kolb
- Location: Tashkent, Uzbekistan
- Email: kolbiliya@yandex.ru

## A2. Background

- Professional background: data analyst in CRM analytics at Uzum: SQL data warehouses (ClickHouse, Greenplum),
  Airflow pipelines and data marts, customer segments, and post-analysis of CRM campaigns (test vs control groups,
  effect sizes, significance).
- Prior experience that helped: reading noisy experiments. In CRM work a result only counts against a decision rule
  fixed in advance and a known noise level; we treated every leaderboard submission the same way (rule declared before
  the score, ±0.001–0.003 noise, one change per comparison) and Gold58 as a small, reused validation set. Daily work
  with LLM tools also made it natural to give an LLM a written reading protocol and check it like any other annotator.
- Why this competition: the goal from the start was a top place on the leaderboard, solo.
- Time spent: about eight weeks (first submission 29.08.2026, last 22.10.2026), about 60 hours per week.
- Tools: training on rented single RTX 5090 instances; Kaggle 2× T4 for submissions; ChatGPT Pro (web) as the image
  reader that produced the training labels; LLMs reading the report text for auxiliary and comparison labels;
  OpenAI Codex and Claude as coding assistants.

## A3. Summary

The training data had radiology reports but almost no image-based labels, while the test set was labelled by radiologists
from the images. We therefore produced our own training labels by having ChatGPT Pro read rendered MRI panels of every
training study, with the report as an untrusted hint ("VM labels"); against the 58 image-labelled training studies these
labels reach 0.940 macro AUC versus 0.919 for report-derived labels, and the same network trained on them gained
+0.027 private AUC. The model is a CoAtNet-rmlp-2-384 encoder over up to 64 three-slice windows per knee with a
per-finding attention-MIL head (12 findings + 48 auxiliary report targets), trained with 24 sampled windows per step
(K24). The selected final blends, by per-finding percentile ranks, ten such checkpoints ("Sixway": five folds, a full
fit, a full fit on repaired labels, a mirror-trained, an extra-series and a three-label-source variant) with H6, a
ConvNeXt-S dense-window model on heliosli's public R7 training code in two separately trained arms (report-reader
labels; VM labels), with OAI knees in the fit (MOAKS grades as targets for Lateral OA, PF OA and synovitis in one arm,
as in nartaa's public solution; VM labels in a separate OAI head in the other) and fastMRI+ boxes as an auxiliary loss.
Inference: ≈ 5 h on 2× T4. Write-up: `docs/SOLUTION.md` (detailed version with all experiments: `docs/SOLUTION_DETAILED.md`).

## A4. Features / inputs

- DICOM → canonical left knee (v8.3 side and orientation resolver: tag, DICOM geometry and a ResNet-50 voting on
  7 frames per series; mirror for right knees).
- Five series slots (sagittal FS, sagittal non-FS, coronal FS, coronal non-FS, axial FS), windows of three adjacent
  slices as RGB, the central 140 mm at 336 px, up to 64 windows per knee, 6 metadata values per window (relative
  position, offset in mm, fat-suppression one-hot). One variant adds up to 24 windows from the remaining series.
- Per-series intensity window from the 0.5–99.5 percentiles of the sampled frames, 8-bit.
- H6: 384 px full-field and 100 mm joint-centred views anchored by an SKM-TEA-trained V-Net, ≈ 3.5 mm slabs, up to
  176 windows per study in training, every window at inference.

## A5. Training

- Loss: masked soft-label BCE with fixed per-finding `pos_weight`; 48 auxiliary heads at weight 0.25.
- Optimiser: AdamW (3e-5 backbone, 1e-3 heads), weight decay 0.02, OneCycle over 24 epochs stopped at 20, batch 8 knees,
  gradient clip 3, BF16, seed 20260908; average of the weights of epochs 18–20.
- Bag sampling: 24 of ≤ 64 windows per step, 15 % window dropout, 10 % series dropout, ±1 slice jitter, random gain;
  image augmentation per series (crop/zoom, rotation, shift, gamma, contrast, bias field, blur, noise).
- Full fits use 4,410 knees: the 4,352 VM-labelled knees plus the 58 Gold58 studies with their binary labels; fold
  models use 80 % of the VM-labelled knees (Gold58 held out).
- H6: ConvNeXt-S (22k→1k), 16 epochs, EMA, SWA of the EMA weights of epochs 14–16; two arms trained separately
  (Core6 report-reader labels + OAI MOAKS on three findings; VM labels + a separate OAI VM head), likelihood-ratio loss
  on soft targets, fastMRI+ box loss at weight 0.3 (`training/H6_RECIPE.md`).
- Blend: per-finding percentile ranks; Medial/Lateral/PF OA and Baker's: Sixway 0.5, H6-core6 0.25, H6-VM 0.25;
  ACL, MCL, menisci: Sixway 0.75, H6-core6 0.25; contusion: 0.5 / 0.25 / 0.25; effusion, synovitis, fracture: Sixway.
  Weights fixed before submission.

## A6. Interesting findings

- Labels from an LLM reading the images beat every report-label pipeline: one fold model on VM labels outscored our full
  report-label ensemble (0.952 vs 0.936 private).
- The student beats its teacher: the K24 model trained on VM labels scores 0.943 on Gold58 versus 0.940 for the labels
  themselves; where it is ≥ 0.4 above its label, the radiologists say "positive" in 88 % of cells. A rule-based repair of
  1.7 % of the cells made a useful extra ensemble member.
- K24 instead of K12 sampled windows: with 12 of 64 windows a focal finding is missing from 60–70 % of training bags;
  K24 gave +0.003 public / +0.002 private, concentrated on lateral OA, PF OA, ACL, synovitis, contusion.
- Hand-written geometry (joint line from intensity profiles, notch, skin line) is enough to place compartment and
  anterior crops; our RSNA-only hybrid with two such crop specialists reached 0.965 / 0.958 without external data.
- Our best blend failed on the last day because a strict guard wrote no file when the hidden run was slow; the same
  notebook scored 0.968 / 0.959 the next day.

## A7. Simple model

The single K24 full-fit CoAtNet (`--model core`, one checkpoint) scores 0.964 public / 0.955 private, runs in
≈ 0.5 h on 2× T4 and is the core of every blend.

## A8. Execution time

| | |
|---|---|
| Training, one CoAtNet full fit | ≈ 5.5 h on one RTX 5090 (fold model ≈ 4.5 h); ten checkpoints ≈ 50 GPU-hours |
| Training, H6 | 16 epochs per arm on one RTX 5090 |
| Inference, FINAL ens | ≈ 4.5–5 h on Kaggle 2× T4 for ≈ 1,300 studies (Sixway ≈ 2.3 h, H6 ≈ 2.3 h) |
| Inference, single core model | ≈ 0.5 h |

## A9. References

- timm `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`; ConvNeXt (Liu et al. 2022, Meta weights `convnext_small_22k_1k_384`).
- Attention-based deep MIL (Ilse et al. 2018).
- SKM-TEA (Desai et al. 2021) V-Net weights and the R7 ConvNeXt-S training code (`compact_train.py` v9.8), both from
  public heliosli datasets; OAI supervision of Lateral OA, PF OA and synovitis as in nartaa's public CoAtNet solution.
- OAI (Osteoarthritis Initiative, NIMH Data Archive); fastMRI (Zbontar et al. 2018) and fastMRI+ annotations (Zhao et al. 2022).
- Public label tables by laymond and yunus (Kaggle datasets).

## B. Submission model

| Item | Where |
|---|---|
| B1 code, data, models | this repository + the `weights/` archive |
| B2 README | `README.md` |
| B3 configuration | `configs/` (blend weights, H6 recipe, weights manifest), `training/host_configs/` |
| B4 requirements | `requirements.txt` (training), `docs/ENVIRONMENT.md` (inference image) |
| B5 directory structure | `directory_structure.txt` |
| B6 settings | `SETTINGS.json` |
| B7 trained models | `weights/` (sha256 of every file in `configs/weights_manifest.json` and `weights/SHA256SUMS`) |
| B8 entry points | `entry_points.md` |
