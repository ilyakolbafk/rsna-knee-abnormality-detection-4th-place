# H6 training recipe

H6 is the second model of FINAL ens: a ConvNeXt-S dense-window network trained in two arms (Core6 and VM) that differ
only in their targets. The network, window sampling, augmentation and the fastMRI+ auxiliary loss come from the
public R7 training code of heliosli; the code that trained the deployed weights is stored byte for byte in
`training/sources/h6` and is run by `python train.py --model h6_core6 | h6_vm`. Numbers below are read from that code
(`CONFIG.json`, `model.py`, `public_math.py`, `sampling.py`, `augmentation.py`, `fm_engine.py`, `loop.py`, `train.py`).

## Data

| Corpus | Studies in the fit | Core6 arm targets | VM arm targets |
|---|---|---|---|
| RSNA (v4 cache) | 4,355 (training set and the three sample test studies); Gold58 excluded | our report-reader labels, 12 findings | our VM labels, 12 findings |
| OAI baseline knees (v4 cache) | 3,254 | MOAKS-derived targets for Lateral OA (2,192 knees), PF OA (2,114) and synovitis (1,642); the other nine findings masked | our VM labels of 1,975 knees, in a separate OAI head |
| fastMRI knee + fastMRI+ boxes | 1,172 volumes | auxiliary box task (both arms) | same |

The targets are frozen in `training/sources/h6/assets/{core6,vm}_targets.npz` (soft probability, likelihood ratio and
weight per cell). **OAI rows are not distributed.** OAI is shared by the NIMH Data Archive under a Data Use Certification, and we do
not redistribute data derived from it, so these files hold only the 4,355 RSNA rows (`source_id` 0); the 3,254 OAI rows (`source_id` 1:
MOAKS-derived targets and our VM reads of OAI knees) were removed, and the sha256 pins in `PACKAGE_PINS.json` and
`training/sources/SOURCE_TREES.json` were updated to the reduced files. To retrain H6 with OAI, obtain OAI access
from the NIMH Data Archive under your own Data Use Certification, rebuild the OAI cache with `data_prep/h6_v4/oai`
and append the OAI rows in the same format: MOAKS grades from the OAI MOAKS release for Lateral OA, PF OA and
synovitis (other nine findings masked) in the Core6 arm, and image-read labels for the OAI head of the VM arm
(reader protocol in `labels/vm_reader`; our OAI reads are not shared). Without OAI rows H6 trains on RSNA and
fastMRI only, which is not the submitted model (`admission.py` stops with `FROZEN_FIT_ROSTER` until the 3,254 OAI
rows are back); `admission.py` checks both arm rules on them before training. The caches are built by
`data_prep/h6_v4` (`data_prep/README.md`).

**Views.** Every series is resampled to 384 × 384 uint8 views: a full field of view and 100 mm local views around the
knee anchors (sagittal and axial: joint centre, patellofemoral, posterior; coronal: medial, lateral), in the canonical
left orientation of our v8.3 side model. Thin stacks are averaged to slabs of ≈ 3.5 mm. A window is three neighbouring
slabs as three channels.

## Network

- Encoder: ConvNeXt-S, `convnext_small.fb_in22k_ft_in1k_384` (Meta 22k→1k checkpoint mapped with timm's
  `checkpoint_filter_fn`, every feature tensor and the pooled LayerNorm loaded strictly), stochastic depth 0.1,
  gradient checkpointing, ImageNet normalisation, 352 px input.
- Class-specific spatial pooling: for each of the 12 findings a query over 128-d keys
  `tanh(W f(p) + W_pos pe(p))` with fixed 2-D sine/cosine position features; the query starts at zero, i.e. at the
  plain average; the pooled LayerNorm of ConvNeXt is applied per class.
- Window tokens: projection to 384 (Linear + LayerNorm) plus series-slot, view and geometry embeddings (zero at
  start), dropout 0.1.
- Study readout: per-finding attention over all windows of the study, LayerNorm, one linear readout per finding; the
  bias starts at the logit of the finding's prevalence (`head_bases` in `CONFIG.json`).
- Synovitis logit = a synovium-specific part + β × the effusion logit, detached (β starts at 0), so the synovitis
  loss cannot reshape effusion.
- Auxiliary head: LayerNorm + linear map to 11 fastMRI+ classes on the 11 × 11 feature map; not used at inference.
- VM arm only: a second copy of the readout (`oai_vm`) for OAI knees. RSNA knees train the main readout, OAI knees
  the OAI readout; the encoder is shared. Inference uses the main (RSNA) readout of both arms.

## One training step

- 2 studies (one batch), drawn from a fixed per-epoch permutation of the 7,609 studies (`seed_base + epoch`),
  last incomplete batch dropped: 3,804 steps per epoch, 60,864 steps in 16 epochs.
- Windows: every (series, view) is covered at a stride of 3 slabs with a random phase; with probability 0.25 a
  window takes its neighbours two slabs apart; at most 176 windows per study (a random subset when there are more).
- Augmentation: one random 352 px crop offset per batch; per study rotation ±10°, scale 0.9–1.1, gamma 0.8–1.25,
  contrast 0.9–1.1; with probability 0.5 an anatomical mirror (sagittal: slice order reversed; coronal and axial:
  horizontal flip), which swaps the medial and lateral labels (menisci, OA) and the medial and lateral views, and
  masks MCL for that batch.
- Clinical loss: likelihood-ratio loss on soft targets, `−log(σ(z)·LR + σ(−z))` per cell, averaged over all
  batch × 12 cells with weight 0 for masked cells; in the VM arm each readout's loss is scaled by its share of the batch.
- fastMRI+ loss: 96 windows per step (4 volumes × 24 slices, 75 % of volumes with boxes, views full / medial /
  lateral, random 352 crop and flip, the same gamma and contrast ranges), dense box BCE with `pos_weight` 8 plus a
  presence BCE on the log-mean-exp of each class map, weight 0.3.
- Optimizer: AdamW, learning rate 1e-4 (encoder) and 1e-3 (head and readouts), weight decay 0.05, linear warm-up over
  5 % of the steps then cosine to zero, gradient clip 5 on the merged gradient, BF16 autocast.
- EMA of the weights after every step with decay `min(0.999, (1 + t) / (10 + t))`.
- Seeds: `seed_base` 1; model initialisation 1000 + 17 · seed; augmentation streams 5000 + 17 · seed and
  9400 + 17 · seed; the fastMRI stream has its own forked seed per step.

## Output and the deployed weights

The trainer writes the raw and EMA weights of every epoch (`epochNN_raw.pt`, `epochNN_ema.pt`) and an in-run average
of the EMA weights of epochs 5–10 (`swa_ema05_10.pt`, not deployed). The deployed weights are the mean of the EMA
weights of epochs 14, 15 and 16, made after training:

```bash
python train.py --model h6_core6 --h6-inputs INPUTS.json --weights WEIGHTS --runs RUNS
python tools/average_h6.py --run RUNS/h6_core6 --out core6_swa_ema14_16.pt
python train.py --model h6_vm --h6-inputs INPUTS.json --weights WEIGHTS --runs RUNS
python tools/average_h6.py --run RUNS/h6_vm --out vm_swa_ema14_16.pt
```

One arm needs one GPU with BF16 and 32 GB (we used an RTX 5090). At inference every window of the study is read
(stride 1) with the centre 352 crop, in FP16 on the Kaggle T4s (`configs/h6_inference_recipe.json`).
