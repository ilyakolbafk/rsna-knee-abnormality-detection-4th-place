# RSNA Knee Abnormality Detection 2026: FINAL ens (4th place, team Ilya Kolb)

This repository reproduces our selected final submission **57532611 "FINAL ens"** (public 0.967 / private 0.958) and
retrains every model in it. FINAL ens is a per-finding percentile-rank blend of **Sixway** (ten CoAtNet-rmlp-2-384
attention-MIL checkpoints in six components) and **H6** (a ConvNeXt-S dense-window model in two arms).

| Document | |
|---|---|
| `docs/MODEL_SUMMARY.md` | winning-model documentation (Kaggle template) |
| `docs/SOLUTION.md` | write-up as published on Kaggle: labels, models, results, what did not work |
| `docs/SOLUTION_DETAILED.md` | longer write-up with every experiment, column probes and the deadline-day run |
| `docs/REPRODUCIBILITY.md` | what is reproduced exactly and what is not |
| `training/README.md`, `training/H6_RECIPE.md` | how each of the twelve networks is retrained |
| `docs/ENVIRONMENT.md` | hardware and software of inference and training |
| `entry_points.md`, `SETTINGS.json`, `directory_structure.txt`, `requirements.txt` | Kaggle template items |
| `LICENSE` | CC BY-NC 4.0 (the competition's winner licence) and the third-party components |

**Verified:** a Kaggle notebook built from this repository gives a byte-identical `submission.csv` on the
sample test studies (`docs/verification/HOST_CHECK.json`); the same pipeline submitted on the hidden test (late
submission 57566117) scored 0.967 / 0.958; the CPU smoke test of all training entry points passes
(`docs/verification/CPU_SMOKE.json`).

## Links

- Scored inference notebook: https://www.kaggle.com/code/ilyakolb/rsna-final-six-h6-ens-20261022
- Write-up on Kaggle: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/4th-place-llm-image-labels-attention-mil
- Video walkthrough (5 min): https://www.youtube.com/watch?v=EG6_foormiM
- Weights, the nine Kaggle datasets attached to the scored notebook:
  - https://www.kaggle.com/datasets/ilyakolb/rsna-sixway-resident-runtime-20261017
  - https://www.kaggle.com/datasets/ilyakolb/rsna-vm336-k24-folds-fullfit-weights-20261016
  - https://www.kaggle.com/datasets/ilyakolb/rsna-v20-ordinary-k24-fullfit-weights-20261013
  - https://www.kaggle.com/datasets/ilyakolb/rsna-vm336-k24-vmrepair-fullfit-weights-20261016
  - https://www.kaggle.com/datasets/ilyakolb/rsna-mirror336-k24-r3-raw-weights-20261016
  - https://www.kaggle.com/datasets/ilyakolb/rsna-adaptive336-fullfit-weights-20261017
  - https://www.kaggle.com/datasets/ilyakolb/rsna-j36-masked336-weights-20261016
  - https://www.kaggle.com/datasets/ilyakolb/rsna-h6-rank-full336-20261022
  - https://www.kaggle.com/datasets/ilyakolb/rsna-h6-swa1416-weights-20261022

## Layout

```
predict.py, train.py   inference and training entry points
src/rsna_knee_final/   inference stages (Sixway, H6, blend), training host, source staging and checks
configs/               blend weights, H6 inference recipe, sha256 of every inference file, reference outputs
training/              the historical training code of each model, stored byte for byte (sources/, SOURCE_TREES.json),
                       cache metadata, launcher configs, H6 recipe; see training/README.md
data_prep/             cache builders for training; see data_prep/README.md
labels/                how the training labels were made (prompts, renderer, readers) and Gold58; see labels/README.md
kaggle/                the scored notebook and the builder of the verification notebook
tests/cpu_smoke.py     CPU test of the training side
tools/average_h6.py    SWA of the H6 EMA weights (epochs 14–16)
docs/                  documents above, verification receipts, figures, probe scores
```

## Weights

Delivered separately as `weights/` (4.3 GB for inference, 0.1 GB for training only): nine folders that are the nine
Kaggle datasets of the scored notebook (links above) (the twelve trained networks, their runtimes, offline wheels), plus
`pretrained/orientation_v8_3_side.pt` (our side and orientation model) and `training_inputs/test3_cache`. The sha256 of
every file is in `weights/SHA256SUMS`; the inference files are also in `configs/weights_manifest.json` and are checked
before every run.

Training additionally needs three public pretrained files in `weights/pretrained/` under these names:

| File | Source | sha256 |
|---|---|---|
| `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k.safetensors` | Hugging Face `timm/coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, `model.safetensors` | `91570c7af10fea0af03c1b0a49be3e4775596239890eb4193c75129b3045623e` |
| `convnext_small.fb_in22k_ft_in1k_384.pth` | `https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_1k_384.pth` | `e891e213cef59c33c099b502ec2adbad7e0d02b6c1ea2847127a3d0f00a88395` |
| `anatomy_skmtea_vnet.pt` (H6 cache builder only) | Kaggle dataset `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` v1, `anatomy.pt` (SKM-TEA V-Net) | `5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e` |

## Inference

Hardware and software as scored: Kaggle notebook with 2× NVIDIA T4, the Kaggle Python 3.13 image
(`docs/ENVIRONMENT.md`), no internet. The pipeline asserts two T4 GPUs. Wall time on the hidden test ≈ 4.5–5 h on a
normal day (Sixway ≈ 2.3 h, H6 ≈ 2.3 h); the run stays inside 9 h on slow days because studies H6 does not reach in
time keep the Sixway rank.

On Kaggle: attach the competition data and the nine weight datasets, add this repository, run

```bash
python predict.py --input-root /kaggle/input --out /kaggle/working
```

`kaggle/host_check/build_check_kernel.py` builds such a notebook with the repository embedded. Locally, with the Kaggle
Docker image on a 2× T4 machine:

```bash
python predict.py --weights WEIGHTS --data /path/to/rsna-knee-abnormality-detection --out EMPTY_DIR
```

Outputs: `sixway_core.csv`, `h6_core6_raw.csv`, `h6_vm_raw.csv`, `submission.csv`, `RUN_RECEIPT.json`. Other blends of
the same models: `--blend late_p4` (late submission 57543464, 0.968 / 0.959), `--sixway-only` (57291068, 0.966 / 0.957).

## Training

```bash
python train.py --list                                    # the 12 deployed networks and their historical code
python train.py --check --weights WEIGHTS                 # sha256 of all stored training files and pretrained files
python train.py --model core --cache-root CACHE336 --weights WEIGHTS --runs RUNS
python train.py --model h6_core6 --h6-inputs INPUTS.json --weights WEIGHTS --runs RUNS
python tools/average_h6.py --run RUNS/h6_core6 --out core6_swa_ema14_16.pt
```

Each network is trained by the exact code that produced it, staged from `training/sources` and verified by sha256;
only launcher code tied to our rented machines is replaced (`training/HOST_PORT.md`). One RTX 5090 (32 GB) per run:
≈ 5.5 h per CoAtNet full fit, ≈ 4.5 h per fold model, 16 epochs per H6 arm. Build the caches first
(`data_prep/README.md`). Per-model commands: `training/README.md`.

## External data and pretrained models

ImageNet initialisations from timm (CoAtNet) and Meta (ConvNeXt); the SKM-TEA V-Net published by heliosli for the knee
anchors of the H6 views; the public R7 training code of heliosli on which H6 is built. H6 also trains on OAI
(NIMH Data Archive) and on fastMRI knee data with fastMRI+ annotations; all other models use only the competition
data. No OAI images, OAI-derived targets or OAI case lists are distributed (OAI is shared under the NIMH Data Archive Data Use Certification); to retrain
H6 you need your own OAI access (`training/H6_RECIPE.md`). The released weights are enough to reproduce the submission. Labels are our own LLM readings of the competition images and reports plus two public label tables (laymond,
yunus); see `labels/README.md`.
