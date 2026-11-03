# Training the 12 FINAL ens checkpoints

FINAL ens deploys twelve trained networks: ten CoAtNet-rmlp-2-384 attention-MIL checkpoints (Sixway) and the two arms of
H6 (ConvNeXt-S). Every one of them is retrained by **the exact historical package that produced it**: the packages are
stored byte for byte in `training/sources/` (1,488 files, sha256 of each in `SOURCE_TREES.json`; large pretrained files
live in the weights folder). `train.py` rebuilds a package into a fresh run directory, verifies every file, and calls the
package's own trainer with the configuration its historical launcher used. Only launcher code that was tied to our rented
machines is replaced (`HOST_PORT.md`).

| `--model` | Sixway component | Package (`training/sources/…`) | Labels (12 main) | Data | Deployed checkpoint (weights folder) |
|---|---|---|---|---|---|
| `fold0` … `fold4` | K24 five-fold mean (one member) | `fold01`, `fold2`, `fold3`, `fold4` | VM (`V_MAIN.csv`) | cache336, 80 % per fold | `rsna-vm336-k24-folds-fullfit-weights-20261016/fold{0..4}.pt` |
| `core` | K24 full-fit core | `core` | VM + Gold58 (`FULLFIT_MAIN.csv`) | cache336 | `rsna-v20-ordinary-k24-fullfit-weights-20261013/checkpoint.pt` |
| `repair` | VM-repair core | `repair` | repaired VM + Gold58 | cache336 | `rsna-vm336-k24-vmrepair-fullfit-weights-20261016/checkpoint.pt` |
| `mirror` | Mirror50, no side at inference | `mirror` | VM + Gold58 | cache336 | `rsna-mirror336-k24-r3-raw-weights-20261016/checkpoint.pt` |
| `adaptive` | Adaptive336 (+ up to 24 extra-series windows) | `adaptive` | VM + Gold58 + 3 public sample studies | cache336 + extension cache + test3 cache | `rsna-adaptive336-fullfit-weights-20261017/checkpoint.pt` |
| `joint36` | Joint36 (VM, Evidence, report heads) | `joint` | VM, Evidence-v1, Core6 | cache336 + test3 cache | `rsna-j36-masked336-weights-20261016/checkpoint.pt` |
| `h6_core6`, `h6_vm` | H6 arms | `h6` | Core6 / VM targets (`assets/*_targets.npz`) | H6 v4 caches (RSNA, OAI, fastMRI) | `rsna-h6-swa1416-weights-20261022/{core6,vm}_swa_ema14_16.pt` |

`python train.py --list` prints the same table from `src/rsna_knee_final/train_host.py`.

## Recipe (CoAtNet, all ten)

`train_core_v2_a.py` / `fullfit_k24.py` of each package: CoAtNet-rmlp-2-384 (timm `sw_in12k_ft_in1k` initialisation,
`weights/pretrained/coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k.safetensors`), per-finding attention MIL head, 12 main + 48
auxiliary report targets (λ 0.25), masked soft-label BCE with frozen `pos_weight`; 24 of ≤ 64 windows per study per step
(K24), series dropout 0.10, window dropout 0.15, ±1 slice jitter, gain U(0.9, 1.1), E1 image augmentation; AdamW
3e-5 / 1e-3, weight decay 0.02, OneCycle over 24 epochs stopped after epoch 20, batch 8, gradient clip 3, BF16, seed
20260908; the deployed checkpoint is `training/averaged.pt`, the mean of the epoch 18, 19 and 20 weights.
Variants: `mirror` adds a 50 % whole-knee mirror draw per sample and epoch; `adaptive` adds up to 24 extra windows
(37.5 %) from series outside the five slots with 8 metadata features; `joint36` trains 36 masked heads (VM, Evidence-v1,
report readers; branch weights .45 / .45 / .10) and is read out as an equal rank mean of its allowed heads.

## H6

Full recipe: `training/H6_RECIPE.md`. In short: ConvNeXt-S 384 (Meta 22k→1k weights,
`weights/pretrained/convnext_small.fb_in22k_ft_in1k_384.pth`), up to 176 dense windows per study, 16 epochs, EMA,
likelihood-ratio loss on soft targets, fastMRI+ box loss (weight 0.3), Gold58 not in the fit; 4,355 RSNA and 3,254 OAI
knees. The deployed weights are the mean of the EMA weights of epochs 14, 15 and 16:

```bash
python train.py --model h6_core6 --h6-inputs INPUTS.json --weights W --runs RUNS
python tools/average_h6.py --run RUNS/h6_core6 --out core6_swa_ema14_16.pt     # same for h6_vm
```

`INPUTS.json` describes the three v4 caches on the training machine (`data_prep/README.md`, H6 section).

## Commands

```bash
python train.py --check --weights W          # every stored source file and the pretrained files against their sha256
python train.py --model fold3 --cache-root CACHE336 --weights W --runs RUNS
python train.py --model adaptive --cache-root CACHE336 --extension-root EXT --test3-cache W/training_inputs/test3_cache --weights W --runs RUNS
python train.py --model joint36 --cache-root CACHE336 --test3-cache W/training_inputs/test3_cache --weights W --runs RUNS
python train.py --model core --check-data --cache-root CACHE336 --weights W --runs RUNS   # data checks only, CPU
```

A run writes `RUNS/<model>/package/` (the verified historical package), `RUNS/<model>/training/` (the trainer's own
outputs: `history.json`, `IDENTITY.json`, `RESULT.json`, epoch weights, `averaged.pt`) and `RUN_INFO.json`.
A run directory is used once; a second attempt needs a fresh `--runs` root.

`--cache-identity original` (default) requires the byte-identical historical cache release (its receipts are checked
byte for byte, as in the historical launcher). A cache rebuilt from DICOM has different receipts (they contain timings);
use `--cache-identity relaxed`, which checks it semantically against `training/cache336_reference` instead (same
samples, sides, token counts, labels, builder version and orientation state).

## Determinism

Seeds, sampling order and augmentation draws are fixed (`_rng(seed, epoch, sample_id, purpose)` per random choice), but
CUDA kernels are not bit-deterministic: two runs of one recipe on fold 3 differed by ≈ 0.0026 macro AUC on held-out
data in September, i.e. ±0.001–0.002 on the leaderboard. Expect retrained checkpoints to score like the deployed ones,
not to be bit-identical.

## Checks without a GPU

`python tests/cpu_smoke.py --weights W --test3-cache W/training_inputs/test3_cache` stages every package, runs the
historical data checks of all ten CoAtNet launchers on a synthetic cache built from the historical cache metadata,
performs one real optimizer step of the CoAtNet model on CPU and builds the H6 network. Last result:
`docs/verification/CPU_SMOKE.json` (PASS).
