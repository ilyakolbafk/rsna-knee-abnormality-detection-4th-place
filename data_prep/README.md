# Data preparation

Inference needs no prepared data: `predict.py` reads the DICOM files of the test set directly. Training needs
model-ready caches, built from the competition DICOM files by the code in this folder. All builders use the frozen
v8.3 orientation runtime (side + canonical left knee) and write lossless, hash-verified NPZ files.

| Cache | Used by | Built by | Size |
|---|---|---|---|
| cache336 (`native64-token-bank-v1`): ≤ 64 windows × 3 slices × 336 px per knee, 4,352 train knees + Gold58 | all ten CoAtNet models | 17 Kaggle CPU notebooks + `cache336/download_and_verify_cache.py` | ≈ 45 GB |
| extension cache (`k24-allseries-ext-v1`): extra windows from the series outside the five slots | `adaptive` | 16 Kaggle CPU notebooks + local Gold58 build + `extension_cache/finalize_cache.py` | ≈ 15 GB |
| test3 cache: the three sample test studies in cache336 format | `adaptive`, `joint36` | `test3_cache/build_test3_cache.py` (CPU) | 27 MB, prebuilt in `weights/training_inputs/test3_cache` |
| H6 v4 caches: RSNA, OAI, fastMRI (384 px views, ≈ 3.5 mm slabs) | `h6_core6`, `h6_vm` | `h6_v4/rsna`, `h6_v4/oai`, `h6_v4/fastmri` | ≈ 262 GB + 228 GB + 11 GB |

## cache336 (CoAtNet models)

The historical cache was built on Kaggle (CPU notebooks, competition data mounted, no internet): 16 train shards and a
Gold58 shard, one builder (`cache336/build_cache336.py`) and one asset dataset (the v8.3 runtime and the series tables).

1. Recreate the asset dataset byte for byte and upload it as your own private Kaggle dataset:
   `python train.py --stage cache336_assets --out ASSETS --weights WEIGHTS`.
2. In each notebook of `cache336/kaggle_kernels/` set your own owner in `kernel-metadata.json` (`id`,
   `dataset_sources`), push and run it (CPU). `cache336/SHARD_RELEASE_PLAN.json` lists the shards and their expected
   sample counts, `KERNEL_RELEASE_REGISTRY.json` the notebook hashes we ran.
3. Assemble and verify the release: `python data_prep/cache336/download_and_verify_cache.py --root CACHE336` (downloads
   the 17 outputs with the Kaggle CLI, checks every component, writes `GLOBAL_RELEASE_RECEIPT.json`,
   `TRAIN_MANIFEST.csv`, `GOLD58_MANIFEST.csv` and the labels). Two train studies have one undecodable source frame
   each; the builder records them as known source errors, as in the historical release.
4. Train with `--cache-identity relaxed`: a rebuilt release has new receipts and is compared semantically with
   `training/cache336_reference`.

## Extension cache (Adaptive)

`extension_cache/kaggle_kernels/rsna-k24-allseries-ext-s00…s15` build the train shards on Kaggle (same asset dataset).
Gold58 is built locally:

```bash
python data_prep/extension_cache/build_extension_cache.py --cohort gold58 --runtime RUNTIME \
    --archives GOLD58_DICOM --series-csv train_series.csv --output EXT/gold58
```

Then `finalize_cache.py` (numeric, metadata and visual checks) and `verify_ext_bank.py --bank EXT ...`, which fails
closed. `extension_cache/reference/` holds the historical manifests to compare against.

## test3 cache

The three sample test studies of the competition data in cache336 format, part of the Adaptive and Joint36 fits. Shipped
prebuilt in `weights/training_inputs/test3_cache`; `test3_cache/build_test3_cache.py` rebuilds it on CPU from
`test_series/`.

## H6 v4 caches (H6 only)

`h6_v4/README.md`: the three producers (RSNA from the competition DICOM files, OAI, fastMRI with fastMRI+ boxes) and the
`INPUTS.json` that `train.py --model h6_core6 | h6_vm` reads. The RSNA builder needs two large files in place first:
`weights/pretrained/orientation_v8_3_side.pt` → `h6_v4/rsna/orientation_v8_3/runtime/side.pt` and the public SKM-TEA
V-Net (`anatomy.pt` of the Kaggle dataset `heliosli/rsna-orthofoundation-fold0-epoch7-20261004`) →
`h6_v4/rsna/weights/anatomy.pt` (`h6_v4/rsna/README.md`).
