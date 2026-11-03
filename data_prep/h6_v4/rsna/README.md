# H6 RSNA cache builder (v4 views)

Builds the RSNA part of the H6 training cache from the competition DICOM files: per series a full-field view and
100 mm local views around the knee anchors (sagittal and axial: joint centre, patellofemoral, posterior; coronal:
medial, lateral), 384 × 384 uint8, thin stacks averaged to ≈ 3.5 mm slabs, every knee in the canonical left
orientation of v8.3. Labels are not read and not stored in the cache.

Contents: `build_px_v4.py` (builder), `runtime_core.py`, `cache_store.py`, `support.py`, `qa_cache.py` (montages and
checks), `test_runtime.py`, `bootstrap.py` (Python 3.11 venv), `vendor/mainline` (decode, geometry and intensity
modules of the public heliosli code), `reference/` (public heliosli renderer and anchor bank), `orientation_v8_3/`
(our side and orientation runtime), `metadata/` (official series tables), `EXPECTED_RSNA_SAMPLES.json` and
`RSNA_SAMPLE_MAPPING.csv` (the 4,355 fit samples and the 58 Gold58 samples), `HANDOFF_MANIFEST.json` (sha256 of
every file; checked at start).

Two large files are not stored here and must be put in place first (the manifest expects them):

```sh
cp WEIGHTS/pretrained/orientation_v8_3_side.pt orientation_v8_3/runtime/side.pt
cp anatomy.pt weights/anatomy.pt
```

`anatomy.pt` is the public SKM-TEA V-Net checkpoint from the Kaggle dataset
`heliosli/rsna-orthofoundation-fold0-epoch7-20261004` (version 1), 112,496,037 bytes, sha256
`5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e`.

Run:

```sh
python3.11 bootstrap.py                               # creates ../rsna_v4_venv and runs test_runtime.py
../rsna_v4_venv/bin/python build_px_v4.py --dicom-root TRAIN_IMAGES --dicom-root TEST_SERIES --out RSNA_V4 --samples test3
../rsna_v4_venv/bin/python qa_cache.py RSNA_V4        # look at the montages before the full run
../rsna_v4_venv/bin/python build_px_v4.py --dicom-root TRAIN_IMAGES --dicom-root TEST_SERIES --out RSNA_V4 --samples all
../rsna_v4_venv/bin/python qa_cache.py RSNA_V4
```

The build resumes after an interruption, writes lossless NPZ chunks of at most 64 rows, and records per sample the
resolved series route, anchors, geometry and source files. When a knee anchor is not found, the documented fallback
of the public anchor bank is used and recorded (`anchor.method = slab_prior`).
