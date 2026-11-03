# Reproducibility

What this repository reproduces, how exactly, and where the limits are.

## 1. Inference: exact

`predict.py` runs the code and weights of the scored notebook `ilyakolb/rsna-final-six-h6-ens-20261022` v2
(submission 57532611). The stages are the historical runtimes, started the same way:

1. **Sixway** (`src/rsna_knee_final/inference/sixway_stage.py`): the resident Sixway runtime of the scored notebook in
   its own process (ten CoAtNet checkpoints, shared DICOM decode, v8.3 side resolver) → `sixway_core.csv`.
2. **H6** (`h6_prep.py`, `h6_fast.py`): the v4 RSNA producer on the test DICOM, both H6 arms in fp16 on the two T4s,
   with the time budget → `h6_core6_raw.csv`, `h6_vm_raw.csv`.
3. **Blend** (`blend.py`, `configs/blend_final_ens.json`): per-finding percentile ranks over the whole test set.

One change against the scored notebook: the H6 stop time is computed from the start of the run (default budget 9 h,
H6 stops 25 min before it) instead of the deadline-day wall clock that the notebook used; on the hidden run that
wall-clock cap did not fire, and the fallback rule (studies H6 did not reach in time get the Sixway rank) is unchanged.

Evidence:

| Check | Result |
|---|---|
| Private Kaggle notebook built from this repository (`kaggle/host_check`), 3 public sample studies | `submission.csv` byte-identical to the scored notebook's sample output (sha `9c9d3132…`); Sixway stage sha `9e42cd08…` = Sixway 57291068 (`docs/verification/HOST_CHECK.json`) |
| Same notebook on the 58 Gold58 training studies (DICOM from the competition data) | all 58 studies complete, both H6 arms within 5e-3 of the fp32 reference |
| Unmasked FINAL run of this repository submitted as a late submission (57566117, 23.10) | public 0.967 / private 0.958, the same as the scored notebook 57532611 |
| Column probes of this repository's FINAL and Sixway stages (13 late submissions) | all scored; per-finding AUC in `docs/SOLUTION_DETAILED.md` § 7, raw scores in `docs/data/column_probes.csv` |

The host-check notebook embeds the inference files as of 23.10; `predict.py` has since gained the `--blend` and
`--sixway-only` options, and its default path is unchanged.

Requirements: 2× T4, the pinned Kaggle image, about 5 h of wall time (`docs/ENVIRONMENT.md`). Other GPUs fail the
device assertion on purpose: fp16 numerics on other hardware would change the last digits of the probabilities and
possibly ties in the ranks.

## 2. Training: same code, same data, same recipe; not bit-identical

`train.py --model X` rebuilds the historical package of model X byte for byte (1,488 files verified by sha256) and
calls its trainer with the historical configuration; only machine-specific launcher code is replaced
(`training/HOST_PORT.md`). All random draws are seeded per sample and epoch, but CUDA kernels are not deterministic:
two runs of one recipe differed by ≈ 0.0026 macro AUC on a held-out fold, ≈ ±0.001–0.002 on the leaderboard. A
retrained ensemble should score like ours within that noise; it will not reproduce the checkpoint bytes.

`tests/cpu_smoke.py` checks the training side without a GPU: package staging and verification, the historical data
checks of all ten CoAtNet launchers on a synthetic cache with the historical cache metadata, one real CoAtNet optimizer
step, H6 target checks and network construction (`docs/verification/CPU_SMOKE.json`, PASS).

## 3. Data preparation

cache336, the extension cache and the 3-study test cache are rebuilt from the competition DICOM files by the builders in
`data_prep/` (Kaggle CPU notebooks for the large ones). The historical cache files themselves (≈ 60 GB of NPZ) were not
kept; a rebuilt cache is checked semantically against the historical metadata (`--cache-identity relaxed`). Two
training studies have one undecodable source frame each (shards 8 and 9); the builder records them as known source
errors, as the historical one did.

## 4. Labels

All label tables that trained the models are shipped inside the frozen packages (`labels/README.md`), with the code,
prompts and documents that produced them. Re-running the LLM readers will not give the same numbers: re-reading a study
flips 4–5 of every 100 binary cells, and the services change over time. The image reader used ordinary ChatGPT Pro chats
(paid subscription, the plan available in September–October 2026).

## 5. Known gaps

1. **H6 retraining** needs the OAI knee MRI (NIMH Data Archive) and fastMRI knee (NYU) data, the OAI targets
   (not distributed: the H6 target files hold RSNA rows only, see `training/H6_RECIPE.md`), the three v4 caches
   rebuilt by `data_prep/h6_v4/*`, and an `INPUTS.json` written for the training machine. The historical
   `INPUTS.json` binds cache files on our rented machine by size, mtime and inode and cannot be reused; writing a new
   one is a manual step (format in `data_prep/h6_v4/README.md`). The receipt `docs/verification/CPU_SMOKE.json` was
   recorded with the full 7,609-row targets (its `target_sha256` values); on the released files
   `tests/cpu_smoke.py` checks the 4,355 RSNA rows instead.
2. **Historical receipts contain machine paths** (`/root/…`, `/workspace/…`, `/Users/…`). They are kept unchanged
   because the packages are verified byte for byte; nothing reads those paths at run time (P3 in `HOST_PORT.md`).
3. **The RSNA-only hybrid** described in the write-up (57156616, crop specialists) is a separate notebook; its code and
   weights are not part of this repository, which reproduces the selected final 57532611.
4. **Late p4** (57543464) uses the same twelve models with different blend weights: `predict.py --blend late_p4`
   reproduces it; its deadline-day attempt (57519900) did not score (`docs/SOLUTION_DETAILED.md` § 6).
