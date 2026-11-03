# Host port: what `train.py` replaces in the historical launchers

The historical trainings ran on rented single-GPU instances, driven by queue scripts that also enforced our budget and
recorded every step. Those launcher parts are tied to machines that no longer exist, so `src/rsna_knee_final/train_host.py`
replaces them. Nothing that determines the numbers is replaced: model code, data loading and sampling, augmentation,
labels and masks, class weights, optimizer, schedule, seeds, numerics flags, checkpoint averaging all run from the staged
historical files. Every run records `host_port` in its `IDENTITY.json` / `RUN_INFO.json`.

| Id | Historical launcher part | Replacement | Applies to |
|---|---|---|---|
| P1 | Instance and budget guards: `BUDGET_GUARD.json`, Vast container label, boot id, paid-hour deadline, GPU UUID lease | `host_guard()`: returns the same keys (instance id = `host:<hostname>`, hourly cost 0, GPU name and count), no time or cost limit | all |
| P2 | Start gates and QA receipts written by the historical queue (`START_GATE.json`, `CPU_QA.json` / `CUDA_QA.json`, `VERIFIED.json`, `CONTROL_MANIFEST.json`; H6 admission, smoke and dispatch receipts) | not required; the data assertions those gates protected run again in `train_host.py`: frame counts per sample, fold and report-group disjointness, Gold58 exclusion or inclusion as historically, sha of the fit-sample list, frozen targets and class weights, Adaptive input freeze, Joint36 frame table | all |
| P3 | Absolute paths recorded on the rented machines (cache roots in queue configs, the `cache_path` column of the Joint36 frame table) | paths from the command line; Joint36 frames relocated by SampleID through the cache manifests (`joint36_relocation`) | all; Joint36 |
| P4 | Byte identity of the cache *receipts* (`GLOBAL_RELEASE_RECEIPT.json` and per-shard `CACHE_COMPLETE.json`, which contain timings and paths) | only with `--cache-identity relaxed`: semantic identity against `training/cache336_reference` (same samples, sides, token counts, padded tokens, labels, builder sha, asset manifest sha, orientation state) | CoAtNet models |

Historical launcher configurations that the replacements read are kept in `training/host_configs/`
(`fold_recipes.json`, `adaptive_QUEUE_CONFIG.historical.json`, the Adaptive extension inventories).

`training/cache336_reference/` holds the historical cache *metadata* (release receipt, manifests, labels, per-shard
completion records) without pixels. `--cache-identity relaxed` compares a rebuilt cache against it, and
`tests/cpu_smoke.py` uses it to build a synthetic pixel cache that passes the original (strict) launcher checks.
