# Entry points

All commands run from the repository root. Paths default to `SETTINGS.json`.

1. `python predict.py --weights W --data DATA --out OUT` (or `--input-root /kaggle/input` inside Kaggle): FINAL ens
   `submission.csv`; `--blend late_p4` for the late p4 blend, `--sixway-only` for Sixway alone.
2. Put the three public pretrained files into `W/pretrained/` (sources and sha256 in `README.md`, Weights), then
   `python train.py --check --weights W`: verify the stored training sources and pretrained files (sha256).
3. Data preparation (`data_prep/README.md`): cache336 and the extension cache via Kaggle CPU notebooks and
   `data_prep/cache336/download_and_verify_cache.py --root CACHE336`; the H6 v4 caches via `data_prep/h6_v4/*` and an
   `INPUTS.json` for the training machine (`data_prep/h6_v4/README.md`).
4. `python train.py --model {fold0,fold1,fold2,fold3,fold4,core,repair,mirror,adaptive,joint36,h6_core6,h6_vm} ...`:
   train one deployed model (arguments per model in `training/README.md`); for H6 then
   `python tools/average_h6.py --run RUNS/h6_<arm> --out <arm>_swa_ema14_16.pt`.
5. `python tests/cpu_smoke.py --weights W --test3-cache W/training_inputs/test3_cache`: CPU check of the training side.
6. `python kaggle/host_check/build_check_kernel.py`: build the private Kaggle notebook that runs this repository.
