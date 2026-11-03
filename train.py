#!/usr/bin/env python3
"""Train one of the 12 FINAL ens models with its historical code (see training/README.md).

  python train.py --list
  python train.py --check [--weights W]                       # verify every stored training source against its SHA-256
  python train.py --stage cache336_assets --out DIR --weights W   # materialize one historical package (e.g. the cache builder assets)

  python train.py --model fold0   --cache-root CACHE336 --weights W --runs RUNS
  python train.py --model core    --cache-root CACHE336 --weights W --runs RUNS
  python train.py --model mirror  --cache-root CACHE336 --weights W --runs RUNS
  python train.py --model adaptive --cache-root CACHE336 --extension-root EXT --test3-cache W/training_inputs/test3_cache --weights W --runs RUNS
  python train.py --model joint36 --cache-root CACHE336 --test3-cache W/training_inputs/test3_cache --weights W --runs RUNS
  python train.py --model h6_core6 --h6-inputs INPUTS.json --weights W --runs RUNS     # then tools/average_h6.py

Each run starts from a fresh <RUNS>/<model>/ directory: package/ is the historical package rebuilt byte for byte from
training/sources (verified), training/ receives the trainer's own outputs. The checkpoint to deploy is
training/averaged.pt (CoAtNet models; SWA of epochs 18-20) or the output of tools/average_h6.py (H6; SWA of EMA epochs 14-16).
--cache-identity original (default) requires the byte-identical historical cache release; use relaxed for a cache rebuilt
from DICOM with data_prep/cache336 (semantic checks against training/cache336_reference instead of receipt bytes).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from rsna_knee_final import sources, train_host  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--settings', default=str(Path(__file__).resolve().parent / 'SETTINGS.json'))
    ap.add_argument('--list', action='store_true', help='list the models and their historical packages')
    ap.add_argument('--check', action='store_true', help='verify training/sources (and pretrained files with --weights)')
    ap.add_argument('--stage', metavar='ALIAS', help='materialize one historical package into --out and exit')
    ap.add_argument('--out', help='destination for --stage')
    ap.add_argument('--model', choices=sorted(train_host.MODELS))
    ap.add_argument('--weights', help='weights folder (pretrained/ and training_inputs/ live there)')
    ap.add_argument('--runs', help='root for run directories')
    ap.add_argument('--cache-root', help='assembled cache336 release (GLOBAL_RELEASE_RECEIPT.json, TRAIN_MANIFEST.csv, ...)')
    ap.add_argument('--cache-identity', choices=['original', 'relaxed'], default='original')
    ap.add_argument('--extension-root', help='Adaptive extension cache (data_prep/extension_cache)')
    ap.add_argument('--test3-cache', help='cache of the 3 public sample studies (weights/training_inputs/test3_cache)')
    ap.add_argument('--h6-inputs', help='H6 INPUTS.json describing the RSNA/OAI/fastMRI v4 caches on this machine')
    ap.add_argument('--workers', type=int, help='data loader workers (default: historical value)')
    ap.add_argument('--smoke', action='store_true', help='H6 only: stop after two optimizer steps')
    ap.add_argument('--check-data', action='store_true', help='stage the package and run the launcher data checks only (CPU, no training)')
    a = ap.parse_args()
    settings = json.loads(Path(a.settings).read_text()) if Path(a.settings).exists() else {}
    weights = a.weights or settings.get('weights_dir')

    if a.list:
        for name, spec in train_host.MODELS.items():
            print(f"{name:10s} package={spec['alias']:9s} deployed={spec['checkpoint']}")
        return
    if a.check:
        checked, problems = sources.verify_repository(weights)
        print(json.dumps({'checked_files': checked, 'problems': problems}, indent=1))
        sys.exit(1 if problems else 0)
    if a.stage:
        if not (a.out and weights):
            ap.error('--stage needs --out and --weights')
        print(sources.stage(a.stage, a.out, weights))
        return
    if not a.model:
        ap.error('give --model (or --list / --check / --stage)')
    runs = a.runs or settings.get('runs_dir')
    cache = a.cache_root or settings.get('cache336_dir')
    if not (weights and runs):
        ap.error('--weights and --runs are required (or weights_dir / runs_dir in SETTINGS.json)')
    started = time.time()
    result = train_host.run(a.model, runs=runs, weights=weights, cache=cache, relaxed=a.cache_identity == 'relaxed',
                            extension_root=a.extension_root or settings.get('extension_cache_dir'),
                            test3_cache=a.test3_cache or (str(Path(weights) / 'training_inputs' / 'test3_cache')),
                            h6_inputs=a.h6_inputs or settings.get('h6_inputs_json'), workers=a.workers, smoke=a.smoke,
                            check_only=a.check_data)
    print(json.dumps({**result, 'wall_seconds': time.time() - started}, indent=1))


if __name__ == '__main__':
    main()
