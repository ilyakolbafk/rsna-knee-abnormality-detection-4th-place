#!/usr/bin/env python3
"""FINAL ens inference (Sixway + H6 SWA14-16, rank blend) -> submission.csv.

Two ways to point it at the inputs:

  1. Kaggle-style input root (datasets and the competition already mounted, e.g. inside a Kaggle notebook):
       python predict.py --input-root /kaggle/input --out /kaggle/working

  2. Local weights folder + competition-style data folder (sample_submission.csv, test.csv, test_series.csv, test_series/):
       python predict.py --weights /path/weights --data /path/rsna-knee-abnormality-detection --out /path/empty_out_dir
     This builds a symlink-only Kaggle-style layout under --staging (default /tmp/rsna_final_ens_input) and runs mode 1 on it.

Hardware: 2x NVIDIA T4 (the runtime asserts T4 devices, exactly as on Kaggle), Python 3.13 Kaggle image
(gcr.io/kaggle-private-byod/python@sha256:2757e0c7...); timm/pydicom/opencv come from the bundled offline wheels.
Run each prediction in a fresh container (clean /tmp); --out must be empty.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from rsna_knee_final import predict_pipeline as pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--settings', default=str(Path(__file__).resolve().parent / 'SETTINGS.json'))
    ap.add_argument('--input-root', help='Kaggle-style input root (mode 1)')
    ap.add_argument('--weights', help='weights folder (mode 2)')
    ap.add_argument('--data', help='competition-style data folder (mode 2)')
    ap.add_argument('--staging', default='/tmp/rsna_final_ens_input', help='symlink staging dir for mode 2')
    ap.add_argument('--out', help='empty output directory (default: SETTINGS.json predictions_dir)')
    ap.add_argument('--time-budget-hours', type=float, default=9.0, help='whole-run budget; H6 stops 25 min before it ends')
    ap.add_argument('--skip-weights-check', action='store_true', help='do not sha256 the weights before running')
    ap.add_argument('--blend', choices=['final', 'late_p4'], default='final',
                    help='stage-3 weights: final = FINAL ens 57532611 (default); late_p4 = late Sixway p4 lite 57543464')
    ap.add_argument('--sixway-only', action='store_true', help='stop after the Sixway stage (Sixway solo, 57291068)')
    a = ap.parse_args()
    settings = json.loads(Path(a.settings).read_text()) if Path(a.settings).exists() else {}
    out = Path(a.out or settings.get('predictions_dir', '/kaggle/working'))
    if a.input_root:
        root = Path(a.input_root)
    else:
        weights = a.weights or settings.get('weights_dir')
        data = a.data or settings.get('competition_data_dir')
        if not (weights and data):
            ap.error('give --input-root, or --weights and --data (or set weights_dir / competition_data_dir in SETTINGS.json)')
        root = pipeline.stage_input_root(weights, data, a.staging)
    pipeline.run(root, out, time_budget_hours=a.time_budget_hours, check_weights=not a.skip_weights_check,
                 blend_name=a.blend, sixway_only=a.sixway_only)


if __name__ == '__main__':
    main()
