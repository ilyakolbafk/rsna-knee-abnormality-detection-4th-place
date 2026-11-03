#!/usr/bin/env python3
"""Export the current orientation standard, or verify a new training cache."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rsna_knee.orientation_policy import (
    export_runtime, load_standard, orientation_identity, require_cache_orientation,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['training', 'inference'], required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--cache-metadata', type=Path)
    args = parser.parse_args()
    policy, _, _ = load_standard(ROOT, training=args.mode == 'training')
    if args.cache_metadata:
        if args.mode != 'training':
            parser.error('--cache-metadata is a training prelaunch check')
        require_cache_orientation(json.loads(args.cache_metadata.read_text()), policy)
    result = {'status': 'PASS', 'orientation_policy': orientation_identity(policy),
              'cache_checked': bool(args.cache_metadata)}
    if args.output_dir:
        result['export'] = export_runtime(ROOT, args.output_dir, training=args.mode == 'training')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
