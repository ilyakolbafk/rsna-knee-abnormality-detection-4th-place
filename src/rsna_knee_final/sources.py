"""Historical training packages: storage layout, staging and byte-level verification.

training/sources/SOURCE_TREES.json lists, for every historical package (alias), every file with its size and SHA-256
exactly as it was when the submitted checkpoint was trained. Files are stored in one of three ways:

  "local"    training/sources/<alias>/<path>                    (the file itself)
  "dedup"    training/sources/<other alias>/<path>              (byte-identical copy kept once in the repository)
  "weights"  <weights>/pretrained/<name>                        (large pretrained binaries, shipped with the weights)

stage(alias, destination, weights_dir) rebuilds the package byte for byte (pretrained files as symlinks) and verifies
every file against the manifest, so a host run always starts from exactly the code and assets that produced the model.
"""
import hashlib
import json
import os
import shutil
from pathlib import Path

from . import PACKAGE_ROOT

SOURCES = PACKAGE_ROOT / 'training' / 'sources'
MANIFEST = SOURCES / 'SOURCE_TREES.json'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def manifest():
    return json.loads(MANIFEST.read_text())


def stage(alias, destination, weights_dir, verify_large=None):
    """Materialize package <alias> into <destination> (must not exist) and verify it. Returns the destination.

    Pretrained binaries are checked by size and SHA-256; RSNA_STAGE_SKIP_PRETRAINED_SHA=1 keeps only the size check
    (used by tests/cpu_smoke.py with placeholder files)."""
    if verify_large is None:
        verify_large = os.environ.get('RSNA_STAGE_SKIP_PRETRAINED_SHA') != '1'
    tree = manifest()['trees'][alias]
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f'{destination} exists: every run starts from a fresh run directory')
    for rel, entry in sorted(tree['files'].items()):
        dst = destination / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        kind, where = entry['stored']
        if kind == 'weights':
            src = Path(weights_dir) / where
            if not src.is_file():
                raise FileNotFoundError(f'{alias}/{rel}: expected {src} (see weights/README.md)')
            if src.stat().st_size != entry['bytes']:
                raise ValueError(f'{alias}/{rel}: size of {src} differs from the historical file')
            if verify_large and sha256(src) != entry['sha256']:
                raise ValueError(f'{alias}/{rel}: SHA-256 of {src} differs from the historical file')
            os.symlink(src.resolve(), dst)
        else:
            src = SOURCES / where
            shutil.copyfile(src, dst)
            if dst.stat().st_size != entry['bytes'] or sha256(dst) != entry['sha256']:
                raise ValueError(f'{alias}/{rel}: staged copy differs from the historical file')
    extra = sorted(str(p.relative_to(destination)) for p in destination.rglob('*') if p.is_file() and str(p.relative_to(destination)) not in tree['files'])
    if extra:
        raise ValueError(f'unexpected files in staged package: {extra[:5]}')
    return destination


def verify_repository(weights_dir=None):
    """Check every stored source file (and, if weights_dir is given, every pretrained file) against the manifest."""
    problems, checked = [], 0
    for alias, tree in manifest()['trees'].items():
        for rel, entry in tree['files'].items():
            kind, where = entry['stored']
            if kind == 'weights':
                if weights_dir is None:
                    continue
                path = Path(weights_dir) / where
            else:
                path = SOURCES / where
            checked += 1
            if not path.is_file():
                problems.append(f'missing {alias}/{rel} ({path})')
            elif path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
                problems.append(f'content differs {alias}/{rel} ({path})')
    return checked, problems
