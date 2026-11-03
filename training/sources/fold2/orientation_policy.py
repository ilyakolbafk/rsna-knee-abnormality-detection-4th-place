"""Approved, hash-pinned orientation release for new training and inference.

The deployed resolver stays byte-identical to the successful Kaggle version.
Export it into a fresh worker process: its historical imports are top-level.
This module never transforms images or relabels a cache by attaching a receipt.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Orientation artifact escapes its root')
    return path


def _read_pinned(path, expected):
    if file_sha256(path) != expected:
        raise ValueError(f'Orientation artifact hash mismatch: {path}')
    return json.loads(Path(path).read_text())


def verify_runtime(runtime, manifest):
    """Verify source and all four model weights, including unexpected files."""
    runtime = Path(runtime)
    expected = manifest['files']
    actual = {str(p.relative_to(runtime)) for p in runtime.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts}
    if actual != set(expected):
        raise ValueError('Orientation runtime file inventory mismatch')
    for name, digest in expected.items():
        if file_sha256(_inside(runtime, name)) != digest:
            raise ValueError(f'Orientation runtime hash mismatch: {name}')
    return True


def load_standard(workspace_root, *, training=False):
    root = Path(workspace_root).resolve()
    path = root / 'configs/knee_orientation_policy.json'
    policy = json.loads(path.read_text())
    manifest_path = _inside(root, policy['release_manifest'])
    manifest = _read_pinned(manifest_path, policy['release_manifest_sha256'])
    if policy['algorithm_sha256'] != manifest['files']['policy_v8.py']:
        raise ValueError('Orientation algorithm identity mismatch')
    runtime = _inside(root, policy['runtime_dir'])
    verify_runtime(runtime, manifest)
    if training:
        overlay_path = _inside(root, policy['training_overlay_manifest'])
        overlay = _read_pinned(overlay_path, policy['training_overlay_manifest_sha256'])
        overlay_root = _inside(
            root,
            policy.get(
                'training_overlay_dir',
                str((overlay_path.parent / 'training_overlay').relative_to(root)),
            ),
        )
        for name, digest in overlay['files'].items():
            if file_sha256(_inside(overlay_root, name)) != digest:
                raise ValueError(f'Training orientation overlay mismatch: {name}')
    return policy, manifest, runtime


def orientation_identity(policy):
    """Persist this in new run config, checkpoint IDENTITY and cache metadata."""
    return {
        'id': policy['id'],
        'release_manifest_sha256': policy['release_manifest_sha256'],
        'algorithm_sha256': policy['algorithm_sha256'],
        'canonical_target': policy['canonical_target'],
    }


def require_cache_orientation(cache_metadata, policy):
    """Call before a new-policy trainer reads already canonicalized tensors.

    The builder, after applying the adapter exactly once, records this identity
    and state. An old cache cannot pass just because shapes happen to match.
    Raw source banks are not canonical caches and must first pass the adapter.
    """
    if cache_metadata.get('orientation_policy') != orientation_identity(policy):
        raise ValueError('Cache orientation differs from the approved training default')
    if cache_metadata.get('orientation_state') != policy['training']['cache_state_required']:
        raise ValueError('Cache was not built as canonical L by the approved adapter')
    return True


def export_runtime(workspace_root, destination, *, training=False):
    """Stage verified code/weights; never export training labels into LB code."""
    policy, manifest, runtime = load_standard(workspace_root, training=training)
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Export destination must be empty; do not overwrite a frozen release')
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(runtime, destination / 'runtime',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    verify_runtime(destination / 'runtime', manifest)
    (destination / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    receipt = {
        'status': 'PASS',
        'orientation_policy': orientation_identity(policy),
        'consumer': 'training' if training else 'inference',
        'runtime_files': len(manifest['files']),
        'training_tables_exported': False,
        'image_cache_built': False,
        'entrypoints': ['resolver.LateralityResolver', 'raw_inputs.prepare_variants',
                        'h1_adapter.apply_tokens', 'raw_inputs.aggregate_knees'],
    }
    (destination / 'ORIENTATION_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt
