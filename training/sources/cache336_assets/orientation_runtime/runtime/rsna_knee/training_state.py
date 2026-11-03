"""Atomic epoch checkpoints and strict, relocatable input identity for resume."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from .constants import TARGET_COLUMNS

FORMAT_VERSION = 1


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic_write(path: str | Path, writer: Any) -> None:
    """Commit one file; a failed writer leaves the previous generation intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        Path(name).unlink(missing_ok=True)


def save_json(path: str | Path, value: Any) -> None:
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    atomic_write(path, lambda f: f.write(data))


def save_torch(path: str | Path, value: Any) -> None:
    import torch
    atomic_write(path, lambda f: torch.save(value, f))


def capture_rng(device: Any) -> dict[str, Any]:
    import torch
    numpy_state = np.random.get_state()
    state = {'python': random.getstate(),
             'numpy': [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
             'torch_cpu': torch.get_rng_state(), 'device_type': device.type}
    if device.type == 'cuda':
        state['torch_device'] = torch.cuda.get_rng_state_all()
    elif device.type == 'mps':
        state['torch_device'] = torch.mps.get_rng_state()
    return state


def restore_rng(state: dict[str, Any], device: Any) -> None:
    import torch
    if state['device_type'] != device.type:
        raise ValueError('Resume requires the same device type')
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state['torch_cpu'].cpu())
    if device.type == 'cuda':
        torch.cuda.set_rng_state_all([x.cpu() for x in state['torch_device']])
    elif device.type == 'mps':
        torch.mps.set_rng_state(state['torch_device'].cpu())


def make_training_identity(config: dict[str, Any], train_frame: pd.DataFrame,
                           valid_frame: pd.DataFrame, series: pd.DataFrame | None,
                           *, fold: int, device: Any) -> dict[str, Any]:
    import torch
    from importlib.metadata import PackageNotFoundError, version
    semantic = deepcopy(config)
    assets: dict[str, Any] = {}
    checkpoint = semantic.get('model', {}).get('encoder_checkpoint')
    if checkpoint:
        assets['encoder_checkpoint'] = file_sha256(checkpoint)
        semantic['model']['encoder_checkpoint'] = '<content-hash>'
    data = semantic['data']
    if data.get('cache_format') == 'full_source_bank_v1':
        from .full_source_cache import FullSourceCacheIndex

        cache = FullSourceCacheIndex(data['cache_index'],
                    require_complete=data.get('require_complete_cache', True),
                    expected_studies=int(data.get('expected_cache_studies', 4407)))
        assets['full_source_content_sha256'] = cache.index['content_sha256']
        data['cache_index'] = '<content-hash>'
        for key in ['cache_roots', 'cache_root', 'images_root']:
            data.pop(key, None)
    if data.get('cache_format') == 'grayscale_shards_v1':
        roots = data.get('cache_roots') or [data.get('cache_root')]
        if isinstance(roots, (str, bytes)):
            roots = [roots]
        records = []
        for root in roots:
            if not root:
                raise ValueError('Sharded resume identity requires cache roots')
            root = Path(root)
            complete = sorted(root.rglob('COMPLETED.json'))
            if not complete:
                raise ValueError(f'Cache has no completed shards: {root}')
            files = complete + sorted(root.glob('*.json'))
            records.append({str(f.relative_to(root)): file_sha256(f) for f in files})
        assets['cache_records'] = records
        assets['cache_integrity_scope'] = 'completion/part records; queue must verify payload hashes before starting'
        for key in ['cache_roots', 'cache_root', 'images_root']:
            data.pop(key, None)
    for key in ['centering_manifest', 'side_manifest']:
        if data.get(key):
            assets[key] = file_sha256(data[key])
            data[key] = '<content-hash>'
    def frame_hash(frame):
        if frame is None:
            return None
        return hashlib.sha256(frame.to_csv(index=False, lineterminator='\n').encode()).hexdigest()
    versions = {'torch': str(torch.__version__), 'numpy': np.__version__, 'pandas': pd.__version__}
    for name in ['timm', 'torchvision']:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    package = Path(__file__).parent
    accelerator = None
    if device.type == 'cuda':
        accelerator = {'cuda_version': torch.version.cuda,
                       'device_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    identity = {'format_version': FORMAT_VERSION, 'fold': fold, 'target_order': list(TARGET_COLUMNS),
                'semantic_config': semantic, 'assets': assets,
                'train_frame_sha256': frame_hash(train_frame), 'valid_frame_sha256': frame_hash(valid_frame),
                'series_frame_sha256': frame_hash(series), 'device_type': device.type,
                'versions': versions,
                'accelerator': accelerator,
                'code_sha256': {f.name: file_sha256(f) for f in sorted(package.glob('*.py'))},
                'torch_num_threads': torch.get_num_threads()}
    identity['sha256'] = json_sha256(identity)
    return identity


def load_training_state(path: str | Path, identity: dict[str, Any]) -> dict[str, Any]:
    import torch
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('format_version') != FORMAT_VERSION or state.get('kind') != 'epoch_training_state':
        raise ValueError('Expected an epoch training checkpoint, not legacy/inference-only weights')
    if state.get('identity') != identity:
        raise ValueError('Resume identity mismatch: fold/config/data/code/runtime must match')
    if state['next_epoch'] != len(state['history']):
        raise ValueError('Checkpoint epoch/history mismatch')
    if [x['epoch'] for x in state['history']] != list(range(state['next_epoch'])):
        raise ValueError('Checkpoint history is not contiguous')
    return state


def validate_best_artifacts(output_dir: Path, artifacts: dict[str, Any]) -> None:
    for item in artifacts.values():
        path = output_dir / item['name']
        if not path.resolve().is_relative_to(output_dir.resolve()):
            raise ValueError('Checkpoint artifact path escapes output directory')
        if file_sha256(path) != item['sha256']:
            raise ValueError(f'Checkpoint artifact hash mismatch: {path.name}')


def export_best_artifacts(output_dir: Path, artifacts: dict[str, Any]) -> None:
    """Publish aliases only for the committed checkpoint generation."""
    validate_best_artifacts(output_dir, artifacts)
    for source_key, targets in {'checkpoint': ['best.pt'],
                                'predictions': ['best_validation_predictions.csv', 'oof_predictions.csv']}.items():
        if source_key == 'predictions' and source_key not in artifacts:
            # A full-data fixed-epoch fit has no held-out predictions or OOF.
            continue
        source = output_dir / artifacts[source_key]['name']
        for target in targets:
            # Copy is portable and bounded to the selected generation.
            atomic_write(output_dir / target, lambda out, source=source: _copy(source, out))


def _copy(source: Path, out: Any) -> None:
    import shutil
    with source.open('rb') as inp:
        shutil.copyfileobj(inp, out, length=4 * 1024**2)
