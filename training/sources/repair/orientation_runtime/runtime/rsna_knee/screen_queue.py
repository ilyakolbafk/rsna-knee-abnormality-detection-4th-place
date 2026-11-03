"""Bounded paired-arm screen; run IDs include arm, fold and seed."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd
import yaml

from rsna_knee.constants import TARGET_COLUMNS
from rsna_knee.training_state import file_sha256, json_sha256, save_json


def verify_completion(folder: Path, *, fold: int, expected_ids: set[str], epochs: int,
                      expected_identity_sha256: str | None = None) -> dict:
    marker = json.loads((folder / 'COMPLETED.json').read_text())
    if marker['fold'] != fold or marker['completed_epochs'] != epochs:
        raise ValueError('Completion marker fold/epoch mismatch')
    if expected_identity_sha256 is not None and marker['training_identity_sha256'] != expected_identity_sha256:
        raise ValueError('Completed fold does not belong to the frozen queue identity')
    required = {'best.pt', 'oof_predictions.csv', 'summary.json', 'history.json',
                'training_identity.json', 'resolved_config.yaml'}
    if not required <= set(marker['files']):
        raise ValueError('Incomplete completion manifest')
    for name, details in marker['files'].items():
        path = folder / name
        if Path(name).name != name or path.is_symlink():
            raise ValueError('Unsafe completion artifact name')
        if path.stat().st_size != details['bytes'] or file_sha256(path) != details['sha256']:
            raise ValueError(f'Artifact integrity failure: {name}')
    summary = json.loads((folder / 'summary.json').read_text())
    if summary['status'] != 'complete' or summary['completed_epochs'] != epochs or summary['fold'] != fold:
        raise ValueError('Summary is not a completed fold')
    if summary['training_identity_sha256'] != marker['training_identity_sha256']:
        raise ValueError('Training identity mismatch')
    identity = json.loads((folder / 'training_identity.json').read_text())
    if identity.get('sha256') != marker['training_identity_sha256']:
        raise ValueError('Identity artifact does not match completion marker')
    identity_body = {k: v for k, v in identity.items() if k != 'sha256'}
    if json_sha256(identity_body) != identity['sha256']:
        raise ValueError('Identity semantic hash is invalid')
    history = json.loads((folder / 'history.json').read_text())
    if [x['epoch'] for x in history] != list(range(epochs)):
        raise ValueError('Incomplete epoch history')
    oof = pd.read_csv(folder / 'oof_predictions.csv', dtype={'StudyInstanceUID': str})
    if not oof.StudyInstanceUID.is_unique or set(oof.StudyInstanceUID) != expected_ids:
        raise ValueError('OOF study set mismatch')
    if not (oof.fold == fold).all():
        raise ValueError('OOF fold mismatch')
    predictions = oof[[f'pred__{t}' for t in TARGET_COLUMNS]]
    if not (predictions.ge(0) & predictions.le(1)).all().all():
        raise ValueError('OOF probability is missing, nonfinite or out of bounds')
    return marker


MANDATORY = [('B', 3, 20260904), ('L16', 3, 20260904),
             ('B', 0, 20260904), ('L16', 0, 20260904), ('B', 3, 20260914)]


def run_id(arm: str, fold: int, seed: int) -> str:
    return f'{arm.lower()}-f{fold}-s{seed}'


def validate_design(contract: dict, *, now: float | None = None) -> None:
    """Fail closed before touching any output, including repeated fold3 runs."""
    runs = contract['runs']
    if [(r['arm'], r['fold'], r['seed']) for r in runs] != MANDATORY:
        raise ValueError('Queue must contain the five mandatory runs in frozen order')
    ids = [r['run_id'] for r in runs]
    if ids != [run_id(*r) for r in MANDATORY] or len(set(ids)) != len(ids):
        raise ValueError('Run IDs must uniquely bind arm/fold/seed')
    start = datetime.fromisoformat(contract['rental_started_at_utc']).timestamp()
    deadline = datetime.fromisoformat(contract['stop_deadline_utc']).timestamp()
    if not 0 < deadline - start <= 8 * 3600:
        raise ValueError('Rental exceeds eight paid hours')
    if now is not None and now >= deadline:
        raise ValueError('Rental deadline passed')
    rate = contract['hourly_rate_usd']
    reserves = contract['traffic_reserve_usd'] + contract['storage_reserve_usd']
    if rate <= 0 or reserves < 0 or not 0 < contract['budget_usd'] <= 20:
        raise ValueError('Invalid cost bounds')
    if (deadline - start) / 3600 * rate + reserves > contract['budget_usd']:
        raise ValueError('Cost bound exceeds funded budget')
    for r in runs:
        if r['estimated_seconds'] <= 0 or len(r['identity_sha256']) != 64:
            raise ValueError('Run lacks measured forecast or identity')
    if contract['post_training_reserve_seconds'] < 1800 or contract['backup_seconds_per_run'] < 120:
        raise ValueError('Diagnostics and backup reserves cannot be removed')


def required_remaining_seconds(contract: dict, pending: list[dict], slowdown: float = 1.) -> float:
    return (sum(r['estimated_seconds'] * max(1., slowdown) for r in pending)
            + len(pending) * contract['backup_seconds_per_run']
            + contract['post_training_reserve_seconds'])


def load_contract(path: Path) -> dict:
    contract = json.loads(path.read_text())
    validate_design(contract, now=time.time())
    for name, digest in contract['input_files'].items():
        if file_sha256(name) != digest:
            raise ValueError(f'Frozen input changed: {name}')
    import torch
    from .train import build_fold_partitions
    from .training_state import make_training_identity
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Screen requires exactly one CUDA GPU')
    frame = pd.read_csv(contract['manifest'], dtype={'StudyInstanceUID': str})
    if len(frame) != 4407 or not frame.StudyInstanceUID.is_unique:
        raise ValueError('Unexpected manifest')
    common_recipe = None
    for run in contract['runs']:
        cfg = yaml.safe_load(Path(run['config']).read_text())
        tr = cfg['train']
        if (cfg['seed'] != run['seed'] or cfg['data']['image_size'] != 224 or
                tr['epochs'] != 12 or tr['max_train_tokens'] != 46 or
                tr['batch_size'] * tr['accumulate_steps'] != 8 or
                tr.get('data_rng_mode') != 'study_epoch_v1'):
            raise ValueError('Run differs from paired-screen recipe')
        local = cfg['model'].get('local_head')
        if (run['arm'] == 'L16') != (local == 'l16'):
            raise ValueError('Arm/head mismatch')
        comparable = json.loads(json.dumps(cfg))
        comparable.pop('seed')
        for field in ('local_head', 'local_gate_init'):
            comparable['model'].pop(field, None)
        if common_recipe is None:
            common_recipe = comparable
        elif comparable != common_recipe:
            raise ValueError('B/L or seed repeat changes non-arm/non-seed recipe')
        train, valid = build_fold_partitions(frame, fold=run['fold'], n_splits=5,
                                            validation_scope='all_holdout')
        actual = make_training_identity(cfg, train, valid, None, fold=run['fold'], device=torch.device('cuda'))
        if actual['sha256'] != run['identity_sha256']:
            raise ValueError('Runtime identity differs from frozen screen')
    return contract


def verify_ack(path: Path, contract_sha: str, run: dict, marker_sha: str) -> None:
    receipt = json.loads(path.read_text())
    expected = dict(contract_sha256=contract_sha, run_id=run['run_id'],
                    training_identity_sha256=run['identity_sha256'], completion_sha256=marker_sha)
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError('Backup ACK is from another arm, seed or generation')


def verify_diagnostics_ack(receipt: dict, contract: dict, contract_sha: str) -> None:
    expected = {}
    for run in contract['runs']:
        if run['arm'] != 'L16': continue
        folder = Path(contract['diagnostics_root']) / ('fold_' + str(run['fold']))
        expected[run['run_id']] = dict(
            manifest_sha256=file_sha256(folder / 'DIAGNOSTICS_COMPLETE.json'),
            training_identity_sha256=run['identity_sha256'],
            checkpoint_sha256=file_sha256(Path(contract['output_root']) / run['run_id'] / 'best.pt'))
    if (receipt.get('contract_sha256') != contract_sha or receipt.get('verified') is not True or
            receipt.get('manifests') != expected):
        raise ValueError('Diagnostic backup acknowledgement is not bound to both current L16 artifacts')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract', type=Path, required=True)
    args = parser.parse_args()
    contract = load_contract(args.contract)
    contract_sha = file_sha256(args.contract)
    root = Path(contract['output_root'])
    runtime = root / 'queue'
    runtime.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(contract['manifest'], dtype={'StudyInstanceUID': str})
    deadline = datetime.fromisoformat(contract['stop_deadline_utc']).timestamp()
    slowdown = 1.
    active_run_id = None
    def status(state, **kw):
        save_json(runtime / 'status.json', dict(state=state, contract_sha256=contract_sha,
                  at_utc=datetime.now(timezone.utc).isoformat(), **kw))
    with (runtime / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for index, run in enumerate(contract['runs']):
                active_run_id = run['run_id']
                folder = root / run['run_id']
                if not (folder / 'COMPLETED.json').exists():
                    pending = [r for r in contract['runs'][index:]
                               if not (root / r['run_id'] / 'COMPLETED.json').exists()]
                    required = required_remaining_seconds(contract, pending, slowdown)
                    if deadline - time.time() < required:
                        raise TimeoutError('Unfinished mandatory screen no longer fits; preserve seed/diagnostic reserve')
                    folder.mkdir(exist_ok=True)
                    command = [sys.executable, '-m', 'rsna_knee.cli', 'train', '--config', run['config'],
                               '--manifest-csv', contract['manifest'], '--fold', str(run['fold']),
                               '--output-dir', str(folder), '--device', 'cuda']
                    if (folder / 'last_training.pt').exists():
                        command += ['--resume', str(folder / 'last_training.pt')]
                    status('training', run_id=run['run_id'], fold=run['fold'], arm=run['arm'])
                    started = time.time()
                    with (folder / 'training.log').open('ab') as log:
                        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                                       timeout=max(1, deadline-time.time()-contract['post_training_reserve_seconds']))
                    slowdown = max(slowdown, (time.time()-started)/run['estimated_seconds'])
                expected_ids = set(frame.loc[frame.fold == run['fold'], 'StudyInstanceUID'])
                verify_completion(folder, fold=run['fold'], epochs=12, expected_ids=expected_ids,
                                  expected_identity_sha256=run['identity_sha256'])
                marker_sha = file_sha256(folder / 'COMPLETED.json')
                status('awaiting_backup', run_id=run['run_id'], completion_sha256=marker_sha)
                ack = runtime / f"backup_ack_{run['run_id']}.json"
                while not ack.exists():
                    if time.time() >= deadline - contract['post_training_reserve_seconds']:
                        raise TimeoutError('Backup acknowledgement missing before reserved deadline')
                    time.sleep(5)
                verify_ack(ack, contract_sha, run, marker_sha)
            # Exporters are fixed in the contract, with scripts/config/selection hashes.
            status('diagnostics')
            for command in contract['diagnostic_commands']:
                subprocess.run(command, check=True, timeout=max(1, deadline-time.time()-600))
            status('awaiting_diagnostics_backup')
            ack_path = runtime / 'diagnostics_backup_ack.json'
            while not ack_path.exists():
                if time.time() >= deadline - 120:
                    raise TimeoutError('Diagnostics backup not acknowledged')
                time.sleep(5)
            ack = json.loads(ack_path.read_text())
            verify_diagnostics_ack(ack, contract, contract_sha)
            status('complete')
        except Exception as exc:
            status('failed', active_run_id=active_run_id, error_type=type(exc).__name__, error=str(exc))
            raise


if __name__ == '__main__':
    main()
