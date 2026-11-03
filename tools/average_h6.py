#!/usr/bin/env python3
"""Deployment checkpoint of an H6 arm: fixed SWA of the EMA weights of epochs 14, 15 and 16 (as deployed in FINAL ens).

  python tools/average_h6.py --run RUNS/h6_core6 --out core6_swa_ema14_16.pt
  python tools/average_h6.py --run RUNS/h6_vm    --out vm_swa_ema14_16.pt

--run is the run directory written by `train.py --model h6_core6` (or h6_vm): it holds package/ (the historical H6
package, staged and verified by train.py) and training/epoch14_ema.pt ... epoch16_ema.pt.

Same arithmetic as the historical averaging job (22.10): the H6 package's own ema_state.swa_update applied in epoch
order 14, 15, 16 (running mean of floating tensors, integer buffers keep the last epoch), saved with the same payload
keys. Given the same three EMA checkpoints the tensors are bit-identical to the deployed file; the file hash differs
because the payload records local paths and times (historical: core6 c9d916ff..., vm 9e096015..., configs/weights_manifest.json).
"""
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

EPOCHS = (14, 15, 16)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True, help='run directory of train.py --model h6_core6 / h6_vm')
    ap.add_argument('--out', required=True, help='output checkpoint (must not exist)')
    a = ap.parse_args()
    import torch
    run, out = Path(a.run), Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    spec = importlib.util.spec_from_file_location('h6_ema_state', run / 'package' / 'ema_state.py')
    ema_state = importlib.util.module_from_spec(spec)
    import sys
    sys.path.insert(0, str(run / 'package'))   # ema_state imports recovery_state from the package
    spec.loader.exec_module(ema_state)
    acc, count, binding, shapes, provenance, arm = None, 0, None, None, [], None
    for epoch in EPOCHS:
        source = run / 'training' / f'epoch{epoch:02d}_ema.pt'
        cp = torch.load(source, map_location='cpu', weights_only=False)
        assert cp['epoch'] == epoch and cp['kind'] == 'ema', (source, cp.get('epoch'), cp.get('kind'))
        if binding is None:
            binding, arm = cp['binding'], cp['arm']
            shapes = {n: (tuple(v.shape), v.dtype) for n, v in cp['model'].items()}
        else:
            assert cp['binding'] == binding and cp['arm'] == arm, 'EMA checkpoints come from different runs'
            assert shapes == {n: (tuple(v.shape), v.dtype) for n, v in cp['model'].items()}
        assert all(not (v.is_floating_point() or v.is_complex()) or torch.isfinite(v).all() for v in cp['model'].values())
        acc, count = ema_state.swa_update(acc, count, cp['model'])
        provenance.append({'epoch': epoch, 'source': str(source), 'sha256': sha256(source), 'bytes': source.stat().st_size})
        del cp
    assert count == len(EPOCHS)
    payload = {'model': acc, 'epochs': list(EPOCHS), 'averaging_source': 'EMA', 'arm': arm, 'package_sha256': binding['package_sha256'],
               'binding': binding, 'derived_recipe': 'fixed_SWA_EMA14_15_16_v1', 'averaging_code_sha256': sha256(run / 'package' / 'ema_state.py'),
               'source_checkpoints': provenance, 'resume_checkpoint': False}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(json.dumps({'out': str(out), 'sha256': sha256(out), 'arm': arm, 'epochs': list(EPOCHS), 'at_unix': time.time()}, indent=1))


if __name__ == '__main__':
    main()
