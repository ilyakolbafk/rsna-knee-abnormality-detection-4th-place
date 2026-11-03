"""Identify the exact producer required by current anatomical input caches."""
import hashlib,json
from pathlib import Path
from functools import lru_cache
from .common import file_hash


def cache_matches(row,signature):
    """An explicitly verified bitwise-equivalent producer may reuse its bytes."""
    if signature in (row.get('producer_signature'),row.get('equivalent_producer_signature')):return True
    path=row.get('path')
    if not path:return False
    registry=Path(path).parent/'execution_equivalences.json'
    try:
        state=_state(registry)
        entries=_read_equivalences(str(registry),state)
        for entry in entries:
            if entry.get('source_signature')!=row.get('producer_signature') or entry.get('target_signature')!=signature:
                continue
            required=entry.get('required_row_fields',{})
            if any(row.get(key)!=value for key,value in required.items()):continue
            proof=Path(entry['evidence_path'])
            blob=_verified_document(str(proof),_state(proof),entry['evidence_sha256'])
            if 'adopted_rows' in blob:
                identity=f"{row.get('StudyInstanceUID')}|{row.get('SeriesInstanceUID')}"
                binding=blob['adopted_rows'].get(identity)
                if binding is None or any(row.get(key)!=value for key,value in binding.items()):continue
            if (blob.get('format')=='anatomy-execution-equivalence/v1'
                and blob.get('source_signature')==entry['source_signature']
                and blob.get('target_signature')==signature
                and blob.get('required_row_fields',{})==required
                and all(blob.get(k) is True for k in ('payload_arrays_equal','quality_and_registration_equal',
                                                     'inference_arithmetic_unchanged'))):return True
    except (OSError,ValueError,KeyError,json.JSONDecodeError):
        return False
    return False


def _state(path):
    value=Path(path).stat();return value.st_size,value.st_mtime_ns,value.st_ctime_ns


@lru_cache(maxsize=16)
def _read_equivalences(path,state):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('format')!='anatomy-execution-equivalences/v1':raise ValueError('unknown execution registry')
    return data['entries']


@lru_cache(maxsize=32)
def _verified_document(path,state,expected):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('execution equivalence proof changed')
    return json.loads(raw)


def producer_signature(weights,assets_dir=None,*,hash_function=None):
    weights=Path(weights);code=Path(__file__).parent
    assets=Path(assets_dir or (weights.parent.parent/'kneexnet' if weights.parent.name=='skm_tea' else weights.parent/'kneexnet'))
    files=[code/name for name in ('anatomy.py','anatomy_quality.py','anatomy_contract.py','geometry_bank.py',
        'ligament_atlas.py','series_registration.py','series_registration_rigid.py','data/ligament_atlas.json')]
    files += [code/name for name in ('geometry_runtime.py','anatomy_runtime.py','registration_runtime.py')
              if (code/name).is_file()]
    files += [weights,assets/'localizer.pt',assets/'best_model_256_noise_notebook_5_0.15.pth']
    hash_function=hash_function or file_hash
    payload={str(path.name):hash_function(path) for path in files}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
