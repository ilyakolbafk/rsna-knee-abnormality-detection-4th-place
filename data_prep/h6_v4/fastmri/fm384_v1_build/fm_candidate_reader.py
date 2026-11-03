"""Guarded reader. The stock loader's advisory JSON flags do not enforce readiness."""
from pathlib import Path
import json
from fm_loader_reference import FmStore

def load_store(path,*,review_geometry_only=False,min_volumes=1000):
 p=Path(path);index=json.loads((p/'index.json').read_text())
 if not review_geometry_only and (index.get('training_ready') is not True or index.get('semantic_gate')!='PASS'):
  raise RuntimeError('CANDIDATE_NOT_PROMOTED_ANATOMICAL_ML_UNVERIFIED')
 return FmStore(str(p),min_volumes=min_volumes)
