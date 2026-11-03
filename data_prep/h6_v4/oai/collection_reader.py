"""Registry enforcement around the unchanged frozen NPZ/window reader."""
from reader import CaseReader
from training_reader import StudiesNPZ
from recipe_registry import validate_audit
import numpy as np

class CollectionReader(CaseReader):
    def read(self,case):
        value=super().read(case);config=validate_audit(case,value[2])
        with np.load(self.root/f'{case}.npz',allow_pickle=False) as z:
            assert str(z['case_id'])==case and str(z['roster_sha256'])==value[2]['roster_sha256'],'NPZ_CASE_OR_ROSTER_CONFLICT'
            assert str(z['format'])==config['format'],'NPZ_FORMAT_CONFLICT'
        return value

class CollectionStudiesNPZ(StudiesNPZ):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.reader=CollectionReader(self.reader.root,max_bytes=self.reader.max_bytes)
