"""Exact original4410 extension roster plus the three actual VM fullfit studies."""
from pathlib import Path
import csv
from adaptive_inputs import ExtensionIndex

class FullfitExtensionIndex(ExtensionIndex):
    def __init__(self, root):
        super().__init__(root)
        assert len(self.records) == 4410
        rows = list(csv.DictReader((Path(root)/'TEST3_EXT_MANIFEST.csv').open()))
        assert len(rows) == 3 and len({r['SampleID'] for r in rows}) == 3
        for row in rows:
            sid = row['SampleID']
            assert sid not in self.records
            self.records[sid] = self._path(row['cache_file'])
        assert len(self.records) == 4413
