"""Read the frozen full-source .fsz bank without depending on its builder."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

FORMAT = 'rsna-knee-full-source-u8-v1'
INDEX_FORMAT = 'rsna-knee-full-source-training-index-v1'
MAX_BYTES = 256 * 2**20


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def read_full_source_archive(path, *, metadata_only=False):
    """Verify each requested member against its frozen SHA; no pickle loading."""
    arrays, metadata = {}, {}
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [i.filename for i in infos]
        if len(names) != len(set(names)) or any(Path(n).name != n for n in names):
            raise ValueError('Duplicate or unsafe full-source archive member')
        if sum(i.file_size for i in infos) > MAX_BYTES:
            raise ValueError('Full-source archive exceeds uncompressed size bound')
        marker = json.loads(archive.read('ARCHIVE.json'))
        if marker['format'] != FORMAT or set(marker['files']) != set(names)-{'ARCHIVE.json'}:
            raise ValueError('Full-source archive inventory mismatch')
        for name, record in marker['files'].items():
            if metadata_only and not name.endswith('.json'):
                continue
            payload = archive.read(name)
            if len(payload) != record['bytes'] or hashlib.sha256(payload).hexdigest() != record['sha256']:
                raise ValueError('Full-source member hash/byte mismatch')
            if name.endswith('.npy'):
                arrays[name[:-4]] = np.load(io.BytesIO(payload), allow_pickle=False)
            elif name.endswith('.json'):
                metadata[name[:-5]] = json.loads(payload)
            else:
                raise ValueError('Unknown full-source archive member type')
    return arrays, metadata, marker


def validate_study_arrays(arrays, metadata, *, expected_uid=None):
    images = arrays['images']
    n = len(images)
    if images.dtype != np.uint8 or images.shape != (n, 336, 336) or n == 0:
        raise ValueError('Expected nonempty uint8 [N,336,336] source images')
    if expected_uid is not None and metadata['binding']['StudyInstanceUID'] != expected_uid:
        raise ValueError('Full-source study UID mismatch')
    if metadata['binding']['source_slices'] != n:
        raise ValueError('Full-source slice count mismatch')
    for key in ('decode_valid', 'geometry_valid'):
        if arrays[key].shape != (n,) or arrays[key].dtype != np.bool_:
            raise ValueError('Invalid full-source validity mask')
    seen = np.zeros(n, bool)
    slots, indices = set(), set()
    for sid, series in metadata['series'].items():
        start, count, index = int(series['offset']), int(series['source_count']), int(series['series_index'])
        end = start + count
        if not 0 <= start < end <= n or seen[start:end].any() or index in indices:
            raise ValueError('Overlapping/invalid series extent or series index')
        indices.add(index); seen[start:end] = True
        if not np.array_equal(arrays['source_index'][start:end], np.arange(count)):
            raise ValueError('Original source indices must remain consecutive, including invalid planes')
        if not np.all(arrays['series_index'][start:end] == index):
            raise ValueError('Source series-index binding mismatch')
        slot = series['legacy_slot']
        if slot is not None:
            if not isinstance(slot, int) or not 0 <= slot < len(metadata['legacy_slots']) or slot in slots:
                raise ValueError('Invalid/duplicate semantic slot binding')
            if str(arrays['old_series_uids'][slot]) != sid:
                raise ValueError('Semantic slot differs from frozen source selection')
            slots.add(slot)
    if not seen.all():
        raise ValueError('Some source images are not bound to a series')


def index_identity(index):
    """Content identity survives moving the same bank to a rented machine."""
    payload = {k: index[k] for k in ('format', 'scope', 'contract_sha256', 'roster_sha256')}
    payload['studies'] = {uid: {k: r[k] for k in ('bytes', 'sha256')}
                          for uid, r in sorted(index['studies'].items())}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class FullSourceCacheIndex:
    def __init__(self, index_path, *, require_complete=True, expected_studies=4407):
        self.path = Path(index_path)
        self.index = json.loads(self.path.read_text())
        if self.index['format'] != INDEX_FORMAT:
            raise ValueError('Unknown full-source training index')
        if self.index['content_sha256'] != index_identity(self.index):
            raise ValueError('Full-source index identity mismatch')
        contract_digest = hashlib.sha256(json.dumps(self.index['build_contract'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if contract_digest != self.index['contract_sha256']:
            raise ValueError('Frozen preprocessing contract differs from bound hash')
        if require_complete and (self.index['scope'] != 'complete_train' or len(self.index['studies']) != expected_studies):
            raise ValueError('Training requires the fully downloaded and verified study bank')
        self.entries = self.index['studies']
        for uid, record in self.entries.items():
            path = self.file(uid)
            if not path.is_file() or path.stat().st_size != record['bytes']:
                raise ValueError(f'Full-source archive missing/incomplete: {uid}')
        self._verified = {}

    def __contains__(self, uid):
        return uid in self.entries

    def file(self, uid):
        return self.path.parent / self.entries[uid]['path']

    def load_study(self, uid):
        self.verify_archive(uid)
        arrays, metadata, _ = read_full_source_archive(self.file(uid))
        validate_study_arrays(arrays, metadata, expected_uid=uid)
        if metadata['binding']['contract_sha256'] != self.index['contract_sha256']:
            raise ValueError('Full-source build contract mismatch')
        return arrays, metadata

    def verify_archive(self, uid):
        """Hash in the parent before worker creation; recheck if file stat changes.

        Member SHA and structural validation still run on every load. This only
        preserves completed outer-archive verification across worker recreation.
        """
        path = self.file(uid)
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
        if self._verified.get(uid) != signature:
            if sha256_file(path) != self.entries[uid]['sha256']:
                raise ValueError('Full-source archive differs from frozen index')
            self._verified[uid] = signature
