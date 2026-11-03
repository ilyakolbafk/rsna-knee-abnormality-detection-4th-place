"""Bounded storage reader for the actual RSNA public-v4 NPZ protocol.

No orientation transform is performed here. The explicit producer basis is
preserved. Production admission remains closed by the separate semantic gate.
"""
from pathlib import Path
import copy, hashlib, io, json, os, stat, zipfile
import numpy as np
from rsna_public_npz import inspect, require_training_release, need, STATE
from full_cache_loader import Reader as PublicReader

def fingerprint(s):
    return [s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_dev, s.st_ino]

def immutable(array):
    return np.frombuffer(array.tobytes(order='C'), dtype=array.dtype).reshape(array.shape)

class TransportStudyReader:
    def __init__(self, bundle, *, purpose='training', acceptance=None, verify_target=True):
        self.bundle = copy.deepcopy(bundle)
        self.assessment = inspect(self.bundle)
        if purpose != 'protocol_probe':
            need(purpose == 'training', 'PURPOSE')
            require_training_release(self.assessment, acceptance)
        self.purpose = purpose
        self.root = Path(bundle['root'])
        need(self.root.is_absolute() and not self.root.is_symlink(), 'ROOT')
        if verify_target:
            need(os.uname().nodename == bundle['target']['hostname'] == '56dc3869206e', 'LIVE_TARGET')
            need(Path('/root/.vast_containerlabel').read_text().strip() == 'C.56967269', 'LIVE_CONTAINER')
            need(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == bundle['target']['boot_id'], 'LIVE_BOOT')
        self.sample_id = self.assessment['sample_id']
        self.orientation_state = STATE
        self.expected = {}
        source = bundle['files']
        receiver = source['.receiver_verified.json']['document']
        members = receiver['verified_files']
        devices = {v['stat']['device'] for v in members.values()}
        need(len(devices) == 1, 'BATCH_DEVICE')
        device = next(iter(devices))
        for rel, entry in source.items():
            m = members.get(rel)
            dev = m['stat']['device'] if m else device
            self.expected[rel] = {'stat': entry['stat'] + [dev, entry['inode']], 'sha256': entry['sha256']}
        rel = self.assessment['payload_path']; payload = self.assessment['payload_destination_binding']; s = payload['stat']
        self.expected[rel] = {'stat': [s['size'], s['mtime_ns'], s['ctime_ns'], s['device'], s['inode']], 'sha256': payload['sha256']}
        # Metadata bytes remain SHA checked; pixels rely on the accepted transport
        # full-SHA receipt plus unchanged exact destination stat/dev/inode.
        for rel, record in source.items():
            actual = json.loads(self._metadata(rel))
            need(actual == record['document'], 'METADATA_DOCUMENT_CHANGED')
        self.side = next(v['document'] for k, v in source.items() if '/studies/' in k)
        self.pid = None; self.cache = None

    def __getstate__(self):
        result = self.__dict__.copy(); result['cache'] = None; result['pid'] = None
        return result

    def _path(self, rel):
        relative = Path(rel)
        need(not relative.is_absolute() and '..' not in relative.parts and rel in self.expected, 'PATH_MEMBERSHIP')
        need(not any(p.is_symlink() for p in [self.root, *self.root.parents]), 'ROOT_SYMLINK')
        p = self.root
        for part in relative.parts:
            p = p / part
            need(not p.is_symlink(), 'SYMLINK')
        need(p.resolve().is_relative_to(self.root.resolve()), 'PATH_ESCAPE')
        s = p.stat()
        need(stat.S_ISREG(s.st_mode) and fingerprint(s) == self.expected[rel]['stat'], 'DESTINATION_STAT')
        return p

    def _metadata(self, rel):
        p = self._path(rel)
        with p.open('rb') as f:
            need(fingerprint(os.fstat(f.fileno())) == self.expected[rel]['stat'], 'OPEN_STAT')
            wire = f.read()
            need(fingerprint(os.fstat(f.fileno())) == self.expected[rel]['stat'], 'READ_STAT')
        self._path(rel)
        need(hashlib.sha256(wire).hexdigest() == self.expected[rel]['sha256'], 'METADATA_SHA')
        return wire

    def _arrays(self):
        if self.pid != os.getpid():
            self.pid = os.getpid(); self.cache = None
        rel = self.assessment['payload_path']; p = self._path(rel)
        if self.cache is not None:
            return self.cache
        declared = self.side['payload']['arrays']; count = self.assessment['rows']
        need(0 < count <= 3500, 'DECODE_ROW_BOUND')
        amount = count * (384 * 384 + 12 * 4)
        need(amount <= 512 * 2**20, 'DECODE_BYTE_BOUND')
        with p.open('rb') as f:
            need(fingerprint(os.fstat(f.fileno())) == self.expected[rel]['stat'], 'OPEN_STAT')
            with zipfile.ZipFile(f) as archive:
                entries = archive.infolist()
                need(len(entries) == 2 and {e.filename for e in entries} == {'px.npy', 'meta.npy'}, 'NPZ_MEMBERS')
                bounds = {'px.npy': count * 384 * 384, 'meta.npy': count * 12 * 4}
                for entry in entries:
                    need(not entry.flag_bits & 1 and entry.compress_type in [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED] and bounds[entry.filename] <= entry.file_size <= bounds[entry.filename] + 1024, 'NPZ_DECOMPRESSED_BOUND')
                    # Validate the NPY header before np.load can allocate from it.
                    with archive.open(entry) as member:
                        version = np.lib.format.read_magic(member)
                        need(version in [(1, 0), (2, 0)], 'NPY_VERSION')
                        width = 2 if version == (1, 0) else 4
                        length_wire = member.read(width)
                        need(len(length_wire) == width, 'NPY_HEADER_LENGTH')
                        length = int.from_bytes(length_wire, 'little')
                        need(0 < length <= 4096, 'NPY_HEADER_LENGTH')
                        header = member.read(length)
                        need(len(header) == length, 'NPY_HEADER_LENGTH')
                        parse = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                        shape, fortran, dtype = parse(io.BytesIO(length_wire + header), max_header_size=4096)
                        name = entry.filename.removesuffix('.npy')
                        wanted_dtype = np.dtype(np.uint8 if name == 'px' else np.float32)
                        need(list(shape) == declared[name]['shape'] and dtype == wanted_dtype and not dtype.hasobject and fortran is False, 'NPY_HEADER_SCHEMA')
                        need(8 + width + length + bounds[entry.filename] == entry.file_size, 'NPY_LENGTH')
            f.seek(0)
            with np.load(f, allow_pickle=False) as archive:
                px, meta = archive['px'], archive['meta']
            need(fingerprint(os.fstat(f.fileno())) == self.expected[rel]['stat'], 'DECODE_STAT')
        self._path(rel)
        need(px.dtype == np.uint8 and list(px.shape) == declared['px']['shape'] and meta.dtype == np.float32 and list(meta.shape) == declared['meta']['shape'] and np.isfinite(meta).all(), 'DECODED_ARRAY_SCHEMA')
        PublicReader.validate_blocks(self.side['blocks'], meta, len(px))
        self.cache = (immutable(px), immutable(meta))
        return self.cache

    def study(self, key):
        need(key == self.sample_id, 'READ_SAMPLE_ID')
        px, meta = self._arrays()
        def take(ids):
            self._path(self.assessment['payload_path'])
            values = np.asarray(ids)
            need(values.dtype.kind in 'iu' and np.all(values >= 0) and np.all(values < len(px)), 'ROW_INDICES')
            return px[values]
        return copy.deepcopy(self.side['blocks']), meta, take

    def windows(self, key, **kwargs):
        # Exactly the existing public sampler and nine-column metadata projection.
        return PublicReader.windows(self, key, **kwargs)
