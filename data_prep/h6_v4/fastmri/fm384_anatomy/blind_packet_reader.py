"""Read one PACKET_A-bound PNG into transient memory and emit bounded chunks.

No images are written locally or remotely. All source bytes are SHA256 checked.
The consumer passes the joined chunks to internal model image input only.
"""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parent
packet_bytes = (ROOT / 'PACKET_A.json').read_bytes()
assert hashlib.sha256(packet_bytes).hexdigest() == '822f202da30afc6f6dfdd0ecac995fd1f01b9d0b9da3adb0b2983d7c707bae86'
panel = json.loads(packet_bytes)['panels'][int(sys.argv[1])]
relative_path = panel['relative_path']
assert relative_path.startswith('review_validation/') and '..' not in relative_path
spec = importlib.util.spec_from_file_location('fmremote', ROOT.parent / 'fm384-v1-build-56967269-20261021' / 'remote.py')
remote = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote)
code = "import pathlib,hashlib,base64\np=pathlib.Path('/workspace/knee_v4_cache_20261021/fastmri_v1/anatomy_v1')/" + repr(relative_path) + "; b=p.read_bytes(); assert hashlib.sha256(b).hexdigest()==" + repr(panel['PNG_sha256']) + "; print(base64.b64encode(b).decode())"
encoded = remote.run(code, timeout=60).strip()
raw = base64.b64decode(encoded, validate=True)
assert hashlib.sha256(raw).hexdigest() == panel['PNG_sha256']
assert raw[:8] == b'\x89PNG\r\n\x1a\n'
width, height = struct.unpack('>II', raw[16:24])
print(json.dumps({'panel': relative_path, 'aliases': [c['alias'] for c in panel['cases']], 'PNG_sha256': panel['PNG_sha256'], 'PNG_bytes': len(raw), 'base64_chars': len(encoded), 'width': width, 'height': height, 'source_sha_verified_remote_and_local': True}), flush=True)
del raw
chunk_chars = 400000
for offset in range(0, len(encoded), chunk_chars):
    if offset:
        assert sys.stdin.readline(), 'Consumer closed before all chunks'
    print(encoded[offset:offset + chunk_chars], flush=True)
