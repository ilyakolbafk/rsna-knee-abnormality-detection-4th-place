from pathlib import Path
import hashlib
import json
import os
import subprocess
import time
import stat
import tempfile
import uuid
import threading

RUN = Path(__file__).resolve().parents[1]
HANDOFF = RUN / 'handoff/oai_fastmri_handoff_56967269'
ROSTER = HANDOFF / 'inputs/OAI_SOURCE_ROSTER.frozen.json'
REMOTE = '/workspace/knee_v4_cache_20261021/oai_v4'
# GPU machine that receives the cache: set these for your own host.
SSH_HOST = os.environ.get('H6_CACHE_SSH_HOST', 'root@gpu-host')
SSH_PORT = os.environ.get('H6_CACHE_SSH_PORT', '22')
SSH_KEY = os.environ.get('H6_CACHE_SSH_KEY', str(Path.home() / '.ssh/id_ed25519'))
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
       '-o', 'IPQoS=none',
       '-o', f'UserKnownHostsFile="{RUN}/target_known_hosts"',
       '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=15',
       '-o', 'ServerAliveCountMax=3', '-i', SSH_KEY,
       '-p', SSH_PORT, SSH_HOST]

# Reuse only the tiny control requests. Pixel transfers retain one independent
# TCP connection per rsync worker, so multiplexing does not collapse the streams.
CONTROL_DIR = Path(tempfile.gettempdir()) / f'oai56967269-meta-{os.getuid()}'
META_SSH = SSH[:-1] + ['-o', 'ControlMaster=auto', '-o', 'ControlPersist=120',
                      '-o', f'ControlPath={CONTROL_DIR}/socket'] + SSH[-1:]
_META_LOCAL = threading.local()
_META_LOCK = threading.Lock()
_META_NEXT = 0

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()

def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.{uuid.uuid4().hex}.writing')
    with tmp.open('w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)

def latch_source_stop(record):
    import fcntl
    with (RUN/'source_stop.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=RUN/'SOURCE_STOP.json'
        if not path.exists():dump(path,record)

def remote_python(code, timeout=60):
    global _META_NEXT
    CONTROL_DIR.mkdir(mode=0o700, exist_ok=True)
    st = CONTROL_DIR.lstat()
    assert stat.S_ISDIR(st.st_mode) and st.st_uid == os.getuid() and not CONTROL_DIR.is_symlink()
    CONTROL_DIR.chmod(0o700)
    if not hasattr(_META_LOCAL,'slot'):
        with _META_LOCK:
            _META_LOCAL.slot=_META_NEXT%4;_META_NEXT+=1
    connection=SSH[:-1]+['-o','ControlMaster=auto','-o','ControlPersist=120',
                       '-o',f'ControlPath={CONTROL_DIR}/socket-{_META_LOCAL.slot}']+SSH[-1:]
    p = subprocess.run(connection + ['python3 -'], input=code, text=True,
                       capture_output=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(f'SSH_PYTHON_EXIT_{p.returncode}: {p.stderr[-1000:]}')
    lines = p.stdout.splitlines()
    return json.loads(next(x for x in reversed(lines) if x.startswith('{')))

def binding():
    b = json.loads((RUN/'TARGET_VERIFIED.json').read_text())
    assert b['provider']['id'] == 56967269 and b['provider']['machine_id'] == 1569
    assert b['hostkey_sha256'] == 'SHA256:6vzF4lwjdje7VVcngBszkx3IDy9PBIfYYr9C8cii9rg'
    return b

def roster():
    assert sha(ROSTER) == '0862f5a4ecd077422db8f7d4cb6dcf2b8e593c8eee5b1bbf8888cc15e4d86961'
    v = json.loads(ROSTER.read_text())
    assert v['source_pool_sha256'] == 'ec7a85668e3c0ae2caf445be7756f247e9970c6078008065e91dfff88e2527a8'
    assert len(v['rows']) == 3254
    assert len({(x['subjectkey'],x['side'],x['visit']) for x in v['rows']}) == 3254
    return v['rows']
