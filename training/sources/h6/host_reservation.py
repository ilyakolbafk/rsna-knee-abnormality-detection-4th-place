from pathlib import Path
import os,json,time
STORE=Path('/workspace/knee_v4_cache_20261021/oai_v4')
SHARED=STORE.parent/'shared'
KEY='h6_fullcache_pair_training_20261021_1310'
G=2**30
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);t=p.with_name(p.name+'.tmp.'+str(os.getpid()))
 with t.open('w') as f:json.dump(v,f);f.flush();os.fsync(f.fileno())
 os.replace(t,p)
 fd=os.open(str(p.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
def reserve(active):
    import fcntl, shutil
    with (SHARED / 'admission.lock').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        d = read(SHARED / 'admission.json'); assert d['target']['instance_id'] == 56967269
        before = d['tasks'].get(KEY, {})
        assert not before.get('active') or before.get('PID') == os.getpid(), 'OWN_RESERVATION_ACTIVE'
        peers = [v for k, v in d['tasks'].items() if k != KEY]; limits = d['limits']
        if active:
            assert sum(v.get('cpu_threads', 0) for v in peers) + 14 <= min(limits['max_total_cpu_threads'], 16), 'CPU_ADMISSION'
            assert shutil.disk_usage(STORE).free >= limits['disk_reserve_bytes'] + sum(v.get('reserved_disk_bytes', 0) for v in peers) + 40*G, 'DISK_ADMISSION'
            mi = {a: int(b.split()[0]) * 1024 for a, b in (z.split(':', 1) for z in Path('/proc/meminfo').read_text().splitlines())}
            available = mi['MemAvailable']; cg = Path('/sys/fs/cgroup/memory')
            if (cg / 'memory.limit_in_bytes').exists():
                st = dict(z.split() for z in (cg / 'memory.stat').read_text().splitlines())
                available = min(available, int((cg / 'memory.limit_in_bytes').read_text()) - int((cg / 'memory.usage_in_bytes').read_text()) + int(st.get('inactive_file', '0')))
            elif Path('/sys/fs/cgroup/memory.max').exists() and Path('/sys/fs/cgroup/memory.max').read_text().strip() != 'max':
                cg = Path('/sys/fs/cgroup'); st = dict(z.split() for z in (cg / 'memory.stat').read_text().splitlines())
                available = min(available, int((cg / 'memory.max').read_text()) - int((cg / 'memory.current').read_text()) + int(st.get('inactive_file', '0')))
            assert available >= max(24 * G, limits['memory_reserve_bytes']) + sum(v.get('reserved_ram_bytes', 0) for v in peers) + 32 * G, 'RAM_ADMISSION'
            full = next(z for z in Path('/proc/pressure/memory').read_text().splitlines() if z.startswith('full '))
            assert float(dict(z.split('=') for z in full.split()[1:])['avg10']) < 1, 'PSI_ADMISSION'
        d['tasks'][KEY] = {'run_id': KEY, 'PID': os.getpid(), 'ctime': Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19], 'active': active, 'cpu_threads': 14 if active else 0, 'download_workers': 0, 'convert_workers': 0, 'reserved_disk_bytes': 40*G if active else 0, 'reserved_ram_bytes': 32 * G if active else 0, 'gpu_indices': [0,1] if active else [], 'phase': 'two_fullH6_fits_main3threads_plus4loaderworkers_each' if active else 'terminal_no_training', 'heartbeat_unix': time.time()}
        write(SHARED / 'admission.json', d)
