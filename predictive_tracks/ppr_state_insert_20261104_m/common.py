"""New immutable namespace, fixed insertion grid, and conservative resource gate."""

from release_paths import runtime_python as _release_python, project_path as _release_path
import hashlib
import json
import os
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT/'predictive_tracks/ppr_position_controls_20261103_k/delivery_20261103_001/report.json'
PYTHON = _release_python('other')
MODELS = ('ocvp_seq','savi_transformer','vjepa2_vitl','videomaev2')
SLOTS = MODELS[:2]
ETAS = (.1,.5,1.)

def read(p):
    return json.loads(Path(p).read_text())

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def write(p,value,replace=False):
    p=Path(p)
    if not p.resolve().is_relative_to(HERE):raise ValueError('New experiment writes only')
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.atomic.{os.getpid()}.{time.time_ns()}')
    try:
        with tmp.open('x') as f:
            json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
        if replace:os.replace(tmp,p)
        else:os.link(tmp,p);tmp.unlink()
    finally:
        if tmp.exists():tmp.unlink()

def save(p,v):write(p,v)
def status(p,v):write(p,v,True)
def mode(model):return 'state_only' if model in SLOTS else 'pre_final_block'
def directory(model,eta):
    if model not in MODELS or eta not in (0.,)+ETAS:raise ValueError('Outside declared grid')
    return HERE/('audit' if eta==0 else 'runs')/model/mode(model)/f'eta_{round(eta*1000):04d}'

def verify(full=False):
    f=read(HERE/'frozen.json')
    for p,sha in f['files'].items():
        if digest(p)!=sha:raise RuntimeError('Frozen file changed: '+p)
    if full:
        for p,sha in f.get('large_files',{}).items():
            if digest(p)!=sha:raise RuntimeError('Original checkpoint changed: '+p)
        from predictive_tracks.ppr_position_controls_20261103_k.common import verify as previous
        previous(full=True)

class ResourceBusy(RuntimeError):pass

def priority_complete():
    p=read(HERE/'annotation.json');s=Path(p['rollout_path']).stat()
    return (p['task_complete_verified'] and p['turn_status']=='completed'
            and s.st_size==p['rollout_size'] and s.st_mtime_ns==p['rollout_mtime_ns'])

def gate(model,own_pid=None):
    from predictive_tracks.ppr_20261028.control import gpu_idle
    from predictive_tracks.ppr_existing_20261102.resource_resume_004.guarded import hardware
    h=hardware(own_pid);idle,gpu=gpu_idle(own_pid);priority=priority_complete()
    large=model not in SLOTS
    safe=(h['available_ram_mib'] >= (10240 if own_pid else 16384)
          and h['process_rss_mib'] <= (9216 if large else 2560)
          and h['gpu_free_mib'] >= (7168 if large and own_pid else 10240)
          and h['gpu_temperature_c']<=62 and h['cpu_temperature_c']<=70)
    d=dict(at=time.time(),priority_complete=priority,gpu_idle=idle,gpu=gpu,hardware=h,operational_limits_passed=safe)
    status(HERE/'monitor/gate.json',d)
    return priority and idle and safe,d

def require_resources(model):
    ok,d=gate(model,os.getpid())
    if not ok:raise ResourceBusy(json.dumps(d))
