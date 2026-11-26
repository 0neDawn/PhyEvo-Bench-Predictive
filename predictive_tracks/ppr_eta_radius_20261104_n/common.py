"""New namespace and authorized finite eta/radius grid, no old disk writes."""

from release_paths import runtime_python as _release_python, project_path as _release_path
import hashlib
import json
import os
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=ROOT/'predictive_tracks/ppr_state_insert_20261104_m/delivery_20261104_001/report.json'
PYTHON=_release_python('other')
MODELS=('ocvp_seq','savi_transformer','videomaev2')
SLOTS=MODELS[:2]
SLOT_ETAS=(.005,.01,.025,.05,.1,.2,.5,1.)
SLOT_ALPHAS=(1.,1.25,1.5,2.)
STATE_ETAS=(.005,.01,.025,.05)
STATE_ALPHAS=(1.,1.25,2.)
VIDEO_ETAS=(.001,.0025,.005,.01,.015,.02,.025,.03,.04,.05,.075,.1,.15,.2,.25,.5,.75,1.)
VIDEO_ALPHAS=(1.,1.05,1.1,1.25,1.5,1.75,2.)
CURRENT={}

def read(p):return json.loads(Path(p).read_text())
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def write(p,value,replace=False):
    p=Path(p)
    if not p.resolve().is_relative_to(HERE):raise ValueError('Writes restricted to new N namespace')
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.atomic.{os.getpid()}.{time.time_ns()}')
    try:
        with tmp.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
        if replace:os.replace(tmp,p)
        else:os.link(tmp,p);tmp.unlink()
    finally:
        if tmp.exists():tmp.unlink()
def save(p,v):write(p,v)
def status(p,v):write(p,v,True)
def setting_id(m,mode,e,a):return f'{m}_{mode}_a{round(a*1000000):07d}_e{round(e*1000000):07d}'
def setting(m,mode,e,a):return dict(model=m,mode=mode,eta=float(e),alpha=float(a),config_id=setting_id(m,mode,e,a))
def path(s):return HERE/'runs'/s['model']/s['mode']/s['config_id']
def mode(model):
    if CURRENT['model']!=model:raise RuntimeError('Current model differs')
    return CURRENT['mode']
def directory(model,eta):
    if model!=CURRENT['model']:raise RuntimeError('Wrong model binding')
    if eta==0:return HERE/'audit'/model/CURRENT['mode']
    if eta!=CURRENT['eta']:raise RuntimeError('Wrong eta binding')
    return path(CURRENT)
def verify(full=False):
    f=read(HERE/'frozen.json')
    for p,w in f['files'].items():
        if digest(p)!=w:raise RuntimeError('Frozen N/source file changed: '+p)
    if (HERE/'STAGE2_FROZEN.json').exists():
        for p,w in read(HERE/'STAGE2_FROZEN.json')['files'].items():
            if digest(p)!=w:raise RuntimeError('Frozen Stage2 proposal changed: '+p)
    if full:
        from predictive_tracks.ppr_state_insert_20261104_m.common import verify as previous
        previous(full=True)

class ResourceBusy(RuntimeError):pass
def priority_complete():
    p=read(HERE/'annotation.json');s=Path(p['rollout_path']).stat()
    return p['task_complete_verified'] and p['turn_status']=='completed' and s.st_size==p['rollout_size'] and s.st_mtime_ns==p['rollout_mtime_ns']
def gate(model,own_pid=None):
    from predictive_tracks.ppr_20261028.control import gpu_idle
    from predictive_tracks.ppr_existing_20261102.resource_resume_004.guarded import hardware
    h=hardware(own_pid);idle,g=gpu_idle(own_pid);priority=priority_complete();large=model=='videomaev2'
    safe=(h['available_ram_mib']>=(10240 if own_pid else 16384) and h['process_rss_mib']<=(9216 if large else 2560)
        and h['gpu_free_mib']>=(7168 if large and own_pid else 10240) and h['gpu_temperature_c']<=62 and h['cpu_temperature_c']<=70)
    d=dict(at=time.time(),priority_complete=priority,gpu_idle=idle,gpu=g,hardware=h,operational_limits_passed=safe)
    status(HERE/'monitor/gate.json',d);return priority and idle and safe,d
def require_resources(model):
    okay,d=gate(model,os.getpid())
    if not okay:raise ResourceBusy(json.dumps(d))
def video_rest():
    time.sleep(2)
    while True:
        okay,d=gate('videomaev2',os.getpid());h=d['hardware']
        if not d['priority_complete'] or not d['gpu_idle'] or h['available_ram_mib']<10240 or h['process_rss_mib']>9216 or h['gpu_free_mib']<7168 or h['gpu_temperature_c']>=68 or h['cpu_temperature_c']>=75:
            raise ResourceBusy(json.dumps(d))
        if okay and h['gpu_temperature_c']<=55 and h['cpu_temperature_c']<=65:return
        status(HERE/'monitor/thermal_idle.json',dict(at=time.time(),state='cooling_no_model_compute',hardware=h));time.sleep(2)
