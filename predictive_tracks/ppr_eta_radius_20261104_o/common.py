"""Isolated O namespace. Only existing eta/alpha are varied."""

from release_paths import runtime_python as _release_python, project_path as _release_path
import hashlib,json,os,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=ROOT/'predictive_tracks/ppr_eta_radius_20261104_n/delivery_20261104_001/report.json'
PYTHON=_release_python('other')
CACHED=('vjepa2_vitl','vjepa2_vitg','dino_foresight','videomaev2','slotformer')
SLOTS=('ocvp_seq','savi_transformer','slotformer')
OBJECTS=('ddlp','gswm')
LARGE=('vjepa2_vitl','videomaev2')
NATIVE=('ocvp_seq','savi_transformer','slotformer','ddlp','gswm','vjepa2_vitl','videomaev2')
ETAS=(.0005,.001,.0025,.005,.01,.02,.03,.04,.05,.075,.1,.125,.2,.25,.35,.5,.75,1.)
ALPHAS=(1.,1.1,1.25,1.5,1.75,2.,2.25,2.5,3.,4.)
CURRENT={}
def read(p):return json.loads(Path(p).read_text())
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def write(p,v,replace=False):
    p=Path(p)
    if not p.resolve().is_relative_to(HERE):raise ValueError('Writes only in new O namespace')
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.atomic.{os.getpid()}.{time.time_ns()}')
    try:
        with tmp.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
        if replace:os.replace(tmp,p)
        else:os.link(tmp,p);tmp.unlink()
    finally:
        if tmp.exists():tmp.unlink()
def save(p,v):write(p,v)
def status(p,v):write(p,v,True)
def job(model,mode,eta,alpha,quality=None,target=None):
    x=dict(model=model,mode=mode,eta=float(eta),alpha=float(alpha),quality=quality,target_task=target)
    h=hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()[:12]
    x['config_id']=f'{model}_{mode}_e{eta:.8g}_a{alpha:.8g}_{h}';return x
def directory(j):return HERE/'runs'/j['model']/j['config_id']
def python(model):return _release_python('ddlp') if model=='ddlp' else PYTHON
def verify(full=False):
    for p,w in read(HERE/'frozen.json')['files'].items():
        if digest(p)!=w:raise RuntimeError('Frozen source changed: '+p)
    if (HERE/'STAGE2_FROZEN.json').exists():
        for p,w in read(HERE/'STAGE2_FROZEN.json')['files'].items():
            if digest(p)!=w:raise RuntimeError('Stage2 declaration changed')
    if full:
        from predictive_tracks.ppr_eta_radius_20261104_n.common import verify as prior
        prior(full=True)
class ResourceBusy(RuntimeError):pass
def priority_complete():
    p=read(HERE/'annotation.json');s=Path(p['rollout_path']).stat()
    return p['task_complete_verified'] and p['turn_status']=='completed' and s.st_size==p['rollout_size'] and s.st_mtime_ns==p['rollout_mtime_ns']
def gate(model,own_pid=None):
    from predictive_tracks.ppr_20261028.control import gpu_idle
    from predictive_tracks.ppr_existing_20261102.resource_resume_004.guarded import hardware
    h=hardware(own_pid);idle,g=gpu_idle(own_pid);large=model in LARGE
    safe=(h['available_ram_mib']>=(10240 if own_pid else 16384) and h['process_rss_mib']<=(9216 if large else 2560)
        and h['gpu_free_mib']>=(7168 if large and own_pid else 10240) and h['gpu_temperature_c']<=62 and h['cpu_temperature_c']<=70)
    d=dict(at=time.time(),priority_complete=priority_complete(),gpu_idle=idle,gpu=g,hardware=h,safe=safe)
    status(HERE/'monitor/gate.json',d);return d['priority_complete'] and idle and safe,d
def require_resources(model):
    okay,d=gate(model,os.getpid())
    if not okay:raise ResourceBusy(json.dumps(d))
def before_native(model):
    if model not in LARGE:require_resources(model);return
    time.sleep(2)
    while True:
        okay,d=gate(model,os.getpid());h=d['hardware']
        if not d['priority_complete'] or not d['gpu_idle'] or h['available_ram_mib']<10240 or h['process_rss_mib']>9216 or h['gpu_free_mib']<7168 or h['gpu_temperature_c']>=68 or h['cpu_temperature_c']>=75:raise ResourceBusy(json.dumps(d))
        if okay and h['gpu_temperature_c']<=55 and h['cpu_temperature_c']<=65:return
        status(HERE/'monitor/thermal_idle.json',dict(at=time.time(),state='cooling',hardware=h));time.sleep(2)
