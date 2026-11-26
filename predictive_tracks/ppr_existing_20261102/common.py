"""Only existing HCAC parameters. There is no radius multiplier argument."""

from release_paths import runtime_python as _release_python, project_path as _release_path
import hashlib
import itertools
import json
import os
from pathlib import Path
import time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUN=HERE/'runs/full_001'
OBJECTS=('ddlp','gswm')
RECURRENT=OBJECTS+('ocvp_seq','savi_transformer','slotformer')
FEATURES=('vjepa2_vitl','vjepa2_vitg','videomaev2','dino_foresight')
MODELS=RECURRENT+FEATURES
ETAS=(.25,.5,.75,1.)
QS=(.25,.5,.75)
PRESENCES=(.5,.75)
MASSES=(4.,8.)
COUNTS=(2,3)
PYTHONS={'ddlp':_release_python('ddlp'),'other':_release_python('other')}
PRIORITY=Path(_release_path('historical_metadata/priority_task.jsonl'))

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())

def save(path,value):
    path=Path(path)
    if not path.resolve().is_relative_to(HERE):raise ValueError('Only this new namespace is writable')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def status(path,value):
    path=Path(path)
    if not path.resolve().is_relative_to(HERE):raise ValueError('Only this new namespace is writable')
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(temp,path)

def configs(model):
    quality=itertools.product(QS,PRESENCES,MASSES,COUNTS) if model in OBJECTS else [(None,None,None,None)]
    return [{'eta':eta,'q':q,'presence':p,'mass':v,'min_history':n} for q,p,v,n in quality for eta in ETAS]

def config_id(c):
    s=f'e{round(c["eta"]*100):03d}'
    if c['q'] is not None:s+=f'_q{round(c["q"]*100):03d}_p{round(c["presence"]*100):03d}_v{c["mass"]:g}_n{c["min_history"]}'
    return s

def quality_id(c):return config_id(dict(c,eta=1.))
def qualities(model):return [c for c in configs(model) if c['eta']==1.]
def primary(model):return 'feedback' if model in RECURRENT else 'output_only'
def directory(model,c):return RUN/model/config_id(c)

def old_folder(model,c):
    if 'alpha' in c:raise ValueError('Radius multiplier is not an allowed parameter')
    root=ROOT/'predictive_tracks';mode=primary(model)
    default=c['q'] is None or (c['q']==.5 and c['presence']==.5 and c['mass']==4 and c['min_history']==2)
    if default:
        if c['eta']==1.:
            sub='ppr_20261028/runs/full_001' if model in OBJECTS else 'ppr_all_20261029/runs/'+('recovery_001' if model=='dino_foresight' else 'full_001')
            p=root/sub/model/mode
        else:p=root/'ppr_eta_20261030/runs/full_001'/model/mode/f'eta_{round(c["eta"]*100):03d}'
    elif model in OBJECTS and c['eta']==1. and c['presence']==.5 and c['mass']==4 and c['min_history']==2:
        p=root/'ppr_visibility_20261031/runs/full_001'/model/mode/f'q_{round(c["q"]*100):03d}'
    else:return None
    return p if (p/'summary.json').exists() else None

def result_folder(model,c):return old_folder(model,c) or directory(model,c)

def verify(full=False):
    frozen=read(HERE/'frozen.json');targets=dict(frozen['own'])
    targets.update(frozen['old'] if full else frozen['old_code_and_metadata'])
    for path,wanted in targets.items():
        if digest(path)!=wanted:raise RuntimeError(f'Frozen source/data changed: {path}')

def priority_complete():
    verified=read(HERE/'monitor/priority_verified.json')
    before=PRIORITY.stat();marker=[before.st_size,before.st_mtime_ns]
    cache=HERE/'monitor/priority_current.json'
    if cache.exists():
        prior=read(cache)
        if prior['marker']==marker:return prior['complete']
    latest=None
    with PRIORITY.open() as f:
        for line in f:
            try:v=json.loads(line)
            except json.JSONDecodeError:continue
            p=v.get('payload',{})
            if v.get('type')=='event_msg' and p.get('type') in ('task_started','task_complete','user_message'):
                latest={'timestamp':v.get('timestamp'),'type':p.get('type'),'turn_id':p.get('turn_id')}
    after=PRIORITY.stat()
    okay=bool(verified['task_complete_verified'] and latest and latest['type']=='task_complete' and latest['turn_id']==verified['turn_id']
              and marker==[after.st_size,after.st_mtime_ns])
    evidence={'complete':okay,'marker':marker,'last_lifecycle':latest,'at':time.time(),
        'rule':'Only the semantically verified completed turn is allowed. A new turn or new user message pauses the queue.'}
    save(HERE/'monitor/priority_evidence'/f'{time.time_ns()}.json',evidence);status(cache,evidence)
    return okay

def gate(own_pid=None):
    from predictive_tracks.ppr_20261028.control import gpu_idle
    try:complete=priority_complete()
    except (OSError,ValueError,KeyError):complete=False
    idle,gpu=gpu_idle(own_pid)
    value={'at':time.time(),'priority_complete':complete,'gpu_idle':idle,'gpu':gpu,'own_pid':own_pid}
    status(HERE/'monitor/gate.json',value);return complete and idle,value

class ResourceBusy(RuntimeError):pass
def require_resources(own_pid=None):
    okay,detail=gate(own_pid)
    if not okay:raise ResourceBusy(json.dumps(detail))
