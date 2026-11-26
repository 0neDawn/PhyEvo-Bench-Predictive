
from release_paths import runtime_python as _release_python, project_path as _release_path
import hashlib
import json
import os
from pathlib import Path
import time
from predictive_tracks.ppr_all_20261029.common import verify as verify_prior
from predictive_tracks.ppr_20261028.control import gpu_idle
from predictive_tracks.controlled_tracks.run import verify_frozen

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUN=HERE/'runs/full_001'
ETAS=(.25,.5,.75)
RECURRENT=('ddlp','gswm','ocvp_seq','savi_transformer','slotformer')
CACHED=('slotformer','vjepa2_vitl','vjepa2_vitg','videomaev2','dino_foresight')
PYTHONS={'ddlp':_release_python('ddlp'),
         'other':_release_python('other')}

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while True:
            b=f.read(2**20)
            if not b:break
            h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())

def save(path,value):
    path=Path(path)
    if not path.resolve().is_relative_to(HERE):raise ValueError('Writes restricted to new experiment')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')

def status(path,value):
    path=Path(path)
    if not path.resolve().is_relative_to(HERE):raise ValueError('Writes restricted to new experiment')
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)

def eta_name(eta):return f'eta_{int(round(100*eta)):03d}'

def directory(model,mode,eta):return RUN/model/mode/eta_name(eta)

def verify():
    verify_prior()
    frozen=read(HERE/'frozen.json')
    verify_frozen(frozen['old']);verify_frozen(frozen['own'])

def gate(own_pid=None):
    try:
        s=read(HERE/'monitor/annotation.json')
        marker=Path(s['rollout_path']).stat()
        task=(s['task_complete_verified'] and s['turn_status']=='completed'
              and marker.st_size==s['rollout_size'] and marker.st_mtime_ns==s['rollout_mtime_ns'])
    except (OSError,ValueError,KeyError):task=False
    idle,gpu=gpu_idle(own_pid)
    result={'checked_at':time.time(),'priority_task_unchanged_and_complete':task,'gpu_idle':idle,'gpu':gpu,'own_pid':own_pid}
    status(HERE/'monitor/gate.json',result)
    return task and idle,result

class ResourceBusy(RuntimeError):pass

def require_resources(own_pid=None):
    okay,detail=gate(own_pid)
    if not okay:raise ResourceBusy(json.dumps(detail))
