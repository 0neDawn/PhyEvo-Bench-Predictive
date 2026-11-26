
from release_paths import runtime_python as _release_python, project_path as _release_path
import hashlib
import json
import os
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PARENT=ROOT/'predictive_tracks/ppr_position_refine_20261106_s'
OLD=ROOT/'predictive_tracks/ppr_eta_radius_20261104_o'
PYTHON=_release_python('other')
SLOTS=('ocvp_seq','savi_transformer')
FEATURES=()


def read(path):return json.loads(Path(path).read_text())
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()
def write(path,value,replace=False):
    path=Path(path);assert path.resolve().is_relative_to(HERE)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.atomic.{os.getpid()}.{time.time_ns()}')
    try:
        with temp.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
        if replace:os.replace(temp,path)
        else:os.link(temp,path);temp.unlink()
    finally:
        if temp.exists():temp.unlink()
def save(path,value):write(path,value)
def status(path,value):write(path,value,True)
def source_path():return Path(read(PARENT/'FINAL_DELIVERY.json')['directory'])/'report.json'
def directory(item):return HERE/'runs'/item['model']/item['config_id']
def job(model,mode,eta,alpha,position=None,**extra):
    assert position is None
    eta,alpha=round(float(eta),8),round(float(alpha),8)
    assert 0<eta<=1 and alpha>=1
    item=dict(model=model,mode=mode,eta=eta,alpha=alpha,quality=None,target_task=None)
    token=hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()[:12]
    item['config_id']=f'T_{model}_{mode}_e{eta:.8g}_a{alpha:.8g}_{token}'
    item.update(extra);return item
def freeze_grid(name,value):
    path=HERE/'grids'/f'{name}.json'
    if path.exists():
        old=read(path)
        assert {k:v for k,v in old.items() if k!='at'}=={k:v for k,v in value.items() if k!='at'}
    else:save(path,value)
    frozen=path.with_suffix('.frozen.json')
    if frozen.exists():assert read(frozen)['sha256']==digest(path)
    else:save(frozen,dict(at=time.time(),file=str(path),sha256=digest(path)))
    return path
def verify(full=False):
    for path,wanted in read(HERE/'frozen.json')['files'].items():assert digest(path)==wanted,path
    for path in (HERE/'grids').glob('*.frozen.json'):
        d=read(path);assert digest(d['file'])==d['sha256']
    if full:
        assert read(PARENT/'FINAL_STATE_001.json')['all_lineage_audits_complete']
        for path,wanted in read(HERE/'PARENT_MANIFESTS.json').items():
            assert digest(path)==wanted,path
            for p,sha in read(path)['files'].items():assert digest(p)==sha,p
def install():
    from predictive_tracks.ppr_eta_radius_20261104_o import common as old
    old.HERE,old.SOURCE,old.verify,old.directory=HERE,source_path(),verify,directory
    old.NATIVE=SLOTS
    return old
def require_resources(model):install().require_resources(model)

