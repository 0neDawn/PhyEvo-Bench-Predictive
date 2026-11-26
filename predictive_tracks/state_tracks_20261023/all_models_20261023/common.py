import hashlib
import json
from pathlib import Path
import sys

from ..integrity import ROOT, HERE as STATE, digest, verify
from ..fullscale_20261023.common import protected_hashes
from predictive_tracks.controlled_tracks.registry import CODE, OUTPUT, NATIVE_TOLERANCE, archive
from predictive_tracks.controlled_tracks.run import records_and_split

HERE = Path(__file__).resolve().parent
MODELS = {
    'ddlp': 'sp_core.adapters.track_state_20261023_ddlp',
    'slotformer': 'sp_core.adapters.track_state_20261023_slotformer',
    'ocvp_seq': 'sp_ocvp.track_state_20261023_ocvp',
    'savi_transformer': 'sp_ocvp.track_state_20261023_transformer',
    'gswm': 'sp_gswm.adapters.track_state_20261023',
    'vjepa2_vitl': 'sp_vjepa2_vitl.track_state_20261023',
    'videomaev2': 'sp_videomae.track_state_20261023',
    'vjepa2_vitg': 'sp_protocol.adapters.track_state_20261023_vjepa2',
    'dino_foresight': 'sp_dino_foresight.track_state_20261023',
}
RGB = {'ddlp','ocvp_seq','savi_transformer','gswm'}
sys.path.insert(0, str(CODE))


def write(path, value):
    path=Path(path).resolve()
    if not path.is_relative_to(HERE): raise ValueError('Only new experiment output allowed')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f: json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def read(path):
    return json.loads(Path(path).read_text())


def rows():
    return [dict(r,benchmark=task) for task in ('sp','ie') for r in records_and_split(task)]


def key(row):
    return f"{row['benchmark']}_{row['archive_index']:03d}"


def freeze():
    old=protected_hashes()
    old.update({str(p):digest(p) for p in (STATE/'fullscale_20261023').iterdir()
                if p.is_file() and p.suffix in ('.py','.json')})
    verify(old)
    own={str(p):digest(p) for p in HERE.iterdir() if p.is_file() and p.suffix in ('.py','.json')}
    own.update({str(CODE/(module.replace('.','/')+'.py')):digest(CODE/(module.replace('.','/')+'.py'))
                for module in MODELS.values()})
    write(HERE/'frozen.json',{'old_sources':old,'new_sources':own,
                            'models':list(MODELS),'pairs':len(rows()),'plan':read(HERE/'plan.json')})


def verify_all():
    frozen=read(HERE/'frozen.json')
    verify(frozen['old_sources']);verify(frozen['new_sources'])
    return frozen


def reference(model,row):
    if model=='dino_foresight': return None
    split='dev' if row['split']=='development' else 'confirmation'
    return OUTPUT/f'{model}_parallel_{split}_001'/'pairs'/(key(row)+'.json')


def existing_cache(model,row):
    if model=='ddlp':
        root=(STATE/'runs/ddlp_native_capture_dev_002' if row['split']=='development' else
              STATE/'fullscale_20261023/runs/native_remaining_001')
        meta=root/'pairs'/(key(row)+'.json')
        payload=read(meta)
        return dict(meta=str(meta),state=str(root/'states'/(key(row)+'.pt')),
                    sha256=payload['state_cache_sha256'],layout='ddlp_particle')
    if model!='dino_foresight' and row['split']=='development':
        root=OUTPUT/f'{model}_boundary_dev_001'
        meta=root/'pairs'/(key(row)+'.json')
        payload=read(meta)
        return dict(meta=str(meta),state=str(root/'native_states'/(key(row)+'.pt')),
                    sha256=payload['native_state_cache_sha256'],layout='native')
    return None


def donors():
    result={}
    for task in ('sp','ie'):
        for split in ('development','confirmation'):
            chosen=[r for r in rows() if r['benchmark']==task and r['split']==split]
            chosen.sort(key=lambda r:hashlib.sha256(('20261023'+r['pair_id']).encode()).hexdigest())
            for i,r in enumerate(chosen): result[key(r)]=key(chosen[(i+1)%len(chosen)])
    return result


if __name__=='__main__': freeze()
