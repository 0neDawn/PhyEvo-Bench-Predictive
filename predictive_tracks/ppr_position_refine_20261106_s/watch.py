"""Read-only, no model imports, bounded progress and resource snapshots."""

from release_paths import runtime_python as _release_python, project_path as _release_path
import json
import time
from pathlib import Path
from . import common as c


def main():
    path=c.HERE/'monitor/queue.json'
    q=c.read(path) if path.exists() else dict(state='not_started')
    active=q.get('active') or {}
    paths=[]
    model=active.get('model')
    if model:
        phase=active['phase']
        if phase=='cached':
            paths=[c.HERE/'cache'/model/'progress.json']
        elif phase=='cached_fine':
            paths=[c.HERE/'cache_stage2'/model/'progress.json']
        elif phase=='capture':
            paths=[c.HERE/'capture'/model/'progress.json']
        elif phase in ('coarse','fine'):
            paths=[c.HERE/'search'/phase/model/'progress.json']
        elif phase=='validation':
            paths=sorted((c.HERE/'runs'/model).glob('*/progress.json'))
    progress=[dict(file=str(p),**c.read(p)) for p in paths if p.exists()]
    hp=c.HERE/'monitor/hardware_latest.json'
    h=c.read(hp) if hp.exists() else None
    result=dict(at=time.time(),state=q.get('state'),active=active,completed_steps=q.get('completed',[]),error=q.get('error'),
        progress=progress,hardware=h,hardware_age_seconds=None if h is None else round(time.time()-h['at'],1),
        declared_dynamic_grids={p.stem:c.read(p).get('settings') for p in (c.HERE/'grids').glob('*.json') if not p.name.endswith('.frozen.json')},
        markers={name:(c.HERE/name).exists() for name in ('MODEL_RUNS_COMPLETE.json','PERSISTED.json','FINAL_STATE_001.json')})
    sources=sorted(c.HERE.glob('SHARED_DISCUSSION_*.json'))
    if sources:
        latest=c.read(sources[-1])
        result['new_discussion']=c.digest(_release_path('historical_metadata/discussion.txt'))!=latest['after_sha256']
    else:
        result['new_discussion']=None
    c.save(c.HERE/'author_monitor'/f'{time.time_ns()}.json',result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
