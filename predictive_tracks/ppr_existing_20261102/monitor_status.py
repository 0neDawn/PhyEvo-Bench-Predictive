"""Read-only supervision; never imports a model or changes research outputs."""

from release_paths import runtime_python as _release_python, project_path as _release_path
from pathlib import Path
import json
import time

base=Path(__file__).resolve().parent/'monitor'
state=json.loads((base/'gpu.json').read_text())
result={'state':state['state'],'elapsed_s':round(time.time()-state['started']),
    'completed':state.get('completed',[]),'failures':state.get('failures',[]),
    'active_model':(state.get('active') or {}).get('model'),'resource_check':state.get('last_resource_check')}
for model in ('ddlp','gswm'):
    p=base/f'{model}_progress.json'
    if p.exists():result[model]=json.loads(p.read_text())
for key in ('FINAL','latest_report'):
    p=base/(key+'.json')
    if p.exists():result[key]=json.loads(p.read_text())
p=Path(_release_path('historical_metadata/discussion.txt'))
if p.exists():
    result['discussion_latest_header']=next((s for s in reversed(p.read_text().splitlines()) if s.startswith('【第')),None)
    result['discussion_mtime_ns']=p.stat().st_mtime_ns
print(json.dumps(result,ensure_ascii=False))
