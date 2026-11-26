import hashlib
import json
import os
from pathlib import Path
import time

from predictive_tracks.controlled_tracks.run import digest, frozen_files, verify_frozen, records_and_split
from predictive_tracks.controlled_tracks.registry import ROOT, CODE, OUTPUT
from predictive_tracks.ppr_20261028.control import gpu_idle, write_status

HERE = Path(__file__).resolve().parent
RUN = HERE / 'runs/full_001'
SLOTS = ('ocvp_seq', 'savi_transformer', 'slotformer')
FEATURES = ('vjepa2_vitl', 'vjepa2_vitg', 'videomaev2')
REMAINING = SLOTS + FEATURES + ('dino_foresight',)
OLD_CACHE = ROOT / 'predictive_tracks/state_tracks_20261023/all_models_20261023/runs/all_models_001'
THREAD = '019f7e4b-0e84-7991-8dd6-4a9a2fba7e27'

def save(path, obj):
    path = Path(path)
    if not path.resolve().is_relative_to(HERE):
        raise ValueError('Only new experiment directory is writable')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def read(path):
    return json.loads(Path(path).read_text())

def rows():
    result = [dict(r, benchmark=t, key=f'{t}_{r["archive_index"]:03d}')
              for t in ('sp', 'ie') for r in records_and_split(t)]
    if len(result) != 600 or len({r['key'] for r in result}) != 600:
        raise RuntimeError('Original full pair count changed')
    return result

def old_meta(model, row):
    p = OLD_CACHE / model / 'native/pairs' / (row['key'] + '.json')
    if not p.exists():
        return None
    # The old interrupted DINO run left a zero-byte metadata file. Preserve it;
    # absence of a completed record is not a reusable native forecast cache.
    if model == 'dino_foresight' and p.stat().st_size == 0:
        return None
    return p

def reference(model, row):
    split = 'dev' if row['split'] == 'development' else 'confirmation'
    return OUTPUT / f'{model}_parallel_{split}_001/pairs' / (row['key'] + '.json')

def freeze():
    old = frozen_files()
    prior = read(ROOT / 'predictive_tracks/state_tracks_20261023/all_models_20261023/frozen.json')
    old.update(prior['old_sources']); old.update(prior['new_sources'])
    for sub in ('ppr_20261028', 'ppr_reports_20261028'):
        for p in (ROOT / 'predictive_tracks' / sub).glob('*.py'):
            old[str(p)] = digest(p)
    verify_frozen(old)
    own = {str(p): digest(p) for p in HERE.rglob('*.py') if 'runs' not in p.parts}
    own[str(HERE / 'PLAN.txt')] = digest(HERE / 'PLAN.txt')
    save(HERE / 'frozen.json', {'old': old, 'new': own, 'created': time.time()})

def verify():
    frozen = read(HERE / 'frozen.json')
    verify_frozen(frozen['old']); verify_frozen(frozen['new'])
    return frozen

def gate(own_pid=None, stable=1):
    count = 0
    while count < stable:
        try:
            a = read(HERE / 'monitor/annotation.json')
            complete = (a['thread_id'] == THREAD and a['task_complete_verified']
                        and a['turn_status'] == 'completed' and time.time()-a['checked_at'] < 300)
        except (OSError, KeyError, ValueError):
            complete = False
        idle, gpu = gpu_idle(own_pid)
        count = count+1 if complete and idle else 0
        write_status(HERE / 'monitor/gate.json', {'time': time.time(), 'task_complete': complete,
                      'gpu_idle': idle, 'gpu': gpu, 'stable': count, 'own_pid': own_pid})
        if count < stable:
            time.sleep(15)

def progress(directory, completed, total):
    write_status(directory / 'progress.json', {'completed': completed, 'total': total, 'time': time.time()})
