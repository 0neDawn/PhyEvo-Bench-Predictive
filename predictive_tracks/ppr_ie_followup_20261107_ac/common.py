
from release_paths import runtime_python as _release_python, project_path as _release_path
import hashlib
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = ROOT/'predictive_tracks/ppr_feedback_refine_20261106_t'
BASE = ROOT/'predictive_tracks/ppr_position_refine_20261106_s'
PYTHON = _release_python('other')
MODELS = ('ocvp_seq', 'savi_transformer', 'dino_foresight')
SLOTS = MODELS[:2]
FEATURES = ('dino_foresight',)


def read(p):
    return json.loads(Path(p).read_text())


def digest(p):
    value = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20), b''):
            value.update(block)
    return value.hexdigest()


def write(p, value, replace=False):
    p = Path(p)
    assert p.resolve().is_relative_to(HERE)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name+f'.atomic.{os.getpid()}.{time.time_ns()}')
    try:
        with temp.open('x') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        if replace:
            os.replace(temp, p)
        else:
            os.link(temp, p)
            temp.unlink()
    finally:
        if temp.exists():
            temp.unlink()


def save(p, value):
    write(p, value)


def status(p, value):
    write(p, value, True)


def install():
    from predictive_tracks.ppr_eta_radius_20261104_o import common as old
    old.HERE = HERE
    old.LARGE = tuple(dict.fromkeys(old.LARGE+FEATURES))
    return old


def source_path():
    return Path(read(PARENT/'FINAL_DELIVERY.json')['directory'])/'report.json'


def job(model, mode, eta, alpha, **extra):
    eta, alpha = round(float(eta), 8), round(float(alpha), 8)
    assert model in MODELS and 0 < eta <= 1 and alpha >= 1
    assert mode in ('feedback','state_only','output_only')
    assert model != 'dino_foresight' or mode == 'feedback'
    value = dict(model=model,mode=mode,eta=eta,alpha=alpha,quality=None,target_task=None)
    token = hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()[:12]
    value['config_id'] = f'AC_{model}_{mode}_e{eta:.8g}_a{alpha:.8g}_{token}'
    value.update(extra)
    return value


def directory(item):
    return HERE/'runs'/item['model']/item['config_id']


def verify_sources(name='SOURCE_READY_001.json'):
    for path, wanted in read(HERE/name)['files'].items():
        assert digest(path) == wanted, path


def verify(full=False):
    verify_sources()
    for path in (HERE/'grids').glob('*.frozen.json'):
        value = read(path)
        assert digest(value['file']) == value['sha256']
    if full:
        for path, wanted in read(HERE/'PARENT_MANIFESTS.json').items():
            assert digest(path) == wanted, path


def freeze_grid(name, value):
    path = HERE/'grids'/f'{name}.json'
    save(path, value)
    save(path.with_suffix('.frozen.json'),dict(file=str(path),sha256=digest(path),at=time.time()))
    return path


def require_resources(model):
    old = install()
    old.require_resources(model)


def dino_rows():
    from predictive_tracks.ppr_dino_feedback_20261106_aa.data import rows
    return rows()
