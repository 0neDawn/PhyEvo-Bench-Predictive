"""All writes stay in S; parent scientific files are verified, never edited."""

from release_paths import runtime_python as _release_python, project_path as _release_path
import hashlib
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = ROOT / 'predictive_tracks/ppr_internal_activation_20261106_r'
OLD = ROOT / 'predictive_tracks/ppr_eta_radius_20261104_o'
PYTHON = _release_python('other')
FEATURES = ('videomaev2', 'vjepa2_vitl')
SLOTS = ('ocvp_seq', 'savi_transformer')
ETAS = (.0025, .01, .05, .1, .25, .5, .75, 1.)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value, replace=False):
    path = Path(path)
    assert path.resolve().is_relative_to(HERE)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.atomic.{os.getpid()}.{time.time_ns()}')
    try:
        with temp.open('x') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        if replace:
            os.replace(temp, path)
        else:
            os.link(temp, path)
            temp.unlink()
    finally:
        if temp.exists():
            temp.unlink()


def save(path, value):
    write(path, value)


def status(path, value):
    write(path, value, True)


def source_path():
    return Path(read(PARENT/'FINAL_DELIVERY.json')['directory'])/'report.json'


def directory(job):
    return HERE/'runs'/job['model']/job['config_id']


def job(model, mode, eta, alpha, position=None, **extra):
    eta, alpha = round(float(eta), 8), round(float(alpha), 8)
    assert 0 < eta <= 1 and alpha >= 1
    value = dict(model=model, mode=mode, eta=eta, alpha=alpha, quality=None, target_task=None)
    if position is not None:
        value['position'] = position
    token = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:12]
    value['config_id'] = f'S_{model}_{mode}_{position or "native"}_e{eta:.8g}_a{alpha:.8g}_{token}'
    value.update(extra)
    return value


def verify(full=False):
    for path, wanted in read(HERE/'frozen.json')['files'].items():
        assert digest(path) == wanted, path
    for path in sorted((HERE/'grids').glob('*.frozen.json')):
        item = read(path)
        assert digest(item['file']) == item['sha256'], item['file']
    if full:
        assert read(PARENT/'FINAL_STATE_001.json')['all_lineage_audits_complete']
        for path, wanted in read(HERE/'PARENT_MANIFESTS.json').items():
            assert digest(path) == wanted, path
            for filename, sha in read(path)['files'].items():
                assert digest(filename) == sha, filename


def freeze_grid(name, value):
    path = HERE/'grids'/f'{name}.json'
    save(path, value)
    save(path.with_suffix('.frozen.json'), dict(at=time.time(), file=str(path), sha256=digest(path)))
    return path


def install():
    from predictive_tracks.ppr_eta_radius_20261104_o import common as old
    old.HERE, old.SOURCE, old.verify, old.directory = HERE, source_path(), verify, directory
    old.NATIVE = SLOTS+FEATURES
    old.CACHED = ('dino_foresight',)
    return old


def require_resources(model):
    install().require_resources(model)


def quantile(values, probability):
    values = sorted(values)
    x = (len(values)-1)*probability
    i, j = int(x), min(int(x)+1, len(values)-1)
    return values[i] if i == j else values[i]*(j-x)+values[j]*(x-i)


def grid_alphas(ratios):
    # Only history/native predictions enter these empirical activation scales.
    return sorted({round(max(1., quantile(ratios, q)), 8) for q in (0., .1, .25, .5, .75, .9, 1.)}|{1.})


def save_tensor(path, value):
    import torch
    path = Path(path)
    assert path.resolve().is_relative_to(HERE)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.atomic.{os.getpid()}.{time.time_ns()}')
    with temp.open('xb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temp, path)
    temp.unlink()
