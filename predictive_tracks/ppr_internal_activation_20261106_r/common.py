"""Separate files and unchanged original operators; no old file writes."""

from release_paths import runtime_python as _release_python, project_path as _release_path
import hashlib
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = ROOT / "predictive_tracks/ppr_radius_activation_20261105_q"
OLD = ROOT / "predictive_tracks/ppr_eta_radius_20261104_o"
PYTHON = _release_python('other')
FOCUS = ("vjepa2_vitl", "videomaev2")


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value, replace=False):
    path = Path(path)
    assert path.resolve().is_relative_to(HERE)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f".atomic.{os.getpid()}.{time.time_ns()}")
    try:
        with temp.open("x") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
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
    return Path(read(PARENT / "FINAL_DELIVERY.json")["directory"]) / "report.json"


def job(model, mode, eta, alpha, **extra):
    eta, alpha = round(float(eta), 8), round(float(alpha), 8)
    assert 0 < eta <= 1 and alpha >= 1
    item = dict(model=model, mode=mode, eta=eta, alpha=alpha, quality=None, target_task=None)
    token = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()[:12]
    item["config_id"] = f"R_{model}_{mode}_e{eta:.8g}_a{alpha:.8g}_{token}"
    item.update(extra)
    return item


def directory(item):
    return HERE / "native_runs" / item["model"] / item["config_id"]


def verify(full=False):
    for path, wanted in read(HERE / "frozen.json")["files"].items():
        assert digest(path) == wanted, path
    if (HERE / "STAGE2.json").exists():
        assert digest(HERE / "STAGE2.json") == read(HERE / "STAGE2_FROZEN.json")["sha256"]
    if (HERE / "VALIDATION_GRID.json").exists():
        assert digest(HERE / "VALIDATION_GRID.json") == read(HERE / "VALIDATION_FROZEN.json")["sha256"]
    if full:
        assert read(PARENT / "FINAL_STATE_001.json")["all_lineage_audits_complete"]
        for name in ("DELIVERY.json", "SUPPLEMENT_MANIFEST_001.json", "USER_SUMMARY_MANIFEST_001.json"):
            manifest = source_path().parent / name
            assert digest(manifest) == read(HERE / "PARENT_MANIFESTS.json")[str(manifest)]
            for path, wanted in read(manifest)["files"].items():
                assert digest(path) == wanted, path


def install():
    from predictive_tracks.ppr_eta_radius_20261104_o import common as old
    old.HERE, old.SOURCE, old.verify = HERE, source_path(), verify
    old.directory = directory
    old.NATIVE = ("vjepa2_vitl", "videomaev2", "ocvp_seq", "savi_transformer")
    return old


def require_resources(model):
    old = install()
    okay, detail = old.gate(model, os.getpid())
    if not okay:
        raise old.ResourceBusy(json.dumps(detail))


def all_tail_jobs(model):
    result = [j for j in read(HERE / "GRID.json")["tail"] if j["model"] == model]
    if (HERE / "STAGE2.json").exists():
        result += [j for j in read(HERE / "STAGE2.json")["tail"] if j["model"] == model]
    return result
