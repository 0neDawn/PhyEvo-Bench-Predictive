"""Search bookkeeping; the previously audited correction is reused verbatim."""

from release_paths import runtime_python as _release_python, project_path as _release_path
import hashlib
import json
import os
import time
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / "predictive_tracks/ppr_eta_radius_20261104_o"
SOURCE = OLD / "delivery_20261104_001/report.json"
PYTHON = _release_python('other')
CACHED = ("vjepa2_vitl", "videomaev2", "slotformer", "vjepa2_vitg", "dino_foresight")
NATIVE = ("ocvp_seq", "savi_transformer", "slotformer", "ddlp", "gswm", "vjepa2_vitl", "videomaev2")
OBJECTS = ("ddlp", "gswm")
LARGE = ("vjepa2_vitl", "videomaev2")
ORDER = ("videomaev2", "ocvp_seq", "savi_transformer", "dino_foresight", "vjepa2_vitg", "vjepa2_vitl", "ddlp", "slotformer", "gswm")


class ResourceBusy(RuntimeError):
    pass


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
    if not path.resolve().is_relative_to(HERE):
        raise ValueError("Writes must remain in the new P directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".atomic.{os.getpid()}.{time.time_ns()}")
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


@lru_cache(maxsize=1)
def source():
    return read(SOURCE)


def quality(model, task):
    if model not in OBJECTS:
        return None
    p = source()["selected"][task][model]["parameters"]
    return {k: p[k] for k in ("q", "presence", "mass", "min_history")}


def quality_key(parameters):
    if parameters.get("q") is None:
        return None
    return json.dumps({k: parameters.get(k) for k in ("q", "presence", "mass", "min_history")}, sort_keys=True)


def semantic_key(model, mode, eta, alpha, q=None):
    return (model, mode, float(eta), float(alpha), json.dumps(q, sort_keys=True))


def existing_keys():
    return {semantic_key(m, x["mode"], x["parameters"]["eta"], x["parameters"].get("alpha", 1.),
                         {k: x["parameters"].get(k) for k in ("q", "presence", "mass", "min_history")} if m in OBJECTS else None)
            for m, values in source()["all_configurations"].items() for x in values}


def job(model, mode, eta, alpha, q=None, target=None, **extra):
    eta, alpha = round(float(eta), 8), round(float(alpha), 8)
    assert 0 < eta <= 1 and 1 <= alpha <= 4
    value = dict(model=model, mode=mode, eta=eta, alpha=alpha, quality=q, target_task=target)
    token = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:12]
    value["config_id"] = f"{model}_{mode}_e{eta:.8g}_a{alpha:.8g}_{token}"
    value.update(extra)
    return value


def cpu_guard():
    available = next(int(line.split()[1]) / 1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    if available < 12288:
        raise ResourceBusy("CPU proposal paused: available RAM below 12 GiB")
    annotation = read(HERE / "annotation.json")
    stat = Path(annotation["rollout_path"]).stat()
    if stat.st_size != annotation["rollout_size"] or stat.st_mtime_ns != annotation["rollout_mtime_ns"]:
        raise ResourceBusy("Priority annotation task changed; recheck before continuing")
    return available


def verify(full=False):
    declaration = HERE / "frozen.json"
    if not declaration.exists():
        declaration = HERE / "ANALYSIS_FROZEN.json"
    for path, wanted in read(declaration)["files"].items():
        if digest(path) != wanted:
            raise RuntimeError("Frozen source changed: " + path)
    if full:
        # Original modules stay unchanged. Only their process-local output root
        # is switched back while their original full preservation audit runs.
        from predictive_tracks.ppr_eta_radius_20261104_o import common as old
        current = old.HERE
        try:
            old.HERE = OLD
            _ORIGINAL_VERIFY(full=True)
        finally:
            old.HERE = current
        for path, wanted in read(OLD / "delivery_20261104_001/input_hashes.json").items():
            if digest(path) != wanted:
                raise RuntimeError("Previous O result changed: " + path)


_ORIGINAL_VERIFY = None


def install_original_runtime():
    """Route unchanged O worker/operator code to new files in this process."""
    global _ORIGINAL_VERIFY
    from predictive_tracks.ppr_eta_radius_20261104_o import common as old
    if _ORIGINAL_VERIFY is None:
        _ORIGINAL_VERIFY = old.verify
    old.HERE, old.SOURCE = HERE, SOURCE
    old.verify = verify
    return old
