"""Low-cost observation only; no GPU/model initialization or job changes."""

from release_paths import runtime_python as _release_python, project_path as _release_path
import hashlib
import json
import time
from pathlib import Path
from . import common as c


def main():
    now = time.time()
    q = c.read(c.HERE / "monitor/queue.json") if (c.HERE / "monitor/queue.json").exists() else {}
    progress = []
    for phase in ("tail", "tail_stage2"):
        for model in c.FOCUS:
            p = c.HERE / phase / model / "progress.json"
            if p.exists():
                progress.append(dict(phase=phase, **c.read(p)))
    active = q.get("active") or {}
    if active.get("phase") in ("native", "validation"):
        grid = c.read(c.HERE / ("VALIDATION_GRID.json" if active["phase"] == "validation" else "GRID.json"))
        for item in grid["native_new"]:
            if item["model"] != active["model"]:
                continue
            p = c.directory(item) / "progress.json"
            if p.exists():
                progress.append(dict(phase=active["phase"], model=item["model"], config_id=item["config_id"], **c.read(p)))
    hw = c.HERE / "monitor/hardware_latest.json"
    hardware = c.read(hw) if hw.exists() else {}
    receipts = list(c.HERE.glob("SHARED_DISCUSSION_*.json"))
    receipt = max(receipts, key=lambda p:int(p.stem.rsplit("_",1)[1])) if receipts else None
    shared_new = False
    if receipt:
        shared_new = hashlib.sha256(Path(_release_path('historical_metadata/discussion.txt')).read_bytes()).hexdigest() != c.read(receipt)["after_sha256"]
    result = dict(at=now, state=q.get("state"), active=active, completed_steps=q.get("completed",[]),
        error=q.get("error"), progress=progress, hardware=hardware,
        hardware_age_seconds=round(now-hardware["at"],1) if hardware else None,
        new_discussion=shared_new, markers={n:(c.HERE/n).exists() for n in ("MODEL_RUNS_COMPLETE.json","PERSISTED.json","FINAL_STATE_001.json")})
    folder = c.HERE / "author_monitor"
    c.save(folder / f"{time.time_ns()}.json", result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
