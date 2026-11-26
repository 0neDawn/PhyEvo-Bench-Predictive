"""New full-scale orchestration helpers. Old source and pilot results read-only."""
import json
from pathlib import Path

from ..integrity import ROOT, HERE as PILOT, digest, verify, write_json

HERE = Path(__file__).resolve().parent


def protected_hashes():
    original = json.loads((PILOT / "preexisting_source_hashes.json").read_text())["files"]
    report = json.loads((PILOT / "runs/pilot_report_001/summary.json").read_text())
    return original | report["final_source_hashes"]


def source_hashes():
    return {str(p): digest(p) for p in sorted(HERE.iterdir())
            if p.is_file() and p.suffix in (".py", ".json")}


def new_run(name):
    if Path(name).name != name:
        raise ValueError("Exclusive simple run name required")
    path = HERE / "runs" / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def original_record_keys():
    from predictive_tracks.controlled_tracks.run import records_and_split
    rows = [{"benchmark": task, **row} for task in ("sp", "ie") for row in records_and_split(task)]
    if len(rows) != 600 or any(sum(r["benchmark"] == t for r in rows) != 300 for t in ("sp", "ie")):
        raise RuntimeError("Original SP300/IE300 denominator changed")
    return {f"{r['benchmark']}_{r['archive_index']:03d}": r for r in rows}
