"""Read-only inventory of pre-existing source; exclusive new-directory writes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def write_json(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(HERE):
        raise ValueError("New artifacts must stay in the new experiment directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def old_inventory():
    from predictive_tracks.controlled_tracks.run import frozen_files
    protected = frozen_files()
    plan = ROOT / "predictive_tracks/controlled_tracks/confirmation_plan.json"
    protected.update(json.loads(plan.read_text())["locked_source_hashes"])
    protected[str(plan)] = digest(plan)
    # Include active source/configs across the workspace, but avoid enormous data,
    # cached runs and checkpoint trees. Their archived protected manifests remain
    # covered above. No file in any of those directories is written by this track.
    skip = {".git", "__pycache__", "data", "datasets_cache", "runs", "result",
            "results", "checkpoints", "pretrained", "node_modules", ".venv",
            "PHYDATA", "eval_results", "lpwm_intphys_pairwise_eval"}
    extensions = {".py", ".sh", ".json", ".yaml", ".yml", ".toml", ".md"}
    for directory, children, files in os.walk(ROOT):
        children[:] = [name for name in children if name not in skip
                       and Path(directory, name).resolve() != HERE
                       and not Path(directory, name).is_symlink()]
        for name in files:
            path = Path(directory, name)
            if path.suffix in extensions and not path.is_symlink():
                protected.setdefault(str(path), digest(path))
    verify(protected)
    return protected


def verify(inventory):
    for filename, expected in inventory.items():
        path = Path(filename)
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"Protected source changed: {path}")


def source_hashes():
    return {str(p): digest(p) for p in sorted(HERE.glob("*.py"))} | {
        str(HERE / "plan_v1.json"): digest(HERE / "plan_v1.json")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("snapshot", "verify"))
    args = parser.parse_args()
    path = HERE / "preexisting_source_hashes.json"
    if args.action == "snapshot":
        hashes = old_inventory()
        write_json(path, {"scope": "Pre-existing active source/configs plus original archive and confirmation protected files", "files": hashes})
    else:
        hashes = json.loads(path.read_text())["files"]
        verify(hashes)
    print(args.action, len(hashes), "protected files verified", flush=True)


if __name__ == "__main__":
    main()
