"""Serial queue, dormant until both annotation completion and GPU idle gates pass."""

from release_paths import runtime_python as _release_python, project_path as _release_path
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from .control import HERE, ROOT, await_gate, write_status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    if Path(args.run).name != args.run:
        raise ValueError("Simple run name required")
    monitor = HERE / "monitor"
    monitor.mkdir(exist_ok=True)
    lock = (monitor / "launcher.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tasks = [(model, "audit") for model in ("ddlp", "gswm")]
    tasks += [(model, mode) for mode in ("feedback", "output_only") for model in ("ddlp", "gswm")]
    status = {"pid": os.getpid(), "started_at": time.time(), "run": args.run,
              "state": "waiting_annotation_and_gpu", "queue": tasks, "completed": [], "failures": []}
    write_status(monitor / "launcher.json", status)
    failed_models = set()
    for model, mode in tasks:
        if model in failed_models:
            continue
        existing = HERE / "runs" / args.run / model / mode
        if (existing / "summary.json").exists():
            status["completed"].append([model, mode])
            continue
        if existing.exists():
            failed_models.add(model)
            status["failures"].append({"model": model, "mode": mode, "reason": "partial run retained; inspect before retry"})
            continue
        await_gate(consecutive=3)
        # Recheck after the waiting period: author may cancel by removing READY.
        if not (monitor / "READY.json").exists():
            status["state"] = "waiting_implementation_readiness"
            write_status(monitor / "launcher.json", status)
            while not (monitor / "READY.json").exists():
                time.sleep(30)
            await_gate(consecutive=3)
        readiness = json.loads((monitor / "READY.json").read_text())
        if not readiness.get("cpu_preflight_passed"):
            raise RuntimeError("Implementation not ready")
        for filename, expected in readiness["sources"].items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Source changed after CPU preflight: {filename}")
        if hashlib.sha256((HERE / "PLAN.txt").read_bytes()).hexdigest() != readiness["plan_sha256"]:
            raise RuntimeError("Experiment plan changed after CPU preflight")
        python = _release_python('ddlp' if model == 'ddlp' else 'other')
        cmd = [python, "-B", "-m", "predictive_tracks.ppr_20261028.run", "--model", model,
               "--mode", mode, "--run", args.run]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        log_path = monitor / f"{args.run}_{model}_{mode}.log"
        with log_path.open("a") as log:
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            status.update(state="running", active={"model": model, "mode": mode, "pid": process.pid,
                                                   "log": str(log_path), "started_at": time.time()})
            write_status(monitor / "launcher.json", status)
            code = process.wait()
        if code:
            status["failures"].append({"model": model, "mode": mode, "returncode": code})
            failed_models.add(model)
        else:
            status["completed"].append([model, mode])
        status.update(state="waiting_annotation_and_gpu", active=None)
        write_status(monitor / "launcher.json", status)
    status.update(state="finished_with_failures" if status["failures"] else "complete", finished_at=time.time())
    write_status(monitor / "launcher.json", status)


if __name__ == "__main__":
    main()
