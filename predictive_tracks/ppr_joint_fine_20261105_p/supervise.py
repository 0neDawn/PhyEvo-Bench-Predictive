"""One original-budget worker at a time, with the unchanged resource limits."""
import fcntl
import os
import signal
import subprocess
import time
import traceback
from . import common as c


def main():
    old = c.install_original_runtime()
    from predictive_tracks.ppr_existing_20261102.resource_resume_004.guarded import hardware
    from predictive_tracks.ppr_20261028.control import gpu_idle
    c.verify()
    assert c.read(c.HERE / "FROZEN_VERIFIED.json")["passed"]
    (c.HERE / "monitor").mkdir(exist_ok=True)
    lock = (c.HERE / "monitor/queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(pid=os.getpid(), at=time.time(), state="starting", completed=[], active=None, failures=[])
    faults = 0

    def update(**values):
        state.update(values, updated_at=time.time())
        c.status(c.HERE / "monitor/queue.json", state)

    def execute(model, cached):
        nonlocal faults
        tag = ("cache_" if cached else "native_") + model
        done = c.HERE / "models_complete" / (tag + ".json")
        if done.exists():
            return
        jobs = [j for j in c.read(c.HERE / "GRID.json")["cached" if cached else "native_new"] if j["model"] == model]
        if not jobs:
            c.save(done, dict(at=time.time(), model=model, settings=0, no_model_initialized=True))
            return
        while True:
            stable = 0
            while stable < 3:
                okay, detail = old.gate(model)
                stable = stable + 1 if okay else 0
                update(state="checking_resources" if okay else "waiting_resources", active=tag, gate=detail)
                if stable < 3:
                    time.sleep(3 if okay else 20)
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                       CUBLAS_WORKSPACE_CONFIG=":4096:8", MPLCONFIGDIR=str(c.HERE / "mplcache"))
            args = [old.python(model), "-B", "-u", "-m", "predictive_tracks.ppr_joint_fine_20261105_p." + ("cache_worker" if cached else "native_worker"), "--model", model]
            log = c.HERE / "logs" / f"{tag}_{time.time_ns()}.log"
            log.parent.mkdir(exist_ok=True)
            emergency = competing = False
            with log.open("x") as stream:
                child = subprocess.Popen(args, cwd=c.ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                update(state="running", active=dict(model=model, cached=cached, pid=child.pid, log=str(log)))
                peakfile = c.HERE / "monitor/resource_peaks.json"
                peak = c.read(peakfile) if peakfile.exists() else {}
                last_sync, sync_job = time.monotonic(), None
                while child.poll() is None:
                    h = hardware(child.pid)
                    c.status(c.HERE / "monitor/hardware_latest.json", h)
                    for key in ("process_rss_mib", "gpu_used_mib", "gpu_temperature_c", "cpu_temperature_c"):
                        peak["max_" + key] = max(peak.get("max_" + key, 0), h[key])
                    peak["min_available_ram_mib"] = min(peak.get("min_available_ram_mib", 1e9), h["available_ram_mib"])
                    c.status(peakfile, peak)
                    large = model in c.LARGE
                    emergency = (h["available_ram_mib"] < 9216 or h["process_rss_mib"] > (10240 if large else 3072)
                                 or h["gpu_free_mib"] < (6144 if large else 8192) or h["gpu_temperature_c"] >= 68 or h["cpu_temperature_c"] >= 75)
                    idle, _ = gpu_idle(child.pid)
                    competing = not idle or not old.priority_complete()
                    if emergency or competing:
                        child.terminate()
                        c.save(c.HERE / "events" / f"yield_{time.time_ns()}.json", dict(at=time.time(), model=model, hardware=h,
                            resource_failure=emergency, competing=competing, action="Only own worker terminated; all saved results retained"))
                        break
                    if time.monotonic()-last_sync >= 300 and (sync_job is None or sync_job.poll() is not None):
                        sync_job = subprocess.Popen(["sync", "-f", str(c.HERE)])
                        last_sync = time.monotonic()
                    time.sleep(2)
                code = child.wait()
                if sync_job is not None:
                    sync_job.wait()
            if code == 75 or emergency or competing or code == -signal.SIGKILL:
                faults += int(emergency or code == -signal.SIGKILL)
                update(state="resource_pause", active=None, hard_resource_faults=faults)
                if faults >= 3:
                    raise RuntimeError("Three hard resource faults; do not raise limits")
                time.sleep(20)
                continue
            if code or not done.exists():
                raise RuntimeError(f"Worker failed: {tag}, returncode={code}, log={log}")
            state["completed"].append(tag)
            update(state="reporting", active=None)
            from .report import interim
            interim()
            subprocess.run(["sync", "-f", str(c.HERE)], check=True)
            return

    try:
        if not (c.HERE / "MODEL_RUNS_COMPLETE.json").exists():
            for model in c.CACHED:
                execute(model, True)
            for model in c.NATIVE:
                execute(model, False)
            c.save(c.HERE / "MODEL_RUNS_COMPLETE.json", dict(at=time.time(), all_declared_joint_configurations_complete=True))
        if not (c.HERE / "PERSISTED.json").exists():
            update(state="auditing", active=None)
            from .report import finish
            finish()
            subprocess.run(["sync", "-f", str(c.HERE)], check=True)
            c.save(c.HERE / "PERSISTED.json", dict(at=time.time(), all_experiments_and_audits_complete=True))
        if not (c.HERE / "FINAL_STATE_001.json").exists():
            update(state="lineage_audit", active=None)
            from .finalize import main as finalize
            finalize()
        directory = c.read(c.HERE / "FINAL_DELIVERY.json")["directory"]
        from pathlib import Path
        if not (Path(directory) / "USER_SUMMARY_MANIFEST_001.json").exists():
            update(state="packaging", active=None)
            from .user_summary import main as package
            package()
        update(state="complete", active=None, finished=time.time(), directory=directory)
        subprocess.run(["sync", "-f", str(c.HERE)], check=True)
    except BaseException as error:
        update(state="failed", active=None, error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
