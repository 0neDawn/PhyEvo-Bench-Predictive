"""One owned worker at a time; preserve other tasks and every completed case."""
import fcntl
import os
import shutil
import signal
import subprocess
import time
import traceback
from pathlib import Path
from . import common as c


def main():
    old = c.install()
    from predictive_tracks.ppr_existing_20261102.resource_resume_004.guarded import hardware
    from predictive_tracks.ppr_20261028.control import gpu_idle
    c.verify()
    (c.HERE / "monitor").mkdir(exist_ok=True)
    lock = (c.HERE / "monitor/queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(at=time.time(), pid=os.getpid(), state="starting", active=None, completed=[])
    child = None
    faults = 0
    def update(**values):
        state.update(values, updated_at=time.time())
        c.status(c.HERE / "monitor/queue.json", state)

    def execute(model, phase):
        nonlocal child, faults
        name = f"{phase}_{model}"
        marker = c.HERE / "models_complete" / f"{name}.json"
        if phase == "native":
            marker = c.HERE / "models_complete" / f"native_{model}.json"
        elif phase == "validation":
            marker = c.HERE / "stage2_complete" / f"native_{model}.json"
        if marker.exists():
            return
        while True:
            stable = 0
            while stable < 3:
                okay, detail = old.gate(model)
                stable = stable+1 if okay else 0
                update(state="checking_resources" if okay else "waiting_resources", active=None, next=name, gate=detail)
                if stable < 3:
                    time.sleep(3 if okay else 20)
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                       CUBLAS_WORKSPACE_CONFIG=":4096:8", MPLCONFIGDIR=str(c.HERE / "mplcache"))
            module = "tail_worker" if phase.startswith("tail") else "native_worker"
            args = [c.PYTHON,"-B","-u","-m",f"predictive_tracks.ppr_internal_activation_20261106_r.{module}","--model",model]
            if phase == "tail_stage2":
                args.append("--stage2")
            elif phase == "validation":
                args.append("--validation")
            log = c.HERE / "logs" / f"{name}_{time.time_ns()}.log"
            log.parent.mkdir(exist_ok=True)
            emergency = competing = probe_failure = False
            with log.open("x") as stream:
                child = subprocess.Popen(args, cwd=c.ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                update(state="running", active=dict(model=model, phase=phase, pid=child.pid, log=str(log)), next=None)
                while child.poll() is None:
                    try:
                        h = hardware(child.pid)
                        idle, gpu = gpu_idle(child.pid)
                    except subprocess.TimeoutExpired as error:
                        probe_failure = True
                        c.save(c.HERE / "events" / f"probe_timeout_{time.time_ns()}.json", dict(at=time.time(), error=repr(error), action="Terminate only own worker; retain completed cases"))
                        child.terminate()
                        break
                    c.status(c.HERE / "monitor/hardware_latest.json", h)
                    peak_path = c.HERE / "monitor/resource_peaks.json"
                    peak = c.read(peak_path) if peak_path.exists() else {}
                    for key in ("process_rss_mib", "gpu_used_mib", "gpu_temperature_c", "cpu_temperature_c"):
                        peak["max_"+key] = max(peak.get("max_"+key, 0), h[key])
                    peak["min_available_ram_mib"] = min(peak.get("min_available_ram_mib",1e9),h["available_ram_mib"])
                    c.status(peak_path,peak)
                    emergency = h["available_ram_mib"] < 10240 or h["process_rss_mib"] > 8192 or h["gpu_free_mib"] < 7168 or h["gpu_temperature_c"] >= 68 or h["cpu_temperature_c"] >= 75
                    competing = not idle or not old.priority_complete()
                    if emergency or competing:
                        child.terminate()
                        c.save(c.HERE / "events" / f"yield_{time.time_ns()}.json", dict(at=time.time(), model=model, phase=phase,
                            hardware=h, foreign_gpu=gpu, resource_failure=emergency, competing=competing,
                            action="Only own worker terminated; all completed records retained; foreign tasks unchanged"))
                        break
                    time.sleep(3)
                code = child.wait()
                child = None
            if code == 75 or emergency or competing or probe_failure or code == -signal.SIGKILL:
                faults += int(emergency or probe_failure or code == -signal.SIGKILL)
                update(state="resource_pause", active=None, faults=faults)
                if faults >= 3:
                    raise RuntimeError("Three resource/probe faults; do not raise limits or interfere with other tasks")
                time.sleep(20)
                continue
            if code or not marker.exists():
                raise RuntimeError(f"Worker failed: {name} exit={code}, log={log}")
            state["completed"].append(name)
            update(state="step_complete",active=None)
            subprocess.run(["sync","-f",str(c.HERE)],check=True)
            from .report import interim
            interim()
            return

    try:
        from . import report
        for model in c.FOCUS:
            execute(model,"tail")
        report.make_stage2()
        for model in c.FOCUS:
            execute(model,"tail_stage2")
        report.make_validation()
        for model in c.FOCUS:
            execute(model,"validation")
        for model in ("ocvp_seq","savi_transformer"):
            execute(model,"native")
        if not (c.HERE / "MODEL_RUNS_COMPLETE.json").exists():
            c.save(c.HERE / "MODEL_RUNS_COMPLETE.json",dict(at=time.time(),all_declared_configurations_complete=True))
        if not (c.HERE / "PERSISTED.json").exists():
            update(state="auditing",active=None)
            report.finish()
            c.save(c.HERE / "PERSISTED.json",dict(at=time.time(),all_experiments_and_audits_complete=True))
        directory = Path(c.read(c.HERE / "FINAL_DELIVERY.json")["directory"])
        if not (c.HERE / "FINAL_STATE_001.json").exists():
            update(state="lineage_audit",active=None)
            c.verify(full=True)
            for path,wanted in c.read(directory / "DELIVERY.json")["files"].items():
                assert c.digest(path) == wanted
            out = directory / "runtime_and_diagnostics_001"
            out.mkdir(exist_ok=True)
            files = list(c.HERE.glob("*.py"))+[p for p in c.HERE.glob("*.json") if p.name not in ("FINAL_STATE_001.json",)]
            files += list((c.HERE / "full_pipeline_preflight").glob("*/*.json"))
            for path in files:
                dest = out / path.relative_to(c.HERE)
                dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists():
                    assert c.digest(path) == c.digest(dest)
                else:
                    with dest.open("xb") as stream:
                        stream.write(path.read_bytes())
            c.save(directory / "SUPPLEMENT_MANIFEST_001.json",dict(at=time.time(), files={str(p):c.digest(p) for p in out.rglob("*") if p.is_file()},
                no_extra_forward_in_deployed_forecast=True, partial_graph_search_disclosed=True))
            c.save(c.HERE / "FINAL_STATE_001.json",dict(at=time.time(),directory=str(directory),all_lineage_audits_complete=True,
                all_declared_configurations_complete=True,every_new_main_row_full600_native_validated=True,same_exposed_data_selection=True))
        if not (directory / "USER_SUMMARY_MANIFEST_001.json").exists():
            update(state="packaging",active=None)
            from .user_summary import main as package
            package()
        update(state="complete",active=None,directory=str(directory),finished=time.time())
        subprocess.run(["sync","-f",str(c.HERE)],check=True)
    except BaseException as error:
        update(state="failed",active=None,error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait()


if __name__ == "__main__":
    main()
