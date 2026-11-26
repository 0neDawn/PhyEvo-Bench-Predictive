"""Read-only experiment/process monitor. No model or scientific code imports."""
import argparse
import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
UNITS = ("predict-ppr-eta-radius-o-lineage-20261104.service",
         "predict-ppr-eta-radius-o-post-20261104.service")


def read(path):
    return json.loads(path.read_text())


def snapshot():
    queue = read(HERE / "monitor/queue.json")
    hardware = read(HERE / "monitor/hardware_latest.json")
    active = queue.get("active")
    services = {}
    for unit in UNITS:
        result = subprocess.run(["systemctl", "--user", "show", unit,
                                 "-p", "ActiveState", "-p", "SubState", "-p", "Result",
                                 "-p", "MainPID", "-p", "MemoryCurrent"],
                                capture_output=True, text=True, timeout=15, check=True)
        services[unit] = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    counts, progress = {}, []
    for stage, name in ((1, "GRID.json"), (2, "STAGE2.json")):
        path = HERE / name
        if not path.exists():
            continue
        grid = read(path)
        jobs = grid["cached"] + grid["native_new"]
        completed = 0
        for job in jobs:
            folder = HERE / "runs" / job["model"] / job["config_id"]
            if (folder / "summary.json").exists():
                completed += 1
            elif isinstance(active, dict) and bool(active["stage2"]) == (stage == 2) and active["model"] == job["model"] and not active["cached"]:
                path = folder / "progress.json"
                if path.exists():
                    progress.append(dict(stage=stage, eta=job["eta"], alpha=job["alpha"], **read(path)))
        counts[str(stage)] = dict(complete=completed, planned=len(jobs))
    if isinstance(active, dict) and active["cached"]:
        folder = HERE / ("cache_stage2" if active["stage2"] else "cache") / active["model"]
        if (folder / "progress.json").exists():
            progress.append(dict(stage=2 if active["stage2"] else 1, cached=True, **read(folder / "progress.json")))
    markers = {name: (HERE / name).exists() for name in (
        "STAGE1_COMPLETE.json", "MODEL_RUNS_COMPLETE.json", "PERSISTED.json", "FINAL_STATE_001.json")}
    result = dict(at=time.time(), state=queue["state"],
                  active_model=active.get("model") if isinstance(active, dict) else active,
                  active_pid=active.get("pid") if isinstance(active, dict) else None,
                  stage_counts=counts, progress=progress, services=services,
                  resources={k: hardware[k] for k in ("available_ram_mib", "process_rss_mib", "gpu_used_mib", "gpu_temperature_c", "cpu_temperature_c")},
                  hardware_age_seconds=round(time.time() - hardware["at"], 1), markers=markers)
    if queue["state"] == "failed":
        result["error"] = queue.get("error")
    if isinstance(active, dict):
        try:
            cmd = Path(f'/proc/{active["pid"]}/cmdline').read_bytes().split(b"\0")
            result["active_pid_is_expected_worker"] = any(b"ppr_eta_radius_20261104_o." in arg for arg in cmd)
        except FileNotFoundError:
            result["active_pid_is_expected_worker"] = False
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    result = snapshot()
    if args.save:
        folder = HERE / "author_monitor_001"
        folder.mkdir(exist_ok=True)
        with (folder / f"{time.time_ns()}.json").open("x") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
