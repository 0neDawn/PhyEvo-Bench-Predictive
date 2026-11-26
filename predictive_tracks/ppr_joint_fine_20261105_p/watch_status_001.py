"""Read-only live progress; never run a model or modify experiment results."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    now = time.time()
    queue = read(HERE / "monitor/queue.json") if (HERE / "monitor/queue.json").exists() else {}
    grid = read(HERE / "GRID.json") if (HERE / "GRID.json").exists() else dict(cached=[], native_new=[])
    jobs = grid["cached"] + grid["native_new"]
    done = sum((HERE / "runs" / j["model"] / j["config_id"] / "summary.json").exists() for j in jobs)
    services = {}
    for unit in ("predict-ppr-p-analysis-20261105.service", "predict-ppr-p-joint-20261105.service"):
        raw = subprocess.run(["systemctl", "--user", "show", unit, "--property=ActiveState", "--property=SubState", "--property=Result", "--property=MainPID", "--property=MemoryCurrent"], capture_output=True, text=True)
        services[unit] = dict(line.split("=", 1) for line in raw.stdout.splitlines() if "=" in line)
    active = queue.get("active")
    progress = []
    expected = None
    if isinstance(active, dict):
        model = active["model"]
        pid = active["pid"]
        try:
            command = Path(f"/proc/{pid}/cmdline").read_bytes()
            expected = b"predictive_tracks.ppr_joint_fine_20261105_p." in command and b"--model\x00" + model.encode() in command
        except FileNotFoundError:
            expected = False
        if active["cached"]:
            p = HERE / "cache" / model / "progress.json"
            if p.exists():
                progress.append(read(p))
        else:
            for j in grid["native_new"]:
                if j["model"] != model:
                    continue
                folder = HERE / "runs" / model / j["config_id"]
                if (folder / "summary.json").exists():
                    continue
                paths = list((folder / "pairs").glob("*.json"))
                if paths:
                    latest = max(paths, key=lambda x: x.stat().st_mtime_ns)
                    progress.append(dict(eta=j["eta"], alpha=j["alpha"], completed=len(paths), total=600, last=latest.stem, at=latest.stat().st_mtime))
    available = next(int(line.split()[1])/1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    result = dict(at=now, state=queue.get("state", "analysis"), active=active, complete=done, planned=len(jobs), progress=progress,
        services=services, available_ram_mib=available, gpu_memory_temperature_utilization=gpu.stdout.strip(), active_pid_is_expected_worker=expected,
        markers={n: (HERE/n).exists() for n in ("ANALYSIS_COMPLETE.json", "GRID_FROZEN.json", "MODEL_RUNS_COMPLETE.json", "PERSISTED.json", "FINAL_STATE_001.json")})
    if queue.get("error"):
        result["error"] = queue["error"]
    if args.save:
        out = HERE / "author_monitor_001"
        out.mkdir(exist_ok=True)
        with (out / f"{time.time_ns()}.json").open("x") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
