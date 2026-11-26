"""Fail-closed resource and task gate. No GPU allocations, only nvidia-smi."""
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
THREAD_ID = "019f7e4b-0e84-7991-8dd6-4a9a2fba7e27"
HOST_ID = "remote-ssh-discovered:codex-server"
ANNOTATION = HERE / "monitor/annotation.json"


def write_status(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def task_complete():
    try:
        state = json.loads(ANNOTATION.read_text())
    except (OSError, ValueError):
        return False, "no verified annotation task state"
    # A completed turn can be blocked or require user input; separate semantic
    # verification by the supervising agent is mandatory before setting true.
    if state.get("thread_id") != THREAD_ID or state.get("host_id") != HOST_ID:
        return False, "wrong annotation task"
    if not state.get("task_complete_verified") or state.get("turn_status") != "completed":
        return False, "annotation task unfinished"
    if time.time() - state.get("checked_at", 0) > 150:
        return False, "annotation task state stale"
    return True, "annotation completion verified"


def gpu_idle(own_pid=None):
    try:
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True, timeout=10)
        gpu = [float(s.strip()) for s in output.splitlines()[0].split(",")]
        processes = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits"], text=True, timeout=10)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return False, {"error": repr(exc)}
    allowed = {"xdg-desktop-portal-gnome", "nautilus", "gnome-text-editor", "ptyxis", "resources"}
    foreign = []
    for line in processes.splitlines():
        fields = line.split(",")
        if len(fields) < 3:
            return False, {"error": "unparseable GPU process record"}
        pid, command = int(fields[0]), fields[1].strip()
        executable = command.split(" --", 1)[0]
        graphical = Path(executable).name in allowed
        graphical |= "--type=gpu-process" in command and Path(executable).name in {"claude-desktop", "chrome", "codex"}
        if pid != own_pid and not graphical:
            foreign.append({"pid": pid, "executable": executable})
    state = {"index": int(gpu[0]), "total_mib": gpu[1], "used_mib": gpu[2],
             "utilization": gpu[3], "foreign_compute": foreign}
    # Once our model is loaded, its allocation is allowed; foreign jobs are not.
    enough_free = own_pid is not None or (gpu[1] - gpu[2] >= 12000 and gpu[3] <= 10)
    return not foreign and enough_free, state


def await_gate(own_pid=None, consecutive=1):
    stable = 0
    while stable < consecutive:
        complete, reason = task_complete()
        idle, gpu = gpu_idle(own_pid)
        stable = stable + 1 if complete and idle else 0
        write_status(HERE / "monitor/gate.json", {
            "checked_at": time.time(), "task_complete": complete, "reason": reason,
            "gpu_idle": idle, "gpu": gpu, "stable_checks": stable,
            "own_pid": own_pid})
        if stable < consecutive:
            time.sleep(30)


if __name__ == "__main__":
    complete, reason = task_complete()
    idle, gpu = gpu_idle()
    print(json.dumps({"task_complete": complete, "reason": reason, "gpu_idle": idle, "gpu": gpu}))
