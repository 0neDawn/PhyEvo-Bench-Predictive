"""Lifecycle cleanup only: original frozen supervisor, workers and probes."""
import os
import signal
import subprocess
import time
from pathlib import Path
from . import common as c


def main():
    from predictive_tracks.ppr_existing_20261102.resource_resume_004 import guarded
    original = guarded.hardware
    observed_child = [None]

    def hardware(pid=None):
        if pid is not None:
            observed_child[0] = pid
        return original(pid)

    def cleanup():
        pid = observed_child[0]
        if pid is None:
            return
        proc = Path('/proc') / str(pid)
        try:
            cmd = (proc / 'cmdline').read_bytes()
            status = (proc / 'status').read_text().splitlines()
            parent = int(next(v for v in status if v.startswith('PPid:')).split()[1])
        except (OSError, StopIteration):
            return
        own = parent == os.getpid() and any(name in cmd for name in (
            b'predictive_tracks.ppr_joint_fine_20261105_p.native_worker',
            b'predictive_tracks.ppr_joint_fine_20261105_p.cache_worker'))
        if own:
            os.kill(pid, signal.SIGTERM)

    guarded.hardware = hardware
    try:
        from .supervise import main as original_supervisor
        original_supervisor()
    except subprocess.TimeoutExpired as error:
        cleanup()
        c.save(c.HERE / 'events' / f'probe_timeout_{time.time_ns()}.json', dict(at=time.time(),
            error=repr(error), action='Stop only own worker; exit75 for bounded operational restart',
            original_probe_timeout_unchanged=True, original_science_and_results_unchanged=True))
        raise SystemExit(75)
    finally:
        cleanup()


if __name__ == '__main__':
    main()
