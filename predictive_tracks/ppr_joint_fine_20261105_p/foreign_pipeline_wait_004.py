"""Pause only our waiting supervisor until the existing external queue exits."""
import os
import signal
import subprocess
import time
from pathlib import Path
from . import common as c


def identity(pid):
    try:
        text = (Path('/proc') / str(pid) / 'stat').read_text()
        return text.rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        return None


def main():
    external = 824586
    root = int(subprocess.check_output(['systemctl', '--user', 'show',
        'predict-ppr-p-joint-20261105.service', '-p', 'MainPID', '--value'], text=True).strip())
    assert root and root != external
    command = (Path('/proc') / str(root) / 'cmdline').read_bytes()
    assert b'predictive_tracks.ppr_joint_fine_20261105_p.operational_resume_003' in command
    foreign_cmd = (Path('/proc') / str(external) / 'cmdline').read_bytes()
    assert b'motivation_probe/predictive_diagnostic_200/run_models.py' in foreign_cmd
    state = c.read(c.HERE / 'monitor/queue.json')
    assert state['state'] in ('waiting_resources', 'checking_resources') and not isinstance(state.get('active'), dict)
    own_id, other_id = identity(root), identity(external)
    assert own_id and other_id
    os.kill(root, signal.SIGSTOP)
    record = dict(at=time.time(), own_supervisor_pid=root, own_start_tick=own_id,
        external_pipeline_pid=external, external_start_tick=other_id,
        only_own_waiting_supervisor_paused=True, foreign_tasks_unchanged=True,
        original_workers_formula_grid_and_resource_limits_unchanged=True)
    resumed = False
    try:
        c.save(c.HERE / f'FOREIGN_PIPELINE_HOLD_{time.time_ns()}.json', record)
        while identity(external) == other_id and identity(root) == own_id:
            c.status(c.HERE / 'monitor/foreign_pipeline_hold.json', dict(record, updated_at=time.time(), state='waiting_external_pipeline_exit'))
            time.sleep(10)
    finally:
        if identity(root) == own_id:
            os.kill(root, signal.SIGCONT)
            resumed = True
        c.status(c.HERE / 'monitor/foreign_pipeline_hold.json', dict(record, updated_at=time.time(),
            state='released', own_supervisor_resumed=resumed, original_GPU_gate_still_required=True))
        print('Own supervisor released to original GPU gate', resumed, flush=True)


if __name__ == '__main__':
    main()
