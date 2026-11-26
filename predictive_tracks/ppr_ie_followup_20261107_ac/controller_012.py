"""Continue after slot/probe workers, retaining every result and safety stop."""
import fcntl
import os
import subprocess
import time
import traceback
from . import common as c


def main():
    c.verify()
    lock = (c.HERE/'controller_012.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old = c.install()
    from predictive_tracks.ppr_existing_20261102.resource_resume_004.guarded import hardware
    from predictive_tracks.ppr_20261028.control import gpu_idle
    def update(**v):
        c.status(c.HERE/'monitor/queue.json',dict(at=time.time(),**v))
    def execute(model,phase):
        marker = c.HERE/'models_complete'/f'{phase}_{model}.json'
        if marker.exists():return
        while True:
            okay,detail = old.gate(model)
            update(state='checking_resources' if okay else 'waiting_resources',active=None,next=dict(model=model,phase=phase),gate=detail)
            if okay:break
            time.sleep(20)
        module = 'native_worker_009' if phase == 'validation' else 'worker_008'
        args = [c.PYTHON,'-B','-u','-m',f'predictive_tracks.ppr_ie_followup_20261107_ac.{module}','--model',model]
        if phase != 'validation':args.extend(['--phase',phase])
        log = c.HERE/'logs'/f'{phase}_{model}_{time.time_ns()}.log'
        env = dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
            CUBLAS_WORKSPACE_CONFIG=':4096:8',MPLCONFIGDIR=str(c.PARENT/'mplcache'))
        with log.open('x') as stream:
            child = subprocess.Popen(args,cwd=c.ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            update(state='running',active=dict(model=model,phase=phase,pid=child.pid,log=str(log)))
            while child.poll() is None:
                try:
                    h = hardware(child.pid);idle,gpu = gpu_idle(child.pid)
                    c.status(c.HERE/'monitor/hardware_latest.json',h)
                    large = model == 'dino_foresight'
                    hard = (h['available_ram_mib']<12288 or h['process_rss_mib']>(7168 if large else 3072)
                        or h['gpu_free_mib']<(7168 if large else 10240) or h['gpu_temperature_c']>=68 or h['cpu_temperature_c']>=75)
                    competing = not idle or not old.priority_complete()
                except BaseException as error:
                    h = dict(error=repr(error));hard=True;competing=False
                if hard or competing:
                    child.terminate();child.wait()
                    c.save(c.HERE/'events'/f'controller_stop_{time.time_ns()}.json',dict(at=time.time(),model=model,phase=phase,hardware=h,
                        hard_resource_fault=hard,competing=competing,action='Stop only this worker, preserve all checkpoints; no limit increase'))
                    raise RuntimeError('Follow-up resource/task stop; inspect before resuming')
                time.sleep(3)
            code = child.wait()
        if code or not marker.exists():raise RuntimeError(f'{model}/{phase} failed {code}; see {log}')
    try:
        while not (c.HERE/'SLOT_RUNS_COMPLETE_004.json').exists():
            state = c.read(c.HERE/'monitor/slots_004.json')
            if state.get('state')=='failed':raise RuntimeError('Owned slot coarse queue failed')
            update(state='waiting_slot_coarse004',active=None)
            time.sleep(20)
        while not (c.HERE/'PROBE_COMPLETE_005.json').exists():
            state = c.read(c.HERE/'monitor/probe.json')
            wait = c.read(c.HERE/'monitor/probe_006_wait.json') if (c.HERE/'monitor/probe_006_wait.json').exists() else {}
            # The old001 failure remains archived; wait006 has not launched the new probe yet.
            if state.get('state')=='failed' and state.get('at',0)>wait.get('at',float('inf')):
                raise RuntimeError('New DINO raw-state/scoring probe failed; no new baseline fabricated')
            update(state='waiting_DINO_native_state_and_score_probe005',active=None)
            time.sleep(20)
        execute('dino_foresight','coarse')
        from . import report_010 as report
        report.make_fine()
        for model in c.MODELS:execute(model,'fine')
        report.make_validation()
        for model in c.MODELS:execute(model,'validation')
        if not (c.HERE/'MODEL_RUNS_COMPLETE.json').exists():
            c.save(c.HERE/'MODEL_RUNS_COMPLETE.json',dict(at=time.time(),models=list(c.MODELS),all_declared_search_and_native_runs_complete=True))
        update(state='auditing',active=None)
        if not (c.HERE/'FINAL_STATE_001.json').exists():
            report.finish()
        if not (c.HERE/'COMPLETED_012.json').exists():
            c.save(c.HERE/'COMPLETED_012.json',dict(at=time.time(),all_experiments_and_audits_complete=True,directory=str(c.read(c.HERE/'FINAL_DELIVERY.json')['directory'])))
        update(state='complete',active=None,directory=c.read(c.HERE/'FINAL_DELIVERY.json')['directory'])
    except BaseException as error:
        update(state='failed',active=None,error=repr(error),traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
