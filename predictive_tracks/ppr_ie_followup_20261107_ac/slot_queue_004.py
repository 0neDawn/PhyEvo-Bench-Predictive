"""Serial IE follow-up for the two slot models, with unchanged safety limits."""
import fcntl
import os
import subprocess
import time
from . import common as c


def main():
    c.verify_sources('SLOT_READY_004.json')
    lock = (c.HERE/'slot_queue_004.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old = c.install()
    from predictive_tracks.ppr_existing_20261102.resource_resume_004.guarded import hardware
    from predictive_tracks.ppr_20261028.control import gpu_idle
    for model in c.SLOTS:
        if (c.HERE/'models_complete'/f'coarse_{model}.json').exists():
            continue
        while True:
            okay,detail = old.gate(model)
            c.status(c.HERE/'monitor/slots_004.json',dict(at=time.time(),state='checking_resources' if okay else 'waiting_resources',model=model,gate=detail))
            if okay:
                break
            time.sleep(20)
        log = c.HERE/'logs'/f'slot_{model}_{time.time_ns()}.log'
        log.parent.mkdir(exist_ok=True)
        env = dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
            CUBLAS_WORKSPACE_CONFIG=':4096:8',MPLCONFIGDIR=str(c.PARENT/'mplcache'))
        with log.open('x') as stream:
            child = subprocess.Popen([c.PYTHON,'-B','-u','-m','predictive_tracks.ppr_ie_followup_20261107_ac.slot_worker_004','--model',model],cwd=c.ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            c.status(c.HERE/'monitor/slots_004.json',dict(at=time.time(),state='running',model=model,pid=child.pid,log=str(log)))
            while child.poll() is None:
                try:
                    h = hardware(child.pid)
                    idle, gpu = gpu_idle(child.pid)
                    c.status(c.HERE/'monitor/slots_hardware_004.json',h)
                    unsafe = (h['available_ram_mib']<12288 or h['process_rss_mib']>3072 or h['gpu_free_mib']<10240
                        or h['gpu_temperature_c']>=68 or h['cpu_temperature_c']>=75 or not idle or not old.priority_complete())
                except BaseException as error:
                    h = dict(error=repr(error))
                    unsafe = True
                if unsafe:
                    child.terminate();child.wait()
                    c.save(c.HERE/'events'/f'slot_stop_{time.time_ns()}.json',dict(at=time.time(),model=model,hardware=h))
                    c.status(c.HERE/'monitor/slots_004.json',dict(at=time.time(),state='failed',model=model,error='Resource protection; no limit increase'))
                    raise RuntimeError('Slot follow-up resource stop; records preserved')
                time.sleep(3)
            code = child.wait()
        if code:
            c.status(c.HERE/'monitor/slots_004.json',dict(at=time.time(),state='failed',model=model,exit_code=code,log=str(log)))
            raise RuntimeError('Slot follow-up failed; inspect preserved log')
    c.save(c.HERE/'SLOT_RUNS_COMPLETE_004.json',dict(at=time.time(),cases_each=600,models=list(c.SLOTS)))
    c.status(c.HERE/'monitor/slots_004.json',dict(at=time.time(),state='coarse_complete',models=list(c.SLOTS),native_selection_validation_pending=True))


if __name__ == '__main__':
    main()
