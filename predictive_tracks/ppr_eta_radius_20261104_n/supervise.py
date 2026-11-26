"""One resumable guarded GPU worker; no background parallel model workloads."""
import fcntl
import os
import signal
import subprocess
import time
from . import common as c

def main():
    from predictive_tracks.ppr_existing_20261102.resource_resume_004.guarded import hardware
    from predictive_tracks.ppr_20261028.control import gpu_idle
    (c.HERE/'monitor').mkdir(exist_ok=True)
    lock=(c.HERE/'monitor/queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    c.verify(full=True)
    state=dict(pid=os.getpid(),at=time.time(),state='starting',completed=[],active=None,failures=[])
    faults=0
    def execute(model,cached=False,stage2=False):
        nonlocal faults
        tag=('cache_' if cached else '')+model
        done=c.HERE/('stage2_complete' if stage2 else 'models_complete')/f'{tag}.json'
        if done.exists():return
        while True:
            stable=0
            while stable<3:
                okay,d=c.gate(model);stable=stable+1 if okay else 0
                state.update(state='checking_resources' if okay else 'waiting_resources',active=tag,gate=d)
                c.status(c.HERE/'monitor/queue.json',state)
                if stable<3:time.sleep(3 if okay else 20)
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                CUBLAS_WORKSPACE_CONFIG=':4096:8',MPLCONFIGDIR=str(c.HERE/'mplcache'))
            module='cache_worker' if cached else 'native_worker'
            args=[c.PYTHON,'-B','-u','-m','predictive_tracks.ppr_eta_radius_20261104_n.'+module]
            if not cached:args+=['--model',model]
            if stage2:args+=['--stage2']
            log=c.HERE/'logs'/f'{tag}_{"stage2" if stage2 else "stage1"}_{time.time_ns()}.log';log.parent.mkdir(exist_ok=True)
            emergency=competing=False
            with log.open('x') as f:
                child=subprocess.Popen(args,cwd=c.ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
                state.update(state='running',active=dict(model=model,cached=cached,stage2=stage2,pid=child.pid,log=str(log)))
                c.status(c.HERE/'monitor/queue.json',state)
                peaksfile=c.HERE/'monitor/resource_peaks.json';peaks=c.read(peaksfile) if peaksfile.exists() else {}
                while child.poll() is None:
                    h=hardware(child.pid);c.status(c.HERE/'monitor/hardware_latest.json',h)
                    for k in ('process_rss_mib','gpu_used_mib','gpu_temperature_c','cpu_temperature_c'):
                        peaks['max_'+k]=max(peaks.get('max_'+k,0),h[k])
                    peaks['min_available_ram_mib']=min(peaks.get('min_available_ram_mib',1e9),h['available_ram_mib']);c.status(peaksfile,peaks)
                    large=model=='videomaev2'
                    emergency=(h['available_ram_mib']<9216 or h['process_rss_mib']>(10240 if large else 3072)
                        or h['gpu_free_mib']<(6144 if large else 8192) or h['gpu_temperature_c']>=68 or h['cpu_temperature_c']>=75)
                    idle,_=gpu_idle(child.pid);competing=not idle or not c.priority_complete()
                    if emergency or competing:
                        child.terminate();c.save(c.HERE/'events'/f'yield_{time.time_ns()}.json',dict(at=time.time(),model=model,hardware=h,
                            resource_failure=emergency,competing=competing,action='Only own worker terminated; completed files retained'))
                        break
                    time.sleep(2)
                code=child.wait()
            if code==75 or emergency or competing or code==-signal.SIGKILL:
                faults+=int(emergency or code==-signal.SIGKILL)
                state.update(state='resource_pause',active=None,hard_resource_faults=faults);c.status(c.HERE/'monitor/queue.json',state)
                if faults>=3:raise RuntimeError('Repeated hard resource faults; do not raise limits')
                time.sleep(20);continue
            if code or not done.exists():raise RuntimeError(f'Worker failed: {tag}, returncode={code}, log={log}')
            state['completed'].append(dict(tag=tag,stage2=stage2));state.update(state='reporting',active=None);c.status(c.HERE/'monitor/queue.json',state)
            from .report import interim
            interim();subprocess.run(['sync','-f',str(c.HERE)],check=True);return
    try:
        execute('videomaev2',cached=True)
        for m in c.MODELS:execute(m)
        if not (c.HERE/'STAGE1_COMPLETE.json').exists():c.save(c.HERE/'STAGE1_COMPLETE.json',dict(at=time.time(),settings=240))
        state.update(state='bounded_refinement',active=None);c.status(c.HERE/'monitor/queue.json',state)
        from .refine import main as refine
        refine()
        execute('videomaev2',cached=True,stage2=True)
        for m in c.MODELS:execute(m,stage2=True)
        c.verify(full=True)
        if not (c.HERE/'MODEL_RUNS_COMPLETE.json').exists():c.save(c.HERE/'MODEL_RUNS_COMPLETE.json',dict(at=time.time(),all_stage1_and_bounded_stage2_complete=True))
        state.update(state='auditing',active=None);c.status(c.HERE/'monitor/queue.json',state)
        from .report import finish
        finish();subprocess.run(['sync','-f',str(c.HERE)],check=True)
        c.save(c.HERE/'PERSISTED.json',dict(at=time.time(),all_experiments_and_final_audits_complete=True))
        state.update(state='complete',finished=time.time());c.status(c.HERE/'monitor/queue.json',state)
    except BaseException as exc:
        import traceback
        state.update(state='failed',active=None,error=repr(exc),traceback=traceback.format_exc());c.status(c.HERE/'monitor/queue.json',state);raise

if __name__=='__main__':main()
