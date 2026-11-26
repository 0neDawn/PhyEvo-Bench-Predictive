"""Operational limits only. The frozen experiment and its parameter grid stay intact."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from ..resource_resume_001.io_guard import BASE,EXPERIMENT,write_json,install

def cpu_temperature():
    values=[]
    for directory in Path('/sys/class/hwmon').glob('hwmon*'):
        name=directory/'name'
        if name.exists() and name.read_text().strip() in ('coretemp','k10temp'):
            for path in directory.glob('temp*_input'):
                try:values.append(float(path.read_text())/1000)
                except (OSError,ValueError):pass
    if not values:raise RuntimeError('CPU temperature sensor unavailable; do not run unmonitored')
    return max(values)

def hardware(pid=None):
    mem={k:int(v.strip().split()[0]) for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
    out=subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,memory.used,memory.free,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True,timeout=10)
    total,used,free,temp,power=[float(x.strip()) for x in out.splitlines()[0].split(',')]
    rss=0.
    if pid is not None:
        try:
            line=next(x for x in Path(f'/proc/{pid}/status').read_text().splitlines() if x.startswith('VmRSS:'));rss=float(line.split()[1])/1024
        except (OSError,StopIteration):pass
    return {'at':time.time(),'available_ram_mib':mem['MemAvailable']/1024,'swap_used_mib':(mem['SwapTotal']-mem['SwapFree'])/1024,
        'process_rss_mib':rss,'gpu_total_mib':total,'gpu_used_mib':used,'gpu_free_mib':free,'gpu_temperature_c':temp,'gpu_power_w':power,'cpu_temperature_c':cpu_temperature()}

def guarded_common(worker=False):
    common=install();base_gate=common.gate
    def gate(own_pid=None):
        allowed,detail=base_gate(os.getpid() if worker else own_pid)
        h=hardware(os.getpid() if worker else own_pid)
        okay=h['available_ram_mib']>=12288 and h['process_rss_mib']<=3072 and h['gpu_free_mib']>=10240 and h['gpu_temperature_c']<=62 and h['cpu_temperature_c']<=70
        detail.update(hardware=h,operational_limits_passed=okay)
        write_json(EXPERIMENT/'monitor/resource_guard.json',detail,True)
        if worker and allowed and okay:time.sleep(.2)
        return allowed and okay,detail
    common.gate=gate
    return common

def worker():
    common=guarded_common(True)
    try:
        common.require_resources(os.getpid())
        import torch
        # Device allocation ceiling only; no precision, layer, step or sampling change.
        torch.cuda.set_per_process_memory_fraction(.2,device=0)
        from ..run_gpu import execute
        execute('gswm')
    except common.ResourceBusy as exc:
        try:detail=json.loads(str(exc))
        except ValueError:detail={}
        write_json(BASE/'events'/f'pause_{time.time_ns()}.json',{'at':time.time(),'reason':str(exc),
            'resource_failure':detail.get('operational_limits_passed') is False,'action':'Worker exits and releases its allocation'});sys.exit(75)
    except BaseException as exc:
        oom='out of memory' in str(exc).lower() or exc.__class__.__name__=='OutOfMemoryError'
        write_json(BASE/'events'/f'worker_error_{time.time_ns()}.json',{'at':time.time(),'error':repr(exc),'traceback':traceback.format_exc(),'resource_failure':oom})
        if oom:sys.exit(75)
        raise

def supervise():
    import fcntl
    common=guarded_common(False)
    lock=(EXPERIMENT/'monitor/gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    common.verify(full=True)
    state={'pid':os.getpid(),'started':time.time(),'state':'starting','completed':['ddlp'],'failures':[],'active':None,
        'queue':['gswm'],'alpha_fixed':1,'resource_limits':{'host_memory_high_gib':3,'host_memory_max_gib':4,'gpu_allocator_fraction':.2,'cpu_quota_cores':1},
        'recovered_run_directory':str(BASE/'runs')}
    common.status(EXPERIMENT/'monitor/gpu.json',state)
    attempts=0
    last_sync=time.monotonic();sync_job=None
    while True:
        stable=0
        while stable<2:
            okay,detail=common.gate();stable=stable+1 if okay else 0
            state.update(state='checking_resources' if okay else 'waiting_resources',active={'model':'gswm'},last_resource_check=detail)
            common.status(EXPERIMENT/'monitor/gpu.json',state)
            if stable<2:time.sleep(5 if okay else 20)
        log=BASE/'logs'/f'gswm_{time.time_ns()}.log';log.parent.mkdir(exist_ok=True)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',MPLCONFIGDIR=str(BASE/'mplcache'))
        with log.open('x') as f:
            child=subprocess.Popen([sys.executable,'-B','-u','-m','predictive_tracks.ppr_existing_20261102.resource_resume_004.entry','--worker'],cwd=common.ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            state.update(state='running',active={'model':'gswm','pid':child.pid,'log':str(log)});common.status(EXPERIMENT/'monitor/gpu.json',state)
            emergency=False
            peak_path=BASE/'resource_peaks.json'
            peaks=common.read(peak_path) if peak_path.exists() else {}
            while child.poll() is None:
                h=hardware(child.pid);common.status(BASE/'hardware_latest.json',h)
                peaks.update(max_process_rss_mib=max(peaks.get('max_process_rss_mib',0),h['process_rss_mib']),
                    max_gpu_used_mib=max(peaks.get('max_gpu_used_mib',0),h['gpu_used_mib']),
                    min_available_ram_mib=min(peaks.get('min_available_ram_mib',float('inf')),h['available_ram_mib']),
                    max_gpu_temperature_c=max(peaks.get('max_gpu_temperature_c',0),h['gpu_temperature_c']),
                    max_cpu_temperature_c=max(peaks.get('max_cpu_temperature_c',0),h['cpu_temperature_c']),at=time.time())
                common.status(peak_path,peaks)
                critical=h['available_ram_mib']<10240 or h['process_rss_mib']>3584 or h['gpu_free_mib']<8192 or h['gpu_temperature_c']>=68 or h['cpu_temperature_c']>=75
                if critical:
                    emergency=True
                    write_json(BASE/'events'/f'emergency_{time.time_ns()}.json',dict(h,action='Terminate only own worker and preserve durable completed records'))
                    child.terminate();break
                if time.monotonic()-last_sync>=300 and (sync_job is None or sync_job.poll() is not None):
                    sync_job=subprocess.Popen(['sync','-f',str(BASE)]);last_sync=time.monotonic()
                time.sleep(5)
            code=child.wait()
        if code==75 or emergency or code==-signal.SIGKILL:
            events=sorted((BASE/'events').glob('*.json'))
            latest=common.read(events[-1]) if events else {}
            limit_failure=emergency or code==-signal.SIGKILL or latest.get('resource_failure',False)
            attempts+=int(limit_failure)
            state.update(state='resource_pause',active=None,resource_retries=attempts);common.status(EXPERIMENT/'monitor/gpu.json',state)
            if attempts>=3:
                state.update(state='attention_required',resource_reason='Repeated resource limit exits; do not force a higher limit automatically');common.status(EXPERIMENT/'monitor/gpu.json',state);return
            time.sleep(30);continue
        if code:
            state['failures'].append({'model':'gswm','returncode':code,'log':str(log)});state.update(state='failed',active=None);common.status(EXPERIMENT/'monitor/gpu.json',state);return
        break
    common.verify(full=True)
    from ..report import main as report
    final=report(final=True)
    state.update(state='complete',active=None,completed=['ddlp','gswm'],finished=time.time());common.status(EXPERIMENT/'monitor/gpu.json',state)
    common.status(EXPERIMENT/'monitor/FINAL.json',{'success':final['all_complete'],'at':time.time(),'operational_guard':str(BASE),
        'latest_report':common.read(EXPERIMENT/'monitor/latest_report.json')})
    subprocess.run(['sync','-f',str(BASE)],check=True)
    write_json(BASE/'PERSISTED.json',{'at':time.time(),'group_sync_success':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');a=p.parse_args()
    if a.worker:worker()
    else:supervise()
