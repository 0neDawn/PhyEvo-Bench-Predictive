"""Detached serial queue; each worker resumes missing cases without extra steps."""
import fcntl
import os
import subprocess
import time
from .common import HERE,ROOT,OBJECTS,configs,directory,old_folder,read,status,verify,gate,PYTHONS

def main():
    monitor=HERE/'monitor';monitor.mkdir(exist_ok=True)
    lock=(monitor/'gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify(full=True)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    state={'pid':os.getpid(),'started':time.time(),'state':'starting','completed':[],'failures':[],'active':None,'queue':list(OBJECTS),'alpha_fixed':1}
    status(monitor/'gpu.json',state)
    with (monitor/'reports.log').open('a') as f:
        subprocess.run([PYTHONS['other'],'-B','-m','predictive_tracks.ppr_existing_20261102.report'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    for model in OBJECTS:
        if all(old_folder(model,c) is not None or (directory(model,c)/'summary.json').exists() for c in configs(model)):
            state['completed'].append(model);continue
        while True:
            stable=0
            while stable<2:
                okay,detail=gate();stable=stable+1 if okay else 0
                state.update(state='checking_resources' if okay else 'waiting_resources',active={'model':model},last_resource_check=detail)
                status(monitor/'gpu.json',state)
                if stable<2:time.sleep(5 if okay else 20)
            log=monitor/'logs'/f'{model}_{time.time_ns()}.log';log.parent.mkdir(exist_ok=True)
            command=[PYTHONS.get(model,PYTHONS['other']),'-B','-u','-m','predictive_tracks.ppr_existing_20261102.run_gpu','--model',model]
            with log.open('x') as f:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
                state.update(state='running',active={'model':model,'pid':child.pid,'log':str(log)});status(monitor/'gpu.json',state);code=child.wait()
            if code==75:
                state.update(state='resource_pause',active=None);status(monitor/'gpu.json',state);time.sleep(10);continue
            if code:state['failures'].append({'model':model,'returncode':code,'log':str(log)})
            else:state['completed'].append(model)
            state.update(active=None,state='between_models');status(monitor/'gpu.json',state);break
        with (monitor/'reports.log').open('a') as f:
            subprocess.run([PYTHONS['other'],'-B','-m','predictive_tracks.ppr_existing_20261102.report'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    state.update(state='failed' if state['failures'] else 'complete',active=None,finished=time.time());status(monitor/'gpu.json',state)
    if not state['failures']:
        verify(full=True)
        with (monitor/'final_report.log').open('a') as f:
            code=subprocess.call([PYTHONS['other'],'-B','-m','predictive_tracks.ppr_existing_20261102.report','--final'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        status(monitor/'FINAL.json',{'success':code==0,'at':time.time(),'log':str(monitor/'final_report.log')})

if __name__=='__main__':main()
