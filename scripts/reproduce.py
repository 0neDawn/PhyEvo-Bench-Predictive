"""One-command reproduction: archived-score verification or full native runs."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
MODELS=('ocvp_seq','savi_transformer','slotformer','ddlp','gswm','videomaev2','vjepa2_vitl','vjepa2_vitg','dino_foresight')


def save(path, value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def resources_ready(model, device):
    memory={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        k,v=line.split(':',1);memory[k]=int(v.split()[0])
    needed_gb=22 if model=='vjepa2_vitg' else 12
    if memory['MemAvailable'] < needed_gb*1024*1024:
        return False,f'Available RAM below {needed_gb} GiB'
    index=device.split(':')[-1]
    proc=subprocess.run(['nvidia-smi','-i',index,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
    if proc.returncode:
        return False,'Cannot read GPU status: '+proc.stderr.strip()
    used,util=[int(x.strip()) for x in proc.stdout.strip().split(',')]
    if used>1024 or util>10:
        return False,f'GPU busy: {used} MiB, {util}% utilization'
    return True,'Idle GPU and sufficient RAM'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('full','verify'),default='full')
    parser.add_argument('--run-dir',type=Path,default=ROOT/'runs/full_reproduction')
    parser.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--memory-fraction',type=float,default=.45)
    parser.add_argument('--assets',type=Path,help='Optional split-asset folder to restore before running')
    parser.add_argument('--wait-for-resources',action='store_true')
    args=parser.parse_args()
    # Keep repeated invocations from writing the same run directory concurrently.
    # The operating system releases the lock if the process exits or crashes.
    run_lock=None
    if args.mode=='full':
        args.run_dir.mkdir(parents=True,exist_ok=True)
        run_lock=(args.run_dir/'.reproduction.lock').open('a')
        try:
            fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('This run directory is already in use; inspect its status.json instead.')
    if args.assets:
        subprocess.run([sys.executable,'-B',str(ROOT/'scripts/restore_assets.py'),str(args.assets)],check=True)
    subprocess.run([sys.executable,'-B',str(ROOT/'scripts/recompute_results.py')],check=True)
    if args.mode=='verify':
        native=ROOT/'audit/native_reproduction'
        if native.exists():
            subprocess.run([sys.executable,'-B',str(ROOT/'scripts/verify_native.py'),
                            '--run-dirs',str(native),'--require-complete',
                            '--output',str(ROOT/'recomputed/native_verification.json')],check=True)
        return
    if any(m in args.models for m in ('ocvp_seq','savi_transformer')):
        subprocess.run([sys.executable,'-B',str(ROOT/'scripts/fetch_ocvp.py')],check=True)
    args.run_dir.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,'-B',str(ROOT/'scripts/verify_inputs.py'),
                    '--output',str(args.run_dir/'input_verification.json')],check=True)
    logdir=args.run_dir/'logs';logdir.mkdir(exist_ok=True)
    jobs=[dict(model=m,task=t,variant=v) for m in args.models for t in ('SP','IE') for v in ('baseline','hsr')]
    status=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),mode='full_native_from_original_frames',jobs=[],status='running')
    save(args.run_dir/'status.json',status)
    for job in jobs:
        key='_'.join(job.values())
        folder=args.run_dir/key
        done=folder/'COMPLETE.json'
        if done.exists():
            expected=next(c for c in json.loads((ROOT/'configs/final_configs.json').read_text())['configurations']
                          if c['model_id']==job['model'] and c['task']==job['task'])
            if json.loads((folder/'config.json').read_text())!=expected:
                raise RuntimeError('Completed run uses different settings; choose a new run directory')
            status['jobs'].append(dict(job,status='complete',directory=str(folder)))
            save(args.run_dir/'status.json',status)
            continue
        while True:
            ready,reason=resources_ready(job['model'],args.device)
            if ready:break
            status.update(status='waiting_for_resources',current=job,reason=reason)
            save(args.run_dir/'status.json',status)
            print(reason,flush=True)
            if not args.wait_for_resources:
                return 75
            time.sleep(30)
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        log=logdir/f'{key}_{stamp}.log'
        status.update(status='running',current=job,log=str(log))
        save(args.run_dir/'status.json',status)
        command=[sys.executable,'-B',str(ROOT/'scripts/run_evaluation.py'),
                 '--model',job['model'],'--task',job['task'],'--variant',job['variant'],
                 '--output',str(folder),'--device',args.device,'--memory-fraction',str(args.memory_fraction),'--resume']
        print('START',key,flush=True)
        with log.open('w') as out:
            proc=subprocess.run(command,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,
                                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
        ok=proc.returncode==0 and done.exists()
        status['jobs'].append(dict(job,status='complete' if ok else 'failed',returncode=proc.returncode,log=str(log),directory=str(folder)))
        save(args.run_dir/'status.json',status)
        print('DONE' if ok else 'FAILED',key,flush=True)
    status.update(status='complete' if all(j['status']=='complete' for j in status['jobs']) else 'completed_with_failures',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    status.pop('current',None)
    save(args.run_dir/'status.json',status)
    subprocess.run([sys.executable,'-B',str(ROOT/'scripts/summarize_native.py'),'--run-dir',str(args.run_dir)],check=True)
    return 0 if status['status']=='complete' else 1


if __name__=='__main__':
    raise SystemExit(main())
