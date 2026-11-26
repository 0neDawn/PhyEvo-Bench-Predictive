"""Frozen original M orchestration, with only existing hook/alpha binding changed."""
import argparse
import importlib
import inspect
import os
import sys
import time
import traceback
from . import common as c
sys.path.insert(0,str(c.ROOT/'predictive_tracks/sp/code'))
import numpy as np
import torch
from .hooks import factory
from predictive_tracks.ppr_state_insert_20261104_m import worker as original
from predictive_tracks.ppr_20261028.run import rows_and_references
from predictive_tracks.controlled_tracks.registry import MODELS

def transformed():
    source=inspect.getsource(original.run_setting)
    assert source.count('radius_multiplier=1')==2
    changed=source.replace('radius_multiplier=1',"radius_multiplier=_nr_job['alpha']")
    ns=dict(vars(original),c=c,StateOnly=factory,PreFinal=factory,_nr_job=c.CURRENT)
    exec(compile(changed,'<eta_radius_n/original_run_setting>','exec'),ns)
    return ns['run_setting'],source,changed

def execute(model,stage2=False):
    c.verify();c.require_resources(model)
    torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
    torch.cuda.set_per_process_memory_fraction(.2 if model in c.SLOTS else .45,device=0)
    grid=c.read(c.HERE/('STAGE2.json' if stage2 else 'GRID.json'))
    jobs=[s for s in grid['neural_new'] if s['model']==model]
    done=c.HERE/('stage2_complete' if stage2 else 'models_complete')/f'{model}.json'
    if done.exists():return
    if not jobs:
        c.save(done,dict(at=time.time(),model=model,settings=0));return
    run,source,changed=transformed()
    assert (c.HERE/'native_transformed_source.txt').read_text()==changed
    data=sorted(rows_and_references(model),key=lambda r:(r[0]['benchmark']!='ie',r[0]['archive_index']))
    # Cooling is outside the forecast/score timer, identically for every case.
    if model=='videomaev2':c.require_resources=lambda m:c.video_rest()
    adapter=importlib.import_module(MODELS[model]).Adapter(device='cuda:0',memory_mode='safe',orientation='identity')
    baseline=original.versions(adapter,model)
    for mode in sorted({s['mode'] for s in jobs}):
        c.CURRENT.clear();c.CURRENT.update(c.setting(model,mode,0.,1.))
        run(adapter,model,0.,original.preselected(data),baseline)
    for s in jobs:
        c.CURRENT.clear();c.CURRENT.update(s)
        run(adapter,model,s['eta'],data,baseline)
    c.verify()
    c.save(done,dict(at=time.time(),model=model,settings=len(jobs),full_pairs_each=600))
    print('MODEL COMPLETE',model,len(jobs),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=c.MODELS,required=True);p.add_argument('--stage2',action='store_true');a=p.parse_args()
    try:execute(a.model,a.stage2)
    except c.ResourceBusy as exc:
        c.save(c.HERE/'events'/f'pause_{time.time_ns()}.json',dict(model=a.model,reason=str(exc),at=time.time()));sys.exit(75)
    except BaseException as exc:
        c.save(c.HERE/'events'/f'error_{time.time_ns()}.json',dict(model=a.model,error=repr(exc),traceback=traceback.format_exc(),at=time.time()));raise

if __name__=='__main__':main()
