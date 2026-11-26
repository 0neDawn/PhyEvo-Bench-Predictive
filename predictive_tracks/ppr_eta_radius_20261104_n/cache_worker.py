"""Complete original-precision CUDA tensor replay; zero neural module forwards."""
import argparse
import os
import sys
import time
import traceback
from . import common as c
sys.path.insert(0,str(c.ROOT/'predictive_tracks/sp/code'))
import numpy as np
import torch
from predictive_tracks.ppr_all_20261029.common import rows
from predictive_tracks.ppr_alpha_20261031.projection import pooled_trajectory
from predictive_tracks.ppr_eta_20261030.cache import meta_path
from predictive_tracks.state_tracks_20261023.all_models_20261023.run import forbid,load_data,evaluator_for,validate_errors,variant_record
from predictive_tracks.ppr_existing_20261102.metrics import make_summary

def folder(stage2=False):return c.HERE/('cache_stage2' if stage2 else 'cache')/'videomaev2'

def expand(payload,job):
    x=dict(payload['native']);x.update(payload['settings'][job['config_id']])
    x.update(eta=job['eta'],radius_multiplier=job['alpha'],mode='output_only',neural_forwards=0,
             score_device='original_cuda',native_meta=payload['native_meta'],source_meta_sha256=payload['source_meta_sha256'],
             source_state_sha256=payload['source_state_sha256'],audit=payload['projection_audits'][str(job['alpha'])])
    return x

def execute(stage2=False):
    c.verify();c.require_resources('videomaev2')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2,device=0)
    jobs=c.read(c.HERE/('STAGE2.json' if stage2 else 'GRID.json'))['cached_video']
    root=folder(stage2);done=c.HERE/('stage2_complete' if stage2 else 'models_complete')/'cache_videomaev2.json'
    if done.exists():return
    if not jobs:c.save(done,dict(at=time.time(),settings=0,neural_forwards=0));return
    guard=torch.nn.modules.module.register_module_forward_pre_hook(forbid)
    try:
        with torch.inference_mode():
            for index,row in enumerate(rows(),1):
                dest=root/'pairs'/f'{row["key"]}.json'
                if dest.exists():
                    old=c.read(dest);assert set(old['settings'])=={s['config_id'] for s in jobs};continue
                c.require_resources('videomaev2')
                mp=meta_path('videomaev2',row);meta=c.read(mp);data=load_data(meta);p=data['prediction']
                evaluator=evaluator_for(data,'videomaev2');evaluator.adapter.device=torch.device('cuda:0')
                evaluator.targets=[v.to(evaluator.adapter.device) for v in evaluator.targets]
                native=evaluator.errors(p)
                wanted=[meta['matched_identity_'+b+'_errors'] for b in ('possible','impossible')]
                validate_errors(native,wanted,meta,'videomaev2')
                difference=max(float(np.max(np.abs(np.asarray(x)-np.asarray(y)))) for x,y in zip(native,wanted))
                if difference!=0:raise RuntimeError('Original CUDA native errors must reproduce exactly')
                settings={};audits={}
                for alpha in sorted({s['alpha'] for s in jobs}):
                    hard,audit=pooled_trajectory(p,data['history'],alpha)
                    audits[str(alpha)]=audit
                    for s in (x for x in jobs if x['alpha']==alpha):
                        eta=s['eta'];pred=p if eta==0 else hard if eta==1 else p+eta*(hard-p)
                        errors=evaluator.errors(pred)
                        settings[s['config_id']]=dict(possible_errors=np.asarray(errors[0]).tolist(),impossible_errors=np.asarray(errors[1]).tolist())
                    del hard,pred
                c.save(dest,dict(native=variant_record(meta,native),settings=settings,projection_audits=audits,
                    native_meta=str(mp),source_meta_sha256=c.digest(mp),source_state_sha256=meta['sha256'],
                    native_max_abs_error=difference,neural_forwards=0,score_device='original_cuda',
                    source_expression='eta0:raw;eta1:hard;interior:p+eta*(hard-p)'))
                c.status(root/'progress.json',dict(at=time.time(),completed=index,total=600,settings=len(jobs)))
                if index%20==0:print('CACHE',index,'/600 for',len(jobs),'configurations;NN0',flush=True)
                del evaluator,data,p,native,settings
    finally:guard.remove()
    payloads=[c.read(p) for p in sorted((root/'pairs').glob('*.json'))];assert len(payloads)==600
    native=make_summary([p['native'] for p in payloads],'videomaev2')
    for t in ('sp','ie'):assert native['metrics'][t]['corrected']==native['metrics'][t]['native_baseline']
    if not (root/'native_summary.json').exists():c.save(root/'native_summary.json',native)
    for s in jobs:
        out=c.path(s)
        if (out/'summary.json').exists():continue
        results=[expand(p,s) for p in payloads]
        summary=make_summary(results,'videomaev2');assert summary['complete'] and summary['cases']==600
        summary.update(mode=s['mode'],eta=s['eta'],radius_multiplier=s['alpha'],neural_forwards=0,
            compact_case_source=str(root/'pairs'),score_device='original_cuda')
        c.save(out/'summary.json',summary);c.save(out/'config.json',dict(s,compact_case_source=str(root/'pairs'),
            frozen_sha256=c.digest(c.HERE/'frozen.json'),source_expression='eta0:raw;eta1:hard;interior:p+eta*(hard-p)'))
    c.verify();c.save(done,dict(at=time.time(),settings=len(jobs),cases_each=600,neural_forwards=0))
    print('CACHE COMPLETE',len(jobs),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage2',action='store_true');a=p.parse_args()
    try:execute(a.stage2)
    except c.ResourceBusy as exc:c.save(c.HERE/'events'/f'pause_{time.time_ns()}.json',dict(at=time.time(),reason=str(exc)));sys.exit(75)
    except BaseException as exc:c.save(c.HERE/'events'/f'error_{time.time_ns()}.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
