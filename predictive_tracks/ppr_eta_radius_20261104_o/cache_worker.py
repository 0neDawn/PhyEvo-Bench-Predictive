"""Only saved native states and unmodified original-CUDA scoring, NN0."""
import argparse,sys,time,traceback
from . import common as c
sys.path.insert(0,str(c.ROOT/'predictive_tracks/sp/code'))
import numpy as np
import torch
from predictive_tracks.ppr_all_20261029.common import rows
from predictive_tracks.ppr_alpha_20261031.projection import trajectory,pooled_trajectory
from predictive_tracks.ppr_eta_20261030.cache import meta_path
from predictive_tracks.state_tracks_20261023.all_models_20261023.run import forbid,load_data,evaluator_for,validate_errors,variant_record
from predictive_tracks.ppr_existing_20261102.metrics import make_summary
def folder(model,stage2=False):return c.HERE/('cache_stage2' if stage2 else 'cache')/model
def expand(p,j):
    x=dict(p['native']);x.update(p['settings'][j['config_id']])
    x.update(eta=j['eta'],radius_multiplier=j['alpha'],mode=j['mode'],model=j['model'],neural_forwards=0,
        native_meta=p['native_meta'],source_meta_sha256=p['source_meta_sha256'],source_state_sha256=p['source_state_sha256'],
        score_device='original_cuda',audit=p['projection_audits'][str(j['alpha'])])
    return x
def execute(model,stage2=False):
    c.verify();c.require_resources(model);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2,device=0)
    grid=c.read(c.HERE/('STAGE2.json' if stage2 else 'GRID.json'));jobs=[j for j in grid['cached'] if j['model']==model]
    done=c.HERE/('stage2_complete' if stage2 else 'models_complete')/f'cache_{model}.json'
    if done.exists():return
    if not jobs:c.save(done,dict(at=time.time(),settings=0));return
    root=folder(model,stage2);fn=trajectory if model=='slotformer' else pooled_trajectory
    guard=torch.nn.modules.module.register_module_forward_pre_hook(forbid)
    try:
        with torch.inference_mode():
            for index,row in enumerate(rows(),1):
                dest=root/'pairs'/f'{row["key"]}.json'
                if dest.exists():
                    old=c.read(dest);assert set(old['settings'])=={j['config_id'] for j in jobs};continue
                c.require_resources(model);mp=meta_path(model,row)
                if mp is None:raise RuntimeError('Missing native cache; no new neural capture allowed')
                meta=c.read(mp);data=load_data(meta);p=data['prediction']
                ev=evaluator_for(data,model);ev.adapter.device=torch.device('cuda:0');ev.targets=[v.to(ev.adapter.device) for v in ev.targets]
                native=ev.errors(p);wanted=[meta['matched_identity_'+b+'_errors'] for b in ('possible','impossible')]
                validate_errors(native,wanted,meta,model)
                diff=max(float(np.max(np.abs(np.asarray(x)-y))) for x,y in zip(native,wanted))
                if diff!=0:raise RuntimeError('Original CUDA errors must be bitwise reproduced')
                settings={};audits={}
                for alpha in sorted({j['alpha'] for j in jobs}):
                    hard,audit=fn(p,data['history'],alpha);audits[str(alpha)]=audit
                    for j in (s for s in jobs if s['alpha']==alpha):
                        eta=j['eta'];pred=p if eta==0 else hard if eta==1 else p+eta*(hard-p);errors=ev.errors(pred)
                        settings[j['config_id']]=dict(possible_errors=np.asarray(errors[0]).tolist(),impossible_errors=np.asarray(errors[1]).tolist())
                    del hard,pred
                c.save(dest,dict(native=variant_record(meta,native),settings=settings,projection_audits=audits,
                    native_meta=str(mp),source_meta_sha256=c.digest(mp),source_state_sha256=meta['sha256'],native_max_abs_error=diff,
                    neural_forwards=0,source_expression='eta0:raw;eta1:hard;interior:p+eta*(hard-p)',diagnostic_only=model=='dino_foresight'))
                c.status(root/'progress.json',dict(at=time.time(),completed=index,total=600,settings=len(jobs)))
                if index%20==0:print('CACHE',model,index,'/600;',len(jobs),'settings;NN0',flush=True)
                del ev,data,p,native,settings
    finally:guard.remove()
    payloads=[c.read(p) for p in sorted((root/'pairs').glob('*.json'))];assert len(payloads)==600
    baseline=make_summary([p['native'] for p in payloads],model)
    for t in ('sp','ie'):assert baseline['metrics'][t]['corrected']==baseline['metrics'][t]['native_baseline']
    if not (root/'native_summary.json').exists():c.save(root/'native_summary.json',baseline)
    for j in jobs:
        out=c.directory(j)
        if (out/'summary.json').exists():continue
        s=make_summary([expand(p,j) for p in payloads],model)
        s.update(job=j,compact_case_source=str(root/'pairs'),native_errors_exact=True,neural_forwards=0,diagnostic_only=model=='dino_foresight')
        c.save(out/'summary.json',s);c.save(out/'config.json',dict(j,source_expression='eta0:raw;eta1:hard;interior:p+eta*(hard-p)',source_manifest_sha256=c.digest(c.HERE/'frozen.json')))
    c.verify();c.save(done,dict(at=time.time(),model=model,settings=len(jobs),cases_each=600,neural_forwards=0))
    print('CACHE MODEL COMPLETE',model,flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=c.CACHED,required=True);p.add_argument('--stage2',action='store_true');a=p.parse_args()
    try:execute(a.model,a.stage2)
    except c.ResourceBusy as e:c.save(c.HERE/'events'/f'pause_{time.time_ns()}.json',dict(at=time.time(),reason=str(e)));sys.exit(75)
    except BaseException as e:c.save(c.HERE/'events'/f'error_{time.time_ns()}.json',dict(at=time.time(),error=repr(e),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
