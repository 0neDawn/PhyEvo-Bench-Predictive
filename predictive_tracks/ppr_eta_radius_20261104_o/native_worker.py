"""Original adapters and original per-case graphs; projection scalars only."""
import argparse,hashlib,importlib,json,os,sys,time,traceback
from pathlib import Path
from . import common as c
sys.path.insert(0,str(c.ROOT/'predictive_tracks/sp/code'))
import numpy as np
import torch
from .hooks import factory
from predictive_tracks.ppr_20261028.run import rows_and_references
from predictive_tracks.controlled_tracks.registry import MODELS,OUTPUT
from predictive_tracks.controlled_tracks.response_full_run import describe
from predictive_tracks.ppr_existing_20261102.metrics import make_summary
from sp_core.common import PairRecord,read_pair_rgb

def modules(a,m):
    if m in ('ocvp_seq','savi_transformer'):return (a.savi,a.predictor)
    if m=='slotformer':return (a.savi,a.dynamics)
    if m=='gswm':return (a.runtime.model,)
    if m in ('ddlp','videomaev2'):return (a.model,)
    return (a.encoder,a.predictor,a.target_encoder)
def versions(a,m):return [(id(p),p._version,tuple(p.shape),str(p.dtype),p.requires_grad) for mod in modules(a,m) for p in mod.parameters()]
def audit_key(j):return j['mode']+'_'+hashlib.sha256(json.dumps(j['quality'],sort_keys=True).encode()).hexdigest()[:10]
def preselected(data):
    selected=[];seen=set()
    for x in data:
        k=x[0]['benchmark'],x[0]['category_directory']
        if k not in seen:seen.add(k);selected.append(x)
    assert len(selected)==6;return selected
def run(a,j,data,base,shared,clip_len,audit=False):
    m=j['model'];out=c.HERE/'audit'/m/audit_key(j) if audit else c.directory(j)
    if (out/'summary.json').exists():return
    c.verify();c.before_native(m)
    cfg=dict(job=j,audit=audit,cases=len(data),provenance=a.provenance(),extra_forwards_per_forecast=0,
        plan_sha256=c.digest(c.HERE/'PLAN.txt'),source_manifest_sha256=c.digest(c.HERE/'frozen.json'),torch=torch.__version__,cuda=torch.version.cuda)
    if (out/'config.json').exists():assert c.read(out/'config.json')==cfg
    else:c.save(out/'config.json',cfg)
    results=[]
    for row,rp,ref in data:
        dest=out/'pairs'/f'{row["key"]}.json'
        if dest.exists():
            x=c.read(dest);assert x['job']==j and x['reference_pair_sha256']==c.digest(rp);results.append(x);continue
        c.before_native(m)
        pair=PairRecord(row['pair_id'],row['category_directory'],int(row['t_ref']),Path(row['possible_dir']),Path(row['impossible_dir']),Path(row['meta_path']))
        p,i=read_pair_rgb(pair);pixels=hashlib.sha256(p.tobytes()+i.tobytes()).hexdigest()
        assert pixels==ref['source_decoded_pixels_sha256']
        expected=ref.get('neural_call_trace',shared)
        if expected is None and row['split']=='development':expected=c.read(OUTPUT/f'{m}_boundary_dev_001/pairs'/f'{row["key"]}.json')['neural_call_trace']
        if expected is None:raise RuntimeError('Original neural trace missing')
        if m=='slotformer':
            n=expected['module_forward_invocations']
            if n not in (1321,2103):raise RuntimeError('Unknown original tokenizer schedule')
            a.savi.clip_len=clip_len if n==2103 else 14
        a.prediction_calls=row['archive_index'];calls=[0];trace=hashlib.sha256()
        def count(mod,inputs):
            calls[0]+=1;trace.update(json.dumps((mod.__class__.__module__,mod.__class__.__qualname__,describe(inputs)),sort_keys=True).encode())
        counter=torch.nn.modules.module.register_module_forward_pre_hook(count);start=time.perf_counter()
        try:
            h=factory(a,j,out/'instrumented_sources')
            with h:result=a.score_pair(p,i)
        finally:counter.remove()
        torch.cuda.synchronize()
        observed=dict(module_forward_invocations=calls[0],ordered_class_and_input_shapes_sha256=trace.hexdigest())
        if observed!=expected:raise RuntimeError(f'Original graph/count/order/input-shapes changed: {observed} vs {expected}')
        if a.prediction_calls!=row['archive_index']+1 or result.frame_spans.tolist()!=ref['frame_spans']:raise RuntimeError('Original budget/time grid changed')
        if versions(a,m)!=base:raise RuntimeError('Parameter identity/version changed')
        if pixels!=hashlib.sha256(p.tobytes()+i.tobytes()).hexdigest():raise RuntimeError('Source pixels mutated')
        if not all(np.isfinite(v).all() for v in (result.possible,result.impossible)):raise RuntimeError('Nonfinite score')
        expected_audits=8 if m=='gswm' else 1;steps=3 if j['mode']=='pre_final_block' else 6
        if len(h.audits)!=expected_audits or any(len(v['steps'])!=steps for v in h.audits):raise RuntimeError('Original MC/forecast-step budget changed')
        if j['mode']=='pre_final_block' and (h.activations!=1 or h.inactive_calls!=(6 if m=='videomaev2' else 0)):raise RuntimeError('Prediction hook leaked into targets')
        extra={}
        if audit:
            diff=max(float(np.max(np.abs(x-np.asarray(ref['matched_identity_'+b+'_errors'])))) for x,b in ((result.possible,'possible'),(result.impossible,'impossible')))
            if diff!=0:raise RuntimeError('Eta0 must reproduce original errors exactly')
            extra['no_op_max_abs_error']=diff
        x=dict(benchmark=row['benchmark'],archive_index=row['archive_index'],pair_id=row['pair_id'],category=row['category_directory'],split=row['split'],t_ref=int(row['t_ref']),
            frame_spans=result.frame_spans.tolist(),possible_errors=result.possible.tolist(),impossible_errors=result.impossible.tolist(),
            matched_identity_possible_errors=ref['matched_identity_possible_errors'],matched_identity_impossible_errors=ref['matched_identity_impossible_errors'],
            job=j,model=m,mode=j['mode'],eta=j['eta'],radius_multiplier=j['alpha'],reference_pair_sha256=c.digest(rp),source_decoded_pixels_sha256=pixels,
            neural_call_trace=observed,trace_matches_original=True,extra_forwards=0,weight_identity_and_version_unchanged=True,
            wall_seconds=time.perf_counter()-start,projection_audits=h.audits,instrumented_function_hashes=h.functions,**extra)
        c.save(dest,x);results.append(x)
        c.status(out/'progress.json',dict(at=time.time(),completed=len(results),total=len(data),last=row['key']))
        if len(results)%20==0 or audit:print('NATIVE',m,j['mode'],j['eta'],j['alpha'],len(results),'/',len(data),flush=True)
    s=dict(complete=True,cases=6,audit_passed=True,no_op_max_abs_error=max(x['no_op_max_abs_error'] for x in results)) if audit else make_summary(results,m)
    s.update(job=j,extra_forwards=0,all_native_traces_matched=True);c.save(out/'summary.json',s)
    print('SETTING COMPLETE',j['config_id'],'audit' if audit else 'full600',flush=True)
def execute(model,stage2=False):
    c.verify();c.require_resources(model);torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
    torch.cuda.set_per_process_memory_fraction(.45 if model in c.LARGE else .2,device=0)
    jobs=[j for j in c.read(c.HERE/('STAGE2.json' if stage2 else 'GRID.json'))['native_new'] if j['model']==model]
    done=c.HERE/('stage2_complete' if stage2 else 'models_complete')/f'native_{model}.json'
    if done.exists():return
    if not jobs:c.save(done,dict(at=time.time(),model=model,settings=0));return
    data=sorted(rows_and_references(model),key=lambda x:(x[0]['benchmark']!='ie',x[0]['archive_index']))
    adapter=importlib.import_module(MODELS[model]).Adapter(device='cuda:0',memory_mode='safe',orientation='identity')
    base=versions(adapter,model);clip=adapter.savi.clip_len if model=='slotformer' else None
    ts={json.dumps(r['neural_trace'],sort_keys=True) for r in c.read(OUTPUT/'deployment_audit_001.json')['details'][model]}
    shared=json.loads(next(iter(ts))) if len(ts)==1 else None
    seen=set()
    for j in jobs:
        k=audit_key(j)
        if k in seen:continue
        seen.add(k);audit=dict(j,eta=0.,alpha=1.);run(adapter,audit,preselected(data),base,shared,clip,True)
    for j in jobs:run(adapter,j,data,base,shared,clip)
    c.verify();c.save(done,dict(at=time.time(),model=model,settings=len(jobs),cases_each=600))
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=c.NATIVE,required=True);p.add_argument('--stage2',action='store_true');a=p.parse_args()
    try:execute(a.model,a.stage2)
    except c.ResourceBusy as e:c.save(c.HERE/'events'/f'pause_{time.time_ns()}.json',dict(at=time.time(),model=a.model,reason=str(e)));sys.exit(75)
    except BaseException as e:c.save(c.HERE/'events'/f'error_{time.time_ns()}.json',dict(at=time.time(),model=a.model,error=repr(e),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
