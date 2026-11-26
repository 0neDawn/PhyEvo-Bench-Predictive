"""Execute original models once per forecast; audit and store new tracks only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import traceback
from . import common as c
sys.path.insert(0,str(c.ROOT/'predictive_tracks/sp/code'))
import numpy as np
import torch
from .hooks import StateOnly,PreFinal
from predictive_tracks.ppr_20261028.run import rows_and_references
from predictive_tracks.controlled_tracks.registry import MODELS,OUTPUT
from predictive_tracks.controlled_tracks.response_full_run import describe
from predictive_tracks.ppr_existing_20261102.metrics import make_summary
from sp_core.common import PairRecord,read_pair_rgb

def modules(adapter,model):
    if model in c.SLOTS:return (adapter.savi,adapter.predictor)
    if model=='vjepa2_vitl':return (adapter.encoder,adapter.predictor,adapter.target_encoder)
    return (adapter.model,)

def versions(adapter,model):
    return [(id(p),p._version,tuple(p.shape),str(p.dtype),p.requires_grad)
            for m in modules(adapter,model) for p in m.parameters()]

def preselected(data):
    selected=[];seen=set()
    for x in data:
        group=x[0]['benchmark'],x[0]['category_directory']
        if group not in seen:seen.add(group);selected.append(x)
    assert len(selected)==6
    return selected

def run_setting(adapter,model,eta,data,baseline_versions):
    out=c.directory(model,eta);mode=c.mode(model)
    if (out/'summary.json').exists():return
    c.verify();c.require_resources(model)
    cfg=dict(model=model,mode=mode,eta=eta,cases=len(data),radius_multiplier=1,distance='Euclidean',
        all_original_history=True,additional_neural_forwards_per_forecast=0,
        plan_sha256=c.digest(c.HERE/'PLAN.txt'),source_manifest_sha256=c.digest(c.HERE/'frozen.json'),
        provenance=adapter.provenance(),torch=torch.__version__,cuda=torch.version.cuda,
        scope='Previously exposed full SP300/IE300 exploratory' if eta else 'Six fixed no-op implementation audits')
    if (out/'config.json').exists():
        if c.read(out/'config.json')!=cfg:raise RuntimeError('Resume configuration differs')
    else:c.save(out/'config.json',cfg)
    refs=c.read(OUTPUT/'deployment_audit_001.json')['details'][model]
    traces={json.dumps(r['neural_trace'],sort_keys=True) for r in refs}
    common=json.loads(next(iter(traces))) if len(traces)==1 else None
    records=[]
    for row,ref_path,ref in data:
        dest=out/'pairs'/f'{row["key"]}.json'
        if dest.exists():
            result=c.read(dest)
            if result['reference_pair_sha256']!=c.digest(ref_path) or result['eta']!=eta:
                raise RuntimeError('Resume reference differs')
            records.append(result);continue
        c.require_resources(model)
        record=PairRecord(row['pair_id'],row['category_directory'],int(row['t_ref']),
            Path(row['possible_dir']),Path(row['impossible_dir']),Path(row['meta_path']))
        possible,impossible=read_pair_rgb(record)
        pixels=hashlib.sha256(possible.tobytes()+impossible.tobytes()).hexdigest()
        if pixels!=ref['source_decoded_pixels_sha256']:raise RuntimeError('Original pixels differ')
        adapter.prediction_calls=row['archive_index']
        expected=ref.get('neural_call_trace',common)
        if expected is None and row['split']=='development':
            expected=c.read(OUTPUT/f'{model}_boundary_dev_001/pairs'/f'{row["key"]}.json')['neural_call_trace']
        if expected is None:raise RuntimeError('Original call trace missing')
        calls=[0];trace=hashlib.sha256()
        def count(module,inputs):
            calls[0]+=1
            trace.update(json.dumps((module.__class__.__module__,module.__class__.__qualname__,describe(inputs)),sort_keys=True).encode())
        counter=torch.nn.modules.module.register_module_forward_pre_hook(count)
        started=time.perf_counter()
        try:
            hook=(StateOnly if model in c.SLOTS else PreFinal)(adapter,model,eta,out/'instrumented_sources')
            with hook:result=adapter.score_pair(possible,impossible)
        finally:counter.remove()
        torch.cuda.synchronize()
        observed=dict(module_forward_invocations=calls[0],ordered_class_and_input_shapes_sha256=trace.hexdigest())
        if observed!=expected:raise RuntimeError(f'Native call/order/shape mismatch: {observed} vs {expected}')
        if adapter.prediction_calls!=row['archive_index']+1:raise RuntimeError('Forecast budget mismatch')
        if result.frame_spans.tolist()!=ref['frame_spans']:raise RuntimeError('Original frame spans differ')
        if versions(adapter,model)!=baseline_versions:raise RuntimeError('Model parameter identity/version changed')
        if pixels!=hashlib.sha256(possible.tobytes()+impossible.tobytes()).hexdigest():raise RuntimeError('Source pixels mutated')
        if not all(np.isfinite(x).all() for x in (result.possible,result.impossible)):raise RuntimeError('Nonfinite score')
        if len(hook.audits)!=1 or len(hook.audits[0]['steps'])!=(6 if model in c.SLOTS else 3):
            raise RuntimeError('Wrong original prediction-step budget or correction activation')
        if model not in c.SLOTS and (hook.activations!=1 or hook.inactive_calls!=(6 if model=='videomaev2' else 0)):
            raise RuntimeError('Hook was not restricted to original prediction, or targets changed')
        extra={}
        if model in c.SLOTS:
            extra['first_raw_prediction_sha256']=hook.first_prediction_hash
            anchor=c.HERE/'anchors'/model/f'{row["key"]}.json'
            if anchor.exists():
                if c.read(anchor)['first_raw_prediction_sha256']!=hook.first_prediction_hash:
                    raise RuntimeError('State-only unexpectedly changed first prediction')
            else:c.save(anchor,dict(first_raw_prediction_sha256=hook.first_prediction_hash,eta=eta))
        else:extra.update(prediction_output_sha256=hook.output_hash,active_hook_calls=hook.activations,inactive_target_hook_calls=hook.inactive_calls)
        if eta==0:
            diffs=[]
            for branch,actual in (('possible',result.possible),('impossible',result.impossible)):
                wanted=np.asarray(ref[f'matched_identity_{branch}_errors'])
                diffs.append(float(np.max(np.abs(actual-wanted))))
                if not np.allclose(actual,wanted,atol=1e-7,rtol=1e-5):raise RuntimeError('Eta0 failed original score reproduction')
            extra['no_op_max_abs_error']=max(diffs)
        payload=dict(benchmark=row['benchmark'],archive_index=row['archive_index'],pair_id=row['pair_id'],
            category=row['category_directory'],split=row['split'],t_ref=int(row['t_ref']),
            frame_spans=result.frame_spans.tolist(),possible_errors=result.possible.tolist(),impossible_errors=result.impossible.tolist(),
            matched_identity_possible_errors=ref['matched_identity_possible_errors'],
            matched_identity_impossible_errors=ref['matched_identity_impossible_errors'],
            source_decoded_pixels_sha256=pixels,reference_pair_sha256=c.digest(ref_path),
            eta=eta,mode=mode,radius_multiplier=1,neural_call_trace=observed,trace_matches_original=True,
            wall_seconds=time.perf_counter()-started,projection_audits=hook.audits,instrumented_function_hashes=hook.functions,
            extra_forwards=0,weight_identity_and_version_unchanged=True,**extra)
        c.save(dest,payload);records.append(payload)
        c.status(out/'progress.json',dict(at=time.time(),completed=len(records),total=len(data),last=row['key']))
        if len(records)%10==0 or eta==0:print(model,mode,eta,len(records),'/',len(data),row['key'],flush=True)
    c.verify()
    summary=(make_summary(records,model) if eta else dict(model=model,cases=6,complete=True,audit_passed=True,
        no_op_max_abs_error=max(r['no_op_max_abs_error'] for r in records)))
    summary.update(mode=mode,eta=eta,all_native_traces_matched=True,extra_forwards=0)
    c.save(out/'summary.json',summary)
    print('COMPLETE',model,mode,eta,flush=True)

def main(model):
    try:
        c.verify();c.require_resources(model)
        torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
        torch.cuda.set_per_process_memory_fraction(.2 if model in c.SLOTS else .45,device=0)
        data=rows_and_references(model)
        # IE first prioritizes the requested question; all600 remain mandatory.
        data=sorted(data,key=lambda r:(r[0]['benchmark']!='ie',r[0]['archive_index']))
        adapter=importlib.import_module(MODELS[model]).Adapter(device='cuda:0',memory_mode='safe',orientation='identity')
        baseline=versions(adapter,model)
        run_setting(adapter,model,0.,preselected(data),baseline)
        for eta in c.ETAS:run_setting(adapter,model,eta,data,baseline)
        c.verify()
        c.save(c.HERE/'models_complete'/f'{model}.json',dict(at=time.time(),model=model,full_settings=3,full_pairs_each=600))
    except c.ResourceBusy as exc:
        c.save(c.HERE/'events'/f'pause_{time.time_ns()}.json',dict(model=model,reason=str(exc),at=time.time()))
        sys.exit(75)
    except BaseException as exc:
        c.save(c.HERE/'events'/f'error_{time.time_ns()}.json',dict(model=model,error=repr(exc),traceback=traceback.format_exc(),at=time.time()))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=c.MODELS,required=True);a=p.parse_args();main(a.model)
