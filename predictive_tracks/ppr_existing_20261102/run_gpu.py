"""One native adapter per model. Each fresh trial preserves its original budget."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'predictive_tracks/sp/code'))
import hashlib
import importlib
import json
import time
import traceback
import numpy as np
import torch
from .common import HERE,OBJECTS,configs,config_id,old_folder,result_folder,directory,read,save,status,digest,verify,require_resources,ResourceBusy
from .hooks import ExistingDDLP,ExistingGSWM
from .metrics import make_summary
from .equivalence import FIELDS,refpath,validpath,register_history,find,compare,save_alias
from predictive_tracks.ppr_20261028.run import rows_and_references
from predictive_tracks.controlled_tracks.registry import MODELS,OUTPUT
from predictive_tracks.controlled_tracks.response_full_run import describe
from sp_core.common import PairRecord,read_pair_rgb

def execute(model):
    require_resources();verify();torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
    if not torch.cuda.is_available():raise RuntimeError('Original CUDA required')
    grid=configs(model);pending=[c for c in grid if old_folder(model,c) is None]
    data=rows_and_references(model)
    if len(data)!=600:raise RuntimeError('Original full pair set required')
    adapter=importlib.import_module(MODELS[model]).Adapter(device='cuda:0',memory_mode='safe',orientation='identity')
    for c in pending:
        out=directory(model,c);cfg={'model':model,'mode':'feedback','parameters':c,'alpha_fixed':1,'cases':600,
            'provenance':adapter.provenance(),'plan_sha256':digest(HERE/'PLAN.txt'),'source_manifest_sha256':digest(HERE/'frozen.json'),
            'torch':torch.__version__,'cuda':torch.version.cuda,'extra_forwards_per_forecast':0}
        if (out/'config.json').exists():
            if read(out/'config.json')!=cfg:raise RuntimeError('Resume configuration changed')
        else:save(out/'config.json',cfg)
    refs=read(OUTPUT/'deployment_audit_001.json')['details'][model]
    traces={json.dumps(r['neural_trace'],sort_keys=True) for r in refs}
    common_trace=json.loads(next(iter(traces))) if len(traces)==1 else None
    compact={config_id(c):[] for c in pending};fresh=aliases=0
    for index,(row,reference_path,reference) in enumerate(data):
        require_resources(own_pid=os.getpid())
        todo=[c for c in pending if not (directory(model,c)/'pairs'/f'{row["key"]}.json').exists()]
        if todo:
            record=PairRecord(row['pair_id'],row['category_directory'],int(row['t_ref']),Path(row['possible_dir']),Path(row['impossible_dir']),Path(row['meta_path']))
            possible,impossible=read_pair_rgb(record)
            pixels=hashlib.sha256(possible.tobytes()+impossible.tobytes()).hexdigest()
            history_pixels=hashlib.sha256(possible[:8].tobytes()).hexdigest()
            if pixels!=reference['source_decoded_pixels_sha256']:raise RuntimeError('Original pixels changed')
            expected=reference.get('neural_call_trace',common_trace)
            if expected is None and row['split']=='development':expected=read(OUTPUT/f'{model}_boundary_dev_001/pairs'/f'{row["key"]}.json')['neural_call_trace']
            if expected is None:raise RuntimeError('Original call trace unavailable')
            for c in todo:
                target=directory(model,c)/'pairs'/f'{row["key"]}.json'
                equivalent=find(model,c,row,history_pixels,pixels)
                validation=read(validpath(model,row['key'])) if validpath(model,row['key']).exists() else None
                if equivalent and validation and validation['passed']:
                    payload=save_alias(target,model,c,row,equivalent,reference_path);aliases+=1;continue
                require_resources(own_pid=os.getpid());adapter.prediction_calls=row['archive_index']
                calls=[0];trace=hashlib.sha256()
                def count(module,inputs):
                    calls[0]+=1;trace.update(json.dumps((module.__class__.__module__,module.__class__.__qualname__,describe(inputs)),sort_keys=True).encode())
                counter=torch.nn.modules.module.register_module_forward_pre_hook(count)
                start=time.perf_counter()
                hook_type=ExistingDDLP if model=='ddlp' else ExistingGSWM
                hooks=hook_type(adapter,'feedback',directory(model,c)/'instrumented_sources',params=c,capture_all=not refpath(model,row['key']).exists())
                try:
                    with hooks:result=adapter.score_pair(possible,impossible)
                finally:counter.remove()
                torch.cuda.synchronize()
                observed={'module_forward_invocations':calls[0],'ordered_class_and_input_shapes_sha256':trace.hexdigest()}
                if observed!=expected or adapter.prediction_calls!=row['archive_index']+1:raise RuntimeError('Native forward budget/order/shapes changed')
                if result.frame_spans.tolist()!=reference['frame_spans']:raise RuntimeError('Original time spans changed')
                if hashlib.sha256(possible.tobytes()+impossible.tobytes()).hexdigest()!=pixels:raise RuntimeError('Input pixels mutated')
                if not all(np.isfinite(x).all() for x in (result.possible,result.impossible)):raise RuntimeError('Nonfinite scores')
                if len(hooks.audits)!=(8 if model=='gswm' else 1) or any(len(a['steps'])!=6 for a in hooks.audits):raise RuntimeError('Original MC samples or prediction steps changed')
                register_history(model,row,history_pixels,hooks)
                payload={k:row[k] for k in ('benchmark','archive_index','pair_id','split')}
                payload.update(category=row['category_directory'],t_ref=int(row['t_ref']),frame_spans=result.frame_spans.tolist(),
                    possible_errors=result.possible.tolist(),impossible_errors=result.impossible.tolist(),
                    matched_identity_possible_errors=reference['matched_identity_possible_errors'],matched_identity_impossible_errors=reference['matched_identity_impossible_errors'],
                    model=model,parameters=c,mode='feedback',eta=c['eta'],source_decoded_pixels_sha256=pixels,reference_pair_sha256=digest(reference_path),
                    neural_call_trace=observed,trace_matches_original=True,wall_seconds=time.perf_counter()-start,
                    projection_audits=hooks.audits,instrumented_function_hashes=hooks.functions,execution_mode='native_budget_forecast',extra_forwards=0)
                equivalent=equivalent or find(model,c,row,history_pixels,pixels)
                if equivalent and validation is None:
                    passed=compare(payload,equivalent['source'])
                    save(validpath(model,row['key']),{'passed':passed,'fresh_parameters':c,'source_parameters':equivalent['source_parameters'],
                        'source_result':str(equivalent['record']),'source_sha256':digest(equivalent['record']),
                        'check':'All original per-step errors and frame spans bitwise equal for an actual grid forecast with the same history-defined operator. If unequal, do not reuse this case.'})
                save(target,payload);fresh+=1
        for c in pending:
            r=read(directory(model,c)/'pairs'/f'{row["key"]}.json')
            if r['pair_id']!=row['pair_id'] or r['parameters']!=c or r['reference_pair_sha256']!=digest(reference_path):raise RuntimeError('Resume identity/parameters/source changed')
            compact[config_id(c)].append({k:r[k] for k in FIELDS})
        status(HERE/'monitor'/f'{model}_progress.json',{'pairs_completed_for_all_settings':index+1,'total_pairs':600,'new_configs':len(pending),
            'fresh_native_forecasts_this_session':fresh,'exact_reused_cases_this_session':aliases,'last':row['key'],'at':time.time()})
        if (index+1)%10==0:print(model,index+1,'/600 for all',len(pending),'new settings; fresh',fresh,'equivalent',aliases,flush=True)
    verify()
    for c in pending:
        out=directory(model,c)
        if not (out/'summary.json').exists():
            save(out/'summary.json',dict(make_summary(compact[config_id(c)],model),parameters=c,mode='feedback',alpha_fixed=1,
                old_sources_unchanged=True,all_executed_forecast_traces_matched_original=True,exact_reuse_documented_per_case=True,extra_forwards=0))
    print('COMPLETE',model,'all existing-parameter configurations',flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--model',choices=OBJECTS,required=True);a=p.parse_args()
    try:execute(a.model)
    except ResourceBusy as exc:
        status(HERE/'monitor'/f'{a.model}_resource_pause.json',{'at':time.time(),'reason':str(exc),'released_by_worker_exit':True});sys.exit(75)
    except BaseException as exc:
        save(HERE/'monitor'/f'FAILED_{a.model}_{time.time_ns()}.json',{'error':repr(exc),'traceback':traceback.format_exc()});raise
