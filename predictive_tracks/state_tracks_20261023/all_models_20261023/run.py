"""Capture and replay separately: one original graph, many non-neural controls."""
import argparse
import hashlib
import importlib
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from .common import (HERE, MODELS, RGB, NATIVE_TOLERANCE, rows, key, existing_cache,
                     reference, donors, read, write, verify_all, digest, archive)
from .method import propose, random_control, match_direction
from predictive_tracks.controlled_tracks.native_cache import NativeCache
from predictive_tracks.controlled_tracks.response_full_run import describe
from predictive_tracks.controlled_tracks.orthogonal_boundary import align_orthogonal_boundary
from predictive_tracks.controlled_tracks.response_calibration import response_calibration
from predictive_tracks.controlled_tracks.full_metrics_report import aggregate
from sp_benchmark.protocol_metrics import summarize


def forbid(module, inputs):
    raise RuntimeError('Extra neural call forbidden during analytic track')


def base_row(row, spans, baseline):
    result={k:row[k] for k in ('benchmark','pair_id','archive_index','split')}
    result.update(category=row['category_directory'],t_ref=int(row['t_ref']),frame_spans=spans)
    for name,value in zip(('possible','impossible'),baseline):
        result['matched_identity_'+name+'_errors']=np.asarray(value).tolist()
    return result


def signature(errors,row,model):
    ep,ei=(np.asarray(e).reshape(1,-1) for e in errors)
    result=summarize(ep,ei,frame_spans=np.asarray(row['frame_spans']),
                     violation_frames=np.asarray([row['t_ref']]),epsilon=1e-12,threshold=0.,
                     persistence_frames=2,joint_tolerances=(NATIVE_TOLERANCE[model],))
    return result['paired_accuracy'], result['t_star'], result['delta_t'], result['joint_hit']


def validate_errors(actual,wanted,row,model):
    for a,b in zip(actual,wanted):
        if not np.allclose(a,b,rtol=1e-5,atol=1e-7):
            raise RuntimeError('Native tensor score mismatch')
    if signature(actual,row,model)!=signature(wanted,row,model):
        raise RuntimeError('Native score rounding changed a benchmark decision/localization')


def capture(model, directory):
    verify_all();torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
    native=directory/'native';native.mkdir(parents=True,exist_ok=True)
    if (native/'summary.json').exists(): return
    adapter=None; new_count=0; reused=0; metadata=[]
    for i,row in enumerate(rows()):
        identifier=key(row); destination=native/'pairs'/(identifier+'.json')
        if destination.exists():
            prior=read(destination)
            if digest(prior['state'])!=prior['sha256']: raise RuntimeError('Completed cache changed')
            metadata.append(prior);continue
        source=existing_cache(model,row)
        if source is not None:
            if digest(source['state'])!=source['sha256']: raise RuntimeError('Previous cache changed')
            original=read(source['meta'])
            baseline=[original['matched_identity_'+b+'_errors'] for b in ('possible','impossible')]
            record=base_row(row,original['frame_spans'],baseline)
            record.update(source,reused_previous_native_cache=True,
                          original_source_meta_sha256=digest(source['meta']))
            write(destination,record);metadata.append(record);reused+=1
            continue
        if adapter is None:
            if not torch.cuda.is_available(): raise RuntimeError('Original CUDA unavailable; no CPU neural fallback')
            adapter=importlib.import_module(MODELS[model]).Adapter(device='cuda:0',memory_mode='safe',orientation='identity')
            provenance=native/'model_provenance.json'
            if not provenance.exists(): write(provenance,adapter.provenance())
        from sp_core.common import PairRecord,read_pair_rgb
        pair=PairRecord(row['pair_id'],row['category_directory'],int(row['t_ref']),
                        Path(row['possible_dir']),Path(row['impossible_dir']),Path(row['meta_path']))
        possible,impossible=read_pair_rgb(pair)
        pixels=hashlib.sha256(possible.tobytes()+impossible.tobytes()).hexdigest()
        previous_path=reference(model,row)
        previous=read(previous_path) if previous_path else None
        if previous and (previous['pair_id']!=row['pair_id'] or previous['source_decoded_pixels_sha256']!=pixels):
            raise RuntimeError('Original identity/pixels changed')
        adapter.prediction_calls=row['archive_index']
        trace=hashlib.sha256();count=[0]
        def record_call(module,inputs):
            count[0]+=1
            event=(module.__class__.__module__,module.__class__.__qualname__,describe(inputs))
            import json
            trace.update(json.dumps(event,sort_keys=True).encode())
        hook=torch.nn.modules.module.register_module_forward_pre_hook(record_call)
        cache=NativeCache(adapter,model);started=time.perf_counter()
        try:
            with cache: outcome=adapter.score_pair(possible,impossible)
        finally: hook.remove()
        torch.cuda.synchronize();elapsed=time.perf_counter()-started
        neural={'module_forward_invocations':count[0],'ordered_class_and_input_shapes_sha256':trace.hexdigest()}
        if previous and neural!=previous['neural_call_trace']: raise RuntimeError('Original neural graph trace changed')
        if adapter.prediction_calls!=row['archive_index']+1: raise RuntimeError('Forecast budget changed')
        if hashlib.sha256(possible.tobytes()+impossible.tobytes()).hexdigest()!=pixels: raise RuntimeError('Source pixels mutated')
        guard=torch.nn.modules.module.register_module_forward_pre_hook(forbid)
        try:
            cache.finish_targets(possible,impossible)
            audit=cache.audit_and_controls(outcome)
            baseline=[np.asarray(audit['native_prediction_'+b+'_errors']) for b in ('possible','impossible')]
            record=base_row(row,outcome.frame_spans.tolist(),baseline)
            if previous:
                if record['frame_spans']!=previous['frame_spans']: raise RuntimeError('Original native spans changed')
                validate_errors(baseline,[previous['matched_identity_'+b+'_errors'] for b in ('possible','impossible')],record,model)
            else:
                with np.load(archive(row['benchmark'])/'models'/model/'raw/error_arrays.npz',allow_pickle=False) as old:
                    if old['pair_ids'][row['archive_index']]!=row['pair_id']: raise RuntimeError('DINO archive ordering changed')
                    if not np.array_equal(old['frame_spans'],outcome.frame_spans): raise RuntimeError('DINO native spans changed')
                    record['archived_error_max_abs_difference']=float(max(np.max(np.abs(baseline[j]-old[b+'_errors'][row['archive_index']])) for j,b in enumerate(('possible','impossible'))))
            data={k:v for k,v in cache.data.items() if k in ('prediction','history','possible_target','impossible_target','alpha')}
            states=native/'states';states.mkdir(exist_ok=True)
            attempt=1
            state=states/f'{identifier}_{attempt:03d}.pt'
            while state.exists():
                attempt+=1;state=states/f'{identifier}_{attempt:03d}.pt'
            with state.open('xb') as f: torch.save(data,f)
            record.update(state=str(state),sha256=digest(state),layout='native',reused_previous_native_cache=False,
                          source_decoded_pixels_sha256=pixels,neural_call_trace=neural,
                          original_trace_match=previous is not None,original_graph_seconds=elapsed,
                          score_replay_audit=audit,additional_neural_forwards=0)
            write(destination,record);metadata.append(record);new_count+=1
        finally: guard.remove()
        if (i+1)%30==0: print(f'CAPTURE {model} {i+1}/600, original-budget graph, {elapsed:.2f}s latest',flush=True)
    if len(metadata)!=600: raise RuntimeError('Incomplete original dataset')
    verify_all()
    write(native/'summary.json',{'model':model,'cases':600,'new_cases_this_invocation':new_count,
                                'reused_cases_this_invocation':reused,'additional_forwards_per_case':0,
                                'source_preservation_verified':True})
    print('CAPTURE_COMPLETE',model,flush=True)


def load_data(meta):
    if digest(meta['state'])!=meta['sha256']: raise RuntimeError('Native cache changed')
    data=torch.load(meta['state'],map_location='cpu',weights_only=True)
    if meta['layout']=='ddlp_particle':
        data={name:data[key].unsqueeze(0) for name,key in (
            ('history','history_rgb'),('prediction','future_rec'),
            ('possible_target','possible_target'),('impossible_target','impossible_target'))}
    return data


def evaluator_for(data,model):
    evaluator=NativeCache(SimpleNamespace(device=torch.device('cpu')),model)
    evaluator.data=data;evaluator.targets=[data['possible_target'],data['impossible_target']]
    return evaluator


def variant_record(meta,errors,elapsed=0.,audit=None):
    r={k:meta[k] for k in ('benchmark','pair_id','archive_index','split','category','t_ref','frame_spans',
                          'matched_identity_possible_errors','matched_identity_impossible_errors')}
    r.update(possible_errors=np.asarray(errors[0]).tolist(),impossible_errors=np.asarray(errors[1]).tolist(),
             analytic_seconds=elapsed,audit=audit or {})
    return r


def replay(model,directory):
    verify_all();torch.set_num_threads(4)
    if not (directory/'native/summary.json').exists(): raise RuntimeError('Finish native capture first')
    if (directory/'summary.json').exists(): return
    records=rows(); donor_map=donors()
    guard=torch.nn.modules.module.register_module_forward_pre_hook(forbid)
    try:
        for index,row in enumerate(records):
            identifier=key(row);path=directory/'core_pairs'/(identifier+'.json')
            if path.exists(): continue
            meta=read(directory/'native/pairs'/(identifier+'.json'))
            data=load_data(meta);evaluation=evaluator_for(data,model)
            with torch.inference_mode():
                p,h=data['prediction'],data['history'];native=evaluation.errors(p)
                validate_errors(native,[meta['matched_identity_'+b+'_errors'] for b in ('possible','impossible')],meta,model)
                start=time.perf_counter();outputs,audit=propose(p,h,rgb=model in RGB)
                method_seconds=time.perf_counter()-start
                payload={'model':model,'case':identifier,'meta':meta,'variants':{},'method_audit':audit,
                         'method_seconds':method_seconds,'neural_forwards_in_replay':0}
                payload['variants']['native']=variant_record(meta,native)
                for name,value in outputs.items():
                    payload['variants'][name]=variant_record(meta,evaluation.errors(value),method_seconds,audit[name])
                delta=outputs['state_tube']-p
                deltas=directory/'deltas';deltas.mkdir(exist_ok=True)
                target=deltas/(identifier+'.pt')
                if not target.exists():
                    with target.open('xb') as f: torch.save(delta,f)
                payload['primary_delta_sha256']=digest(target)
                for seed in (20261023,20261024,20261025):
                    start=time.perf_counter();value,control=random_control(p,delta,seed,rgb=model in RGB)
                    payload['variants'][f'random_{seed}']=variant_record(meta,evaluation.errors(value),time.perf_counter()-start,control)
                for name,fn in (('frozen_parallel',lambda:align_orthogonal_boundary(p,h,rgb=model in RGB,component='parallel')),
                                ('frozen_response',lambda:response_calibration(p,h,rgb=model in RGB,mode='response'))):
                    start=time.perf_counter();value,control=fn()
                    error=evaluation.errors(value)
                    if name=='frozen_parallel' and model!='dino_foresight':
                        old=read(reference(model,row))
                        validate_errors(error,[old[b+'_errors'] for b in ('possible','impossible')],meta,model)
                    payload['variants'][name]=variant_record(meta,error,time.perf_counter()-start,control)
                write(path,payload)
            if (index+1)%30==0: print(f'REPLAY {model} core {index+1}/600; zero neural forwards',flush=True)
        for index,row in enumerate(records):
            identifier=key(row);target=directory/'pairs'/(identifier+'.json')
            if target.exists(): continue
            payload=read(directory/'core_pairs'/(identifier+'.json'))
            own_delta=directory/'deltas'/(identifier+'.pt')
            donor_key=donor_map[identifier]
            donor_delta=directory/'deltas'/(donor_key+'.pt')
            donor_payload=read(directory/'core_pairs'/(donor_key+'.json'))
            if digest(own_delta)!=payload['primary_delta_sha256'] or digest(donor_delta)!=donor_payload['primary_delta_sha256']:
                raise RuntimeError('Control delta changed')
            with torch.inference_mode():
                data=load_data(payload['meta']);evaluation=evaluator_for(data,model)
                delta=torch.load(own_delta,map_location='cpu',weights_only=True)
                direction=torch.load(donor_delta,map_location='cpu',weights_only=True)
                start=time.perf_counter()
                value,audit=match_direction(data['prediction'],direction,float(torch.linalg.vector_norm(delta.double())),rgb=model in RGB)
                audit['donor_key']=donor_key
                payload['variants']['donor_norm_matched']=variant_record(payload['meta'],evaluation.errors(value),time.perf_counter()-start,audit)
            write(target,payload)
            if (index+1)%100==0: print(f'REPLAY {model} donor {index+1}/600; zero neural forwards',flush=True)
    finally: guard.remove()
    collect=[read(directory/'pairs'/(key(r)+'.json')) for r in records]
    report={'model':model,'cases':len(collect),'diagnostic_only':model=='dino_foresight','metrics':{}}
    for scope,split in (('full300',None),('development30','development'),('remaining270','confirmation')):
        report['metrics'][scope]={}
        for task in ('sp','ie'):
            selected=[p for p in collect if p['meta']['benchmark']==task and (split is None or p['meta']['split']==split)]
            expected=300 if split is None else (30 if split=='development' else 270)
            if len(selected)!=expected or len({p['meta']['pair_id'] for p in selected})!=expected: raise RuntimeError('Pair coverage changed')
            report['metrics'][scope][task]={v:aggregate([p['variants'][v] for p in selected],model)['corrected'] for v in read(HERE/'plan.json')['variants']}
    verify_all();report['source_preservation_verified']=True
    report['new_neural_forwards_in_replay']=0
    write(directory/'summary.json',report)
    print('MODEL_COMPLETE',model,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=MODELS,required=True)
    parser.add_argument('--name',required=True);parser.add_argument('--stage',choices=('capture','replay'),required=True)
    args=parser.parse_args()
    if Path(args.name).name!=args.name: raise ValueError('Simple new run name required')
    directory=HERE/'runs'/args.name/args.model;directory.mkdir(parents=True,exist_ok=True)
    try:
        (capture if args.stage=='capture' else replay)(args.model,directory)
    except BaseException as exc:
        name=f'FAILED_{args.stage}_{time.time_ns()}.json'
        write(directory/name,{'exception':repr(exc),'traceback':traceback.format_exc()})
        raise


if __name__=='__main__': main()
