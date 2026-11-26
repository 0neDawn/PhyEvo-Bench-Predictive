"""Search exact original suffixes, with original-pipeline equivalence checks."""
import argparse
import hashlib
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from . import common as c
sys.path.insert(0,str(c.ROOT/'predictive_tracks/sp/code'))
import numpy as np
import torch
from predictive_tracks.controlled_tracks.registry import MODELS
from predictive_tracks.controlled_tracks.response_full_run import describe
from predictive_tracks.ppr_20261028.run import rows_and_references
from predictive_tracks.ppr_20261028.hooks import Hooks
from predictive_tracks.ppr_eta_20261030.cache import meta_path
from predictive_tracks.ppr_alpha_20261031.projection import trajectory
from predictive_tracks.ppr_all_20261029.projection import reference
from predictive_tracks.ppr_eta_radius_20261104_o.native_worker import versions
from predictive_tracks.ppr_internal_activation_20261106_r.tail_worker import (
    cpu,cuda,native_prefix,prepare_shift,shift,expected_trace,exact)
from predictive_tracks.ppr_state_insert_20261104_m.hooks import tensor_hash
from predictive_tracks.state_tracks_20261023.all_models_20261023.run import load_data,evaluator_for,variant_record
from predictive_tracks.ppr_existing_20261102.metrics import make_summary
from predictive_tracks.ppr_taskwise_best_20261102_001.select_report import rank
from sp_core.common import PairRecord,read_pair_rgb
from .hooks import factory,LayerShift


def adapter_for(model):
    torch.set_num_threads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.set_per_process_memory_fraction(.45 if model in c.FEATURES else .2,device=0)
    return importlib.import_module(MODELS[model]).Adapter(device='cuda:0',memory_mode='safe',orientation='identity')


def modules(adapter,model):
    if model=='videomaev2':
        return adapter.model.decoder.blocks,adapter.model.decoder.norm
    return adapter.predictor.predictor_blocks,adapter.predictor.predictor_norm


def positions(adapter,model):
    blocks,_=modules(adapter,model)
    depth=len(blocks)
    if model=='videomaev2':
        indices=list(range(depth-1)) if depth<=4 else sorted({depth//2,depth-2})
    else:
        indices=sorted({depth//2,max(0,depth-3)})
    # The old final placement is retained only as a full600 confirmation.
    return [f'block_{i}' for i in indices]+['before_norm',f'block_{depth-1}']


class SlotCapture(Hooks):
    def __init__(self,adapter):
        super().__init__(adapter,'audit',None)
        self.history=None
        self.prediction=None
        self.decoded=None
    def begin(self,history):
        assert self.history is None and tuple(history.shape[:2])==(1,8)
        self.history=cpu(history)
    def __enter__(self):
        self.instrument(self.adapter,'_predict_decoded_rgb',[
            ('history_slots = torch.stack(history_slots, dim=1)',
             'history_slots = torch.stack(history_slots, dim=1)\n    _ppr.begin(history_slots)'),
            ('return decoded','_ppr.decoded = decoded.detach().cpu().clone()\n    return decoded')])
        def capture(module,args,output):
            assert self.prediction is None
            self.prediction=cpu(output)
        self.handle=self.adapter.predictor.register_forward_hook(capture)
        return self
    def __exit__(self,*args):
        self.handle.remove()
        super().__exit__(*args)


def payload_path(model,key):
    return c.HERE/'capture'/model/'pairs'/f'{key}.json'


def read_capture(model,key):
    meta=c.read(payload_path(model,key))
    assert c.digest(meta['tensor'])==meta['tensor_sha256']
    assert c.digest(meta['source_meta'])==meta['source_meta_sha256']
    return meta,torch.load(meta['tensor'],map_location='cpu',weights_only=True)


def move_suffix(adapter,model):
    if model=='videomaev2':
        adapter.model.decoder.to('cuda:0')
    elif model=='vjepa2_vitl':
        adapter.predictor.to('cuda:0')
    else:
        adapter.savi.to('cuda:0')


def predicted(adapter,model,item,data):
    """Replay exactly the unmodified suffix, with one radial insertion."""
    if model in c.SLOTS:
        p,h=data['prediction'],data['history']
        hard,_=trajectory(p,h,item['alpha'])
        e=item['eta']
        value=p if e==0 else hard if e==1 else p+e*(hard-p)
        decoded=adapter.savi.decode(value.reshape(6,6,128))[0]
        return decoded.reshape(1,6,3,64,64).clamp(0.,1.)
    args,kwargs=data[item['position']]
    values,rest=args[0],args[1:]
    modified=shift(values,prepare_shift(values,item['alpha']),item['eta'])
    blocks,norm=modules(adapter,model)
    first=len(blocks) if item['position']=='before_norm' else int(item['position'].split('_')[-1])
    x=modified
    for block in blocks[first:]:
        x=block(x,*rest,**kwargs)
    x=norm(x)
    if model=='vjepa2_vitl':
        mask=data['original_mask']
        identity=torch.arange(7*256,device=mask.device)[None]
        assert torch.equal(mask,identity)
        assert not adapter.predictor.return_all_tokens and adapter.predictor.chop_last_n_tokens==0
        reverse=torch.argsort(torch.argsort(mask,dim=1),dim=1)
        x=torch.stack([x[i,row,:] for i,row in enumerate(reverse)],dim=0)
        x=x[:,4*256:]
        x=adapter.predictor.predictor_proj(x)
    else:
        x=x[:,-3*256:]
    return x.reshape(1,3,256,-1)


def trace_counter():
    count=[0]
    trace=hashlib.sha256()
    def record(module,inputs):
        count[0]+=1
        trace.update(json.dumps((module.__class__.__module__,module.__class__.__qualname__,describe(inputs)),sort_keys=True).encode())
    handle=torch.nn.modules.module.register_module_forward_pre_hook(record)
    return count,trace,handle


def full_check(adapter,model,row,ref,possible,impossible,data,evaluator,item):
    p=c.HERE/'preflight'/model/(item.get('position') or item['mode'])/f'{row["key"]}_e{item["eta"]:.8g}_a{item["alpha"]:.8g}.json'
    if p.exists():
        return
    move_suffix(adapter,model)
    if model in c.FEATURES:
        args,_=data[item['position']]
        original=LayerShift(adapter,item).shift(args[0])
        optimized=shift(args[0],prepare_shift(args[0],item['alpha']),item['eta'])
        assert torch.equal(original,optimized),'Search expression differs from original formula'
    replay=evaluator.errors(predicted(adapter,model,item,data))
    before=versions(adapter,model)
    adapter.prediction_calls=row['archive_index']
    count,trace,handle=trace_counter()
    try:
        hook=factory(adapter,item,None)
        with hook:
            result=adapter.score_pair(possible,impossible)
    finally:
        handle.remove()
    exact(replay,(result.possible,result.impossible))
    observed=dict(module_forward_invocations=count[0],ordered_class_and_input_shapes_sha256=trace.hexdigest())
    assert observed==expected_trace(model,row,ref)
    assert adapter.prediction_calls==row['archive_index']+1
    assert versions(adapter,model)==before
    assert result.frame_spans.tolist()==ref['frame_spans']
    if model in c.FEATURES:
        assert hook.activations==1 and hook.inactive_calls==(6 if model=='videomaev2' else 0)
    assert len(hook.audits)==1 and len(hook.audits[0]['steps'])==(3 if model in c.FEATURES else 6)
    c.save(p,dict(at=time.time(),job=item,key=row['key'],original_trace_exact=True,
        full_vs_suffix_max_abs_error=0.,weights_identity_version_unchanged=True,
        no_extra_forward_in_deployed_forecast=True,original_trace=observed))


def capture(model):
    c.verify()
    c.require_resources(model)
    adapter=adapter_for(model)
    before=versions(adapter,model)
    ps=positions(adapter,model) if model in c.FEATURES else ['output_only']
    architecture=c.HERE/'capture'/model/'architecture.json'
    declaration=dict(model=model,positions=ps,original_depth=len(modules(adapter,model)[0]) if model in c.FEATURES else None,
        one_projection_position_per_configuration=True)
    if architecture.exists():
        assert c.read(architecture)==declaration
    else:
        c.save(architecture,declaration)
    data_rows=sorted(rows_and_references(model),key=lambda x:(x[0]['benchmark']!='ie',x[0]['archive_index']))
    old=c.install()
    for index,(row,rp,ref) in enumerate(data_rows,1):
        dest=payload_path(model,row['key'])
        if dest.exists():
            meta,_=read_capture(model,row['key'])
            assert meta['positions']==ps
            continue
        old.before_native(model)
        mp=meta_path(model,row)
        source=c.read(mp)
        cached=load_data(source)
        pair=PairRecord(row['pair_id'],row['category_directory'],int(row['t_ref']),Path(row['possible_dir']),Path(row['impossible_dir']),Path(row['meta_path']))
        possible,impossible=read_pair_rgb(pair)
        pixels=hashlib.sha256(possible.tobytes()+impossible.tobytes()).hexdigest()
        assert pixels==ref['source_decoded_pixels_sha256']
        adapter.prediction_calls=row['archive_index']
        if model in c.SLOTS:
            count,trace,handle=trace_counter()
            try:
                capture_hook=SlotCapture(adapter)
                with capture_hook:
                    result=adapter.score_pair(possible,impossible)
            finally:
                handle.remove()
            observed=dict(module_forward_invocations=count[0],ordered_class_and_input_shapes_sha256=trace.hexdigest())
            assert observed==expected_trace(model,row,ref)
            exact((result.possible,result.impossible),[source['matched_identity_'+b+'_errors'] for b in ('possible','impossible')])
            assert adapter.prediction_calls==row['archive_index']+1
            assert torch.equal(capture_hook.decoded,cached['prediction'])
            tensors=dict(history=capture_hook.history,prediction=capture_hook.prediction)
        else:
            blocks,norm=modules(adapter,model)
            tensors={}
            handles=[]
            def collector(position):
                def before_call(module,args,kwargs):
                    assert position not in tensors
                    tensors[position]=(cpu(args),cpu(kwargs))
                return before_call
            try:
                for position in ps:
                    mod=norm if position=='before_norm' else blocks[int(position.split('_')[-1])]
                    handles.append(mod.register_forward_pre_hook(collector(position),with_kwargs=True))
                raw=native_prefix(adapter,model,possible)
            finally:
                for handle in handles:
                    handle.remove()
            assert torch.equal(raw.detach().cpu(),cached['prediction']),'Original prediction differs from old cache'
            assert set(tensors)==set(ps)
            if model=='vjepa2_vitl':
                masks=[kwargs['mask'] for args,kwargs in tensors.values() if 'mask' in kwargs]
                assert masks and all(torch.equal(m,masks[0]) for m in masks)
                tensors['original_mask']=masks[0]
        assert versions(adapter,model)==before
        tensor=c.HERE/'capture'/model/'states'/f'{row["key"]}.pt'
        if tensor.exists():
            # Preserve an orphan left by a power loss; never replace it.
            n=2
            while tensor.with_name(f'{row["key"]}_{n:03d}.pt').exists():
                n+=1
            tensor=tensor.with_name(f'{row["key"]}_{n:03d}.pt')
        c.save_tensor(tensor,tensors)
        c.save(dest,dict(at=time.time(),key=row['key'],model=model,positions=ps,tensor=str(tensor),tensor_sha256=c.digest(tensor),
            source_meta=str(mp),source_meta_sha256=c.digest(mp),source_state_sha256=source['sha256'],
            original_reference=str(rp),original_reference_sha256=c.digest(rp),source_pixels_sha256=pixels,
            original_prediction_bitwise_equal=True,weights_identity_version_unchanged=True,
            capture_is_original_unmodified_forecast=True))
        c.status(c.HERE/'capture'/model/'progress.json',dict(at=time.time(),completed=index,total=600,positions=ps))
        if index%20==0:
            print('S CAPTURE',model,index,'/600',flush=True)
        del cached,tensors,possible,impossible
    c.save(c.HERE/'models_complete'/f'capture_{model}.json',dict(at=time.time(),cases=600,positions=ps))
    make_coarse_grid(model)


def make_coarse_grid(model):
    path=c.HERE/'grids'/f'coarse_{model}.json'
    if path.exists():
        return
    architecture=c.read(c.HERE/'capture'/model/'architecture.json')
    ratios={p:[] for p in architecture['positions']}
    input_hashes={}
    for key in [f'{t}_{i:03d}' for t in ('sp','ie') for i in range(300)]:
        meta,data=read_capture(model,key)
        input_hashes[str(payload_path(model,key))]=c.digest(payload_path(model,key))
        for position in architecture['positions']:
            if model in c.SLOTS:
                history,prediction=data['history'],data['prediction']
            else:
                values=data[position][0][0].reshape(1,7,256,-1)
                history,prediction=values[:,:4].mean(2,keepdim=True),values[:,4:].mean(2,keepdim=True)
            center,radius,valid=reference(history)
            distance=torch.linalg.vector_norm(prediction-center,dim=-1)
            good=valid & (radius>0)
            ratios[position].extend((distance[...,good]/radius[good]).reshape(-1).double().tolist())
    parent=c.read(c.source_path())
    oldrows=parent['all_configurations'][model]
    jobs=[]
    final_position=f'block_{architecture["original_depth"]-1}' if model in c.FEATURES else None
    for position in architecture['positions']:
        mode='output_only' if model in c.SLOTS else 'pre_final_block'
        if position!=final_position:
            alphas=c.grid_alphas(ratios[position])
            for alpha in alphas:
                for eta in c.ETAS:
                    if model in c.SLOTS and any(v['mode']==mode and v['parameters']['eta']==eta and v['parameters'].get('alpha',1.)==alpha for v in oldrows):
                        continue
                    jobs.append(c.job(model,mode,eta,alpha,None if model in c.SLOTS else position,
                        proposal='label-free native activation scale; explicit original placement'))
        # Two existing full600 settings are exact cross-checks, not new gains.
        eligible=[v for v in oldrows if v['mode']==mode and v.get('eligible_for_selection',True)]
        if eligible and (model in c.SLOTS or position==final_position):
            seen=set()
            for task in ('sp','ie'):
                v=min(eligible,key=lambda x:rank(x,task))
                pp=v['parameters'];pair=(pp['eta'],pp.get('alpha',1.))
                if pair in seen:
                    continue
                seen.add(pair)
                jobs=[j for j in jobs if not(j['eta']==pair[0] and j['alpha']==pair[1] and j.get('position')==(None if model in c.SLOTS else position))]
                jobs.append(c.job(model,mode,*pair,None if model in c.SLOTS else position,confirmation_only=True,
                    confirmation_source=v.get('directory') or v.get('source_directory'),confirmation_row=v,
                    proposal='old full600 exact confirmation, excluded from new selection'))
    c.freeze_grid(f'coarse_{model}',dict(at=time.time(),model=model,jobs=jobs,settings=len(jobs),
        ratios_quantiles={p:{str(q):c.quantile(v,q) for q in (0.,.1,.25,.5,.75,.9,1.)} for p,v in ratios.items()},
        proposal_uses_only_history_and_original_prediction=True,input_hashes=input_hashes))
    print('S COARSE GRID',model,len(jobs),flush=True)


def expand(payload,item):
    value=dict(payload['native'],**payload['settings'][item['config_id']])
    value.update(model=item['model'],eta=item['eta'],radius_multiplier=item['alpha'],mode=item['mode'],
        placement=item.get('position'),partial_graph_search=True,extra_forwards_in_deployed_forecast=0)
    return value


def execute_search(model,phase='coarse'):
    c.verify()
    c.require_resources(model)
    grid=c.read(c.HERE/'grids'/f'{phase}_{model}.json')
    jobs=grid['jobs']
    root=c.HERE/'search'/phase/model
    done=c.HERE/'models_complete'/f'{phase}_{model}.json'
    if done.exists():
        return
    if not jobs:
        c.save(done,dict(at=time.time(),settings=0,cases_each=600))
        return
    adapter=adapter_for(model)
    before=versions(adapter,model)
    data_rows=sorted(rows_and_references(model),key=lambda x:(x[0]['benchmark']!='ie',x[0]['archive_index']))
    preflight={f'{t}_{i:03d}' for t in ('sp','ie') for i in (0,100,200)}
    original_confirmations={}
    from predictive_tracks.ppr_eta_radius_20261104_o.path_runtime_001 import install as install_paths
    install_paths()
    from predictive_tracks.ppr_eta_radius_20261104_o.report import records
    for item in jobs:
        if item.get('confirmation_row'):
            original_confirmations[item['config_id']]={f'{r["benchmark"]}_{r["archive_index"]:03d}':r for r in records(item['confirmation_row'])}
    old=c.install()
    with torch.inference_mode():
        for index,(row,rp,ref) in enumerate(data_rows,1):
            dest=root/'pairs'/f'{row["key"]}.json'
            if dest.exists():
                oldrow=c.read(dest)
                assert set(oldrow['settings'])=={j['config_id'] for j in jobs}
                continue
            c.require_resources(model)
            meta,loaded=read_capture(model,row['key'])
            source=c.read(meta['source_meta'])
            cached=load_data(source)
            data=cuda(loaded)
            evaluator=evaluator_for(cached,model)
            evaluator.adapter.device=torch.device('cuda:0')
            evaluator.targets=[v.to('cuda:0') for v in evaluator.targets]
            raw=cached['prediction'].to('cuda:0')
            native=evaluator.errors(raw)
            exact(native,[source['matched_identity_'+b+'_errors'] for b in ('possible','impossible')])
            move_suffix(adapter,model)
            for position in sorted({j.get('position') for j in jobs},key=str):
                sample=next(j for j in jobs if j.get('position')==position)
                eta0=dict(sample,eta=0.,alpha=1.)
                unmodified=predicted(adapter,model,eta0,data)
                assert torch.equal(unmodified,raw),'Original suffix does not reproduce cached prediction'
                if row['key'] in preflight:
                    pair=PairRecord(row['pair_id'],row['category_directory'],int(row['t_ref']),Path(row['possible_dir']),Path(row['impossible_dir']),Path(row['meta_path']))
                    possible,impossible=read_pair_rgb(pair)
                    for e in (0.,.1):
                        full_check(adapter,model,row,ref,possible,impossible,data,evaluator,dict(sample,eta=e))
                    move_suffix(adapter,model)
            settings={}
            confirmed={}
            for item in jobs:
                output=predicted(adapter,model,item,data)
                errors=evaluator.errors(output)
                identifier=item['config_id']
                if identifier in original_confirmations:
                    original=original_confirmations[identifier][row['key']]
                    exact(errors,[original[b+'_errors'] for b in ('possible','impossible')])
                    confirmed[identifier]=True
                settings[identifier]=dict(possible_errors=np.asarray(errors[0]).tolist(),impossible_errors=np.asarray(errors[1]).tolist())
            assert versions(adapter,model)==before
            c.save(dest,dict(native=variant_record(source,native),settings=settings,confirmed_original_settings=confirmed,
                capture_meta=str(payload_path(model,row['key'])),capture_meta_sha256=c.digest(payload_path(model,row['key'])),
                source_meta=meta['source_meta'],source_meta_sha256=meta['source_meta_sha256'],source_state_sha256=meta['source_state_sha256'],
                original_native_errors_exact=True,original_suffix_output_bitwise_equal=True,
                partial_original_graph_search=True,original_batch_dtype_device_preserved=True,
                extra_forwards_in_deployed_forecast=0))
            c.status(root/'progress.json',dict(at=time.time(),model=model,phase=phase,completed=index,total=600,settings=len(jobs)))
            if index%10==0:
                print('S SEARCH',phase,model,index,'/600',len(jobs),'settings',flush=True)
            del data,cached,raw,evaluator,loaded,settings,output,unmodified
    payloads=[c.read(p) for p in sorted((root/'pairs').glob('*.json'))]
    assert len(payloads)==600
    for item in jobs:
        folder=c.HERE/'search_settings'/model/item['config_id']
        if (folder/'summary.json').exists():
            continue
        summary=make_summary([expand(v,item) for v in payloads],model)
        summary.update(job=item,compact_case_source=str(root/'pairs'),complete=True,
            partial_original_graph_search=True,full600_native_validation_required=True)
        c.save(folder/'summary.json',summary)
    c.verify()
    c.save(done,dict(at=time.time(),model=model,settings=len(jobs),cases_each=600,all_original_metrics=True))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=c.SLOTS+c.FEATURES,required=True)
    parser.add_argument('--phase',choices=('capture','coarse','fine'),required=True)
    args=parser.parse_args()
    try:
        c.install()
        with torch.inference_mode():
            if args.phase=='capture':
                capture(args.model)
            else:
                execute_search(args.model,args.phase)
    except BaseException as error:
        from predictive_tracks.ppr_eta_radius_20261104_o.common import ResourceBusy
        pause=isinstance(error,ResourceBusy)
        c.save(c.HERE/'events'/f'{"pause" if pause else "error"}_{time.time_ns()}.json',
            dict(at=time.time(),model=args.model,phase=args.phase,error=repr(error),traceback=traceback.format_exc()))
        if pause:
            raise SystemExit(75)
        raise


if __name__=='__main__':
    main()
