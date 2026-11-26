"""Label-free effective radius bounds for existing slot feedback positions."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import math
import time
from . import common as c


def main():
    import torch
    from predictive_tracks.ppr_all_20261029.projection import reference
    from predictive_tracks.state_tracks_20261023.all_models_20261023.run import forbid
    torch.set_num_threads(1)
    handle=torch.nn.modules.module.register_module_forward_pre_hook(forbid)
    parent=c.read(c.source_path())
    results={}
    proposed=[]
    try:
        with torch.inference_mode():
            for model in c.SLOTS:
                all_ratios=[]
                early_ratios=[]
                zero_radius_outside=0
                source_hashes={}
                radii=[]
                for task in ('sp','ie'):
                    for index in range(300):
                        mp=c.HERE/'capture'/model/'pairs'/f'{task}_{index:03d}.json'
                        meta=c.read(mp)
                        assert c.digest(meta['tensor'])==meta['tensor_sha256']
                        data=torch.load(meta['tensor'],map_location='cpu',weights_only=True)
                        h,p=data['history'],data['prediction']
                        center,radius,valid=reference(h)
                        distance=torch.linalg.vector_norm(p-center,dim=-1)
                        positive=radius>0
                        all_ratios.extend((distance[...,positive]/radius[positive]).reshape(-1).tolist())
                        early_ratios.extend((distance[:,:-1,positive]/radius[positive]).reshape(-1).tolist())
                        zero_radius_outside+=int((distance[...,~positive]>0).sum())
                        radii.extend(radius.tolist())
                        source_hashes[str(mp)]=c.digest(mp)
                detail=dict(history_frames=8,cases=600,zero_radius_future_differences=zero_radius_outside,
                    raw_radius_min=min(radii),raw_radius_max=max(radii),source_metadata_hashes=source_hashes,modes={})
                for mode,ratios in (('feedback',all_ratios),('state_only',early_ratios)):
                    previous=[x for x in parent['all_configurations'][model] if x['mode']==mode]
                    high=max(x['parameters'].get('alpha',1.) for x in previous)
                    bound=max(ratios)
                    controls=[]
                    if zero_radius_outside==0:
                        controls=[dict(model=model,mode=mode,eta=1.,alpha=float(math.ceil(bound)),
                            proposal='Unmodified-trajectory no-operation bound control; actual native verification required')]
                    alphas=sorted({round(high+(bound-high)*fraction,8) for fraction in (.25,.5,.75,.9) if bound>high})
                    existing={(x['parameters']['eta'],x['parameters'].get('alpha',1.)) for x in previous}
                    jobs=[dict(model=model,mode=mode,eta=e,alpha=a,proposal='Previously uncovered radius range from raw native states')
                        for a in alphas for e in (.01,.05,.1,.25,.5,.75,1.) if (e,a) not in existing]
                    proposed.extend(jobs)
                    detail['modes'][mode]=dict(previous_max_alpha=high,raw_distance_ratio_max=bound,
                        raw_distance_ratio_quantiles={str(q):c.quantile(ratios,q) for q in (.5,.9,.95,.99,1.)},
                        proposed_alphas=alphas,proposed_settings=len(jobs),proposed_noop_controls=controls,
                        ideal_noop_bound_available=zero_radius_outside==0)
                results[model]=detail
    finally:
        handle.remove()
    assert not torch.cuda.is_initialized()
    c.save(c.HERE/'PARAMETER_DOMAIN_ANALYSIS_015.json',dict(at=time.time(),models=results,
        no_new_model_calls=True,CUDA_initialized=False,only_history_and_raw_prediction_used=True,
        CPU_range_proposal_not_a_CUDA_formal_proof=True,
        derivation='If every raw state is within alpha*R, induction gives no first feedback change and hence the original trajectory. For state_only, the last state write is not consumed by a later forecast.',
        planned_jobs_not_scheduled=True,proposed_settings=len(proposed),jobs=proposed))
    print({m:{k:v for k,v in x.items() if k!='source_metadata_hashes'} for m,x in results.items()},flush=True)


if __name__=='__main__':main()
