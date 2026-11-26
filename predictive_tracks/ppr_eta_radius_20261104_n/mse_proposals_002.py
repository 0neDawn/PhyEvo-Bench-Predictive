"""Propose global eta/radius from saved MSE states; never report ideal scores."""
import sys
import time
import numpy as np
import torch
from . import common as c
sys.path.insert(0,str(c.ROOT/'predictive_tracks/sp/code'))
from predictive_tracks.ppr_ie_eta_20261103_f.loss_curves import decisions

def intervals(base,slope,spans,tref,alpha,previous,best_actual):
    """All eta sign intervals for one fixed alpha, in real arithmetic only."""
    with np.errstate(divide='ignore',invalid='ignore'):
        roots=np.r_[(-base/slope).ravel(),-base.mean(1)/slope.mean(1)]
    roots=roots[np.isfinite(roots)&(roots>0)&(roots<1)]
    edges=np.unique(np.r_[0.,roots,1.]);eta=(edges[:-1]+edges[1:])/2
    width=np.diff(edges);out=[]
    for start in range(0,len(eta),256):
        es=eta[start:start+256];k=len(es)
        q=base[None]+es[:,None,None]*slope[None]
        a,b,_=decisions(q.transpose(2,1,0).reshape(base.shape[1],-1),spans,np.repeat(tref,k),2)
        pa=a.reshape(300,k).sum(0);hit=b.reshape(300,k).sum(0)
        for j,e in enumerate(es):
            counts=(int(hit[j]),int(pa[j]));rounded=round(float(e),6)
            if counts<=best_actual or rounded<=0 or rounded>=1 or (rounded,alpha) in previous:continue
            at=start+j
            # Rounding must remain inside the proposed eta interval.
            if not edges[at]<rounded<edges[at+1]:continue
            s=c.setting('videomaev2','output_only',rounded,alpha)
            s.update(proposal_only_counts=list(counts),proposal_interval=[float(edges[at]),float(edges[at+1])],
                proposal_width=float(width[at]),proposal_kind='MSE affine-margin ideal proposal from archived states; original CUDA validation mandatory')
            out.append(s)
    out.sort(key=lambda x:(-x['proposal_only_counts'][0],-x['proposal_only_counts'][1],-x['proposal_width'],x['config_id']))
    return out[:6]

def generate(best_actual,previous):
    """Correction sees only prediction/history. Targets enter this tuning analysis."""
    from predictive_tracks.ppr_all_20261029.common import rows
    from predictive_tracks.ppr_all_20261029.projection import reference
    from predictive_tracks.ppr_eta_20261030.cache import meta_path
    from predictive_tracks.state_tracks_20261023.all_models_20261023.run import forbid,load_data
    assert (c.HERE/'STAGE1_COMPLETE.json').exists()
    coeff=c.HERE/'MSE_IDEAL_COEFFICIENTS_002.json'
    guard=torch.nn.modules.module.register_module_forward_pre_hook(forbid)
    try:
        if coeff.exists():data=c.read(coeff)
        else:
            rows_out=[]
            for index,row in enumerate(sorted((r for r in rows() if r['benchmark']=='ie'),key=lambda r:r['archive_index'])):
                if not c.priority_complete():raise c.ResourceBusy('Priority work changed; stop CPU proposal analysis')
                mp=meta_path('videomaev2',row);meta=c.read(mp);d=load_data(meta);p=d['prediction'];h=d['history']
                pp=p.mean(2,keepdim=True);hh=h.mean(2,keepdim=True);center,radius,_=reference(hh)
                distance=torch.linalg.vector_norm(pp[0]-center,dim=-1).reshape(-1).double().numpy()
                rad=float(radius.reshape(-1)[0])
                ratio=distance/rad if rad>0 else np.full_like(distance,np.inf)
                # delta(alpha)=max(0,1-alpha/ratio)*(center-pp), shared over tokens.
                direction=(center-pp[0]).double()
                target_delta=(d['possible_target'].double()-d['impossible_target'].double()).mean(2,keepdim=True)[0]
                slope=2*(direction*target_delta).mean(tuple(range(1,direction.ndim))).numpy()
                base=np.asarray(meta['matched_identity_impossible_errors'])-np.asarray(meta['matched_identity_possible_errors'])
                rows_out.append(dict(key=row['key'],base=base.tolist(),slope=slope.tolist(),
                    ratio=[None if not np.isfinite(x) else float(x) for x in ratio],frame_spans=meta['frame_spans'],t_ref=int(row['t_ref']),
                    source_meta_sha256=c.digest(mp),source_state_sha256=meta['sha256']))
                if (index+1)%50==0:print('MSE PROPOSAL COEFFICIENTS',index+1,'/300; NN0',flush=True)
            data=dict(at=time.time(),rows=rows_out,neural_forwards=0,derivation='M_h(eta,alpha)=M_h(0)+eta*S_h*(1-alpha/q_h)_+. Actual float32 CUDA scores remain authoritative.')
            c.save(coeff,data)
    finally:guard.remove()
    rs=data['rows'];assert len(rs)==300
    base=np.asarray([r['base'] for r in rs]);scale=np.asarray([r['slope'] for r in rs]);ratio=np.asarray([[np.inf if x is None else x for x in r['ratio']] for r in rs])
    spans=np.asarray(rs[0]['frame_spans']);tref=np.asarray([r['t_ref'] for r in rs])
    alphas=sorted(set(np.linspace(1.,2.,257).tolist()+[float(x) for x in ratio.ravel() if 1<x<2]))
    all_proposals=[];max_counts=(0,0)
    for alpha in alphas:
        with np.errstate(divide='ignore',invalid='ignore'):factor=np.maximum(0,1-alpha/ratio)
        factor=np.where(ratio==0,0,factor);slope=scale*factor
        ps=intervals(base,slope,spans,tref,alpha,previous,best_actual)
        all_proposals.extend(ps)
        for x in ps:max_counts=max(max_counts,tuple(x['proposal_only_counts']))
    all_proposals.sort(key=lambda x:(-x['proposal_only_counts'][0],-x['proposal_only_counts'][1],-x['proposal_width'],x['config_id']))
    # Keep a bounded pool with distinct global radii; final Stage2 still max3.
    pool=[]
    for p in all_proposals:
        if any(abs(p['alpha']-x['alpha'])<.01 and abs(p['eta']-x['eta'])<.002 for x in pool):continue
        pool.append(p)
        if len(pool)==12:break
    c.save(c.HERE/'MSE_PROPOSAL_ANALYSIS_002.json',dict(at=time.time(),alpha_samples=len(alphas),native_best_counts=list(best_actual),
        candidate_pool=pool,ideal_only=True,no_global_continuous_optimum_claim=True,actual_scores_required=True,neural_forwards=0,
        float_warning='Uses float64 ideal arithmetic with native float32 intercepts; marginal/tie cases can disagree with actual CUDA.',
        coefficient_sha256=c.digest(coeff)))
    return pool
