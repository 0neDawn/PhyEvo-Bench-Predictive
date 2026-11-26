"""Surrogates propose only; actual original scoring is always required."""
from collections import defaultdict
import json,time
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.stats import rankdata
from . import common as c
from .report import assemble,records
from predictive_tracks.controlled_tracks.registry import NATIVE_TOLERANCE
from predictive_tracks.ppr_ie_eta_20261103_f.loss_curves import decisions

def counts_key(x,t):
    v=x['metrics'][t]['corrected']['overall'];tol=str(NATIVE_TOLERANCE[x['model']]);mae=v['localization_mae']
    return (round(v['joint_hit'][tol]*300),round(v['paired_accuracy']*300),round(v['auc']*180000),-(mae if mae is not None else 1e30))
def fast_auc(p,i):
    d=p+i+1e-12;sc=np.concatenate((i/d,p/d),axis=1);ranks=rankdata(sc,axis=1,method='average');n=p.shape[1]
    return (ranks[:,:n].sum(1)-n*(n+1)/2)/(n*n)
def qkey(x):
    if x['model'] not in c.OBJECTS:return None
    return json.dumps({k:x['parameters'].get(k) for k in ('q','presence','mass','min_history')},sort_keys=True)
def main():
    if (c.HERE/'STAGE2.json').exists():assert (c.HERE/'STAGE2_FROZEN.json').exists();return
    r=assemble();assert r['completed_round_configurations']==1022
    group_jobs=defaultdict(list)
    for j in c.read(c.HERE/'GRID.json')['native_new']+c.read(c.HERE/'GRID.json')['cached']:
        group_jobs[(j['model'],j['mode'],j['alpha'],json.dumps(j['quality'],sort_keys=True),j['target_task'])].append(j)
    neural=[];cached=[];diagnostics={}
    for m in tuple(c.CACHED)+tuple(x for x in c.NATIVE if x not in c.CACHED):
        proposals={'sp':[],'ie':[]};notes=[]
        for (model,mode,alpha,quality_s,target),js in group_jobs.items():
            if model!=m:continue
            quality=json.loads(quality_s);keyq=json.dumps(quality,sort_keys=True) if quality is not None else None
            rows=[x for x in r['all_configurations'][m] if x.get('mode')==mode and x['parameters'].get('alpha',1.)==alpha and qkey(x)==keyq]
            by_eta={}
            # Prefer this round's declared uniform numeric path as curve knots;
            # confirmations still cannot become primary result rows.
            for x in rows:by_eta[float(x['parameters']['eta'])]=x
            if not by_eta:continue
            known=sorted(by_eta);loaded={e:sorted(records(by_eta[e]),key=lambda x:(x['benchmark'],x['archive_index'])) for e in known}
            group_max={}
            for task in ('sp','ie'):
                if target and target!=task:continue
                rr={e:[x for x in loaded[e] if x['benchmark']==task] for e in known};base=rr[known[0]];assert len(base)==300
                baseline=np.asarray([[x['matched_identity_possible_errors'],x['matched_identity_impossible_errors']] for x in base])
                values=np.asarray([[[x['possible_errors'],x['impossible_errors']] for x in rr[e]] for e in known])
                etas=np.asarray([0.]+known);curve=PchipInterpolator(etas,np.concatenate((baseline[None],values)),axis=0)
                sample=np.linspace(0,float(known[-1]),4097);best=counts_key(dict(r['selected'][task][m],model=m),task)
                tref=np.asarray([x['t_ref'] for x in base]);spans=np.asarray(base[0]['frame_spans']);cand=[];max_key=best
                for start in range(0,len(sample),128):
                    es=sample[start:start+128];v=curve(es);k=len(es)
                    pp=v[:,:,0,:];ii=v[:,:,1,:];margin=ii-pp
                    pa,hit,loc=decisions(margin.transpose(2,1,0).reshape(margin.shape[2],-1),spans,np.repeat(tref,k),NATIVE_TOLERANCE[m])
                    pa=pa.reshape(300,k).T;hit=hit.reshape(300,k).T;loc=loc.reshape(300,k).T
                    auc=fast_auc(pp.mean(2),ii.mean(2));valid=pa&(loc>=0);dist=np.abs(loc-tref[None]);n=valid.sum(1)
                    mae=np.where(n>0,(dist*valid).sum(1)/np.maximum(n,1),1e30)
                    for idx,e in enumerate(es):
                        key=(int(hit[idx].sum()),int(pa[idx].sum()),round(auc[idx]*180000),-float(mae[idx]));max_key=max(max_key,key)
                        rounded=round(float(e),6)
                        if key<=best or rounded<=0 or rounded in known or rounded>known[-1]:continue
                        j=c.job(m,mode,rounded,alpha,quality,target)
                        # Do not count source-expression confirmation as a new point.
                        if any(abs(rounded-y['parameters']['eta'])<1e-12 for y in rows):continue
                        j.update(proposal_task=task,proposal_rank=list(key),proposal_baseline_rank=list(best),
                            proposal_only=True,proposal_kind='PCHIP original error arrays; full original-metric rank; actual validation mandatory')
                        cand.append(j)
                cand.sort(key=lambda j:(tuple(-z for z in j['proposal_rank']),j['config_id']))
                proposals[task].extend(cand[:6]);group_max[task]=list(max_key)
            notes.append(dict(mode=mode,alpha=alpha,quality=quality,knots=known,surrogate_max=group_max))
        picks=[]
        for task in ('sp','ie'):
            ps=proposals[task];ps.sort(key=lambda j:(tuple(-z for z in j['proposal_rank']),j['config_id']))
            for j in ps:
                if j['config_id'] not in {x['config_id'] for x in picks}:picks.append(j);break
        assert len(picks)<=2
        for j in picks:(cached if m in c.CACHED and j['mode']=='output_only' else neural).append(j)
        diagnostics[m]=dict(groups=notes,candidates=picks)
        print('BOUNDED PROPOSALS',m,[(j['proposal_task'],j['eta'],j['alpha'],j['mode']) for j in picks],flush=True)
    c.save(c.HERE/'STAGE2.json',dict(at=time.time(),native_new=neural,cached=cached,diagnostics=diagnostics,maximum_two_per_model=True,
        scope='Surrogate proposal only; all actual full600 scores required. No inference/model/distance change.'))
    c.save(c.HERE/'STAGE2_FROZEN.json',dict(at=time.time(),files={str(c.HERE/'STAGE2.json'):c.digest(c.HERE/'STAGE2.json')},proposal_source_sha256=c.digest(c.HERE/'refine.py')))
if __name__=='__main__':main()
