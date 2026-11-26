"""At most three global eta proposals/model, then real full600 validation."""
from collections import defaultdict
import time
import numpy as np
from scipy.interpolate import PchipInterpolator
from . import common as c
from .report import assemble,records
from predictive_tracks.ppr_ie_eta_20261103_f.loss_curves import decisions

def main():
    if (c.HERE/'STAGE2.json').exists():
        assert (c.HERE/'STAGE2_FROZEN.json').exists();return
    r=assemble();grid=c.read(c.HERE/'GRID.json');assert len(r['current_round_settings'])==240
    neural=[];video=[];diagnostics={}
    for model in c.MODELS:
        # Only the current fixed grid and exact reused controls are curve knots.
        allj=grid['neural_new']+[dict(x,source=x['source']) for x in grid['neural_reuse']]+grid['cached_video']
        groups=defaultdict(list)
        for j in allj:
            if j['model']!=model:continue
            groups[(j['mode'],j['alpha'])].append(j)
        chosen=r['selected']['ie'][model]
        best_actual=(round(300*chosen['metrics']['ie']['corrected']['overall']['joint_hit']['2' if model=='videomaev2' else '1']),
            round(300*chosen['metrics']['ie']['corrected']['overall']['paired_accuracy']))
        proposals=[];notes=[]
        for (mode,alpha),js in sorted(groups.items()):
            js=sorted(js,key=lambda j:j['eta'])
            if len(js)<3:continue
            values=[];previous=[];base=None;spans=None;tref=None
            for j in js:
                if j.get('source'):setting=j['source']
                else:
                    setting=next(x for x in r['current_round_settings'] if x['config_id']==j['config_id'])
                rs=sorted((x for x in records(setting) if x['benchmark']=='ie'),key=lambda x:x['archive_index'])
                assert len(rs)==300
                raw=np.asarray([x['matched_identity_impossible_errors'] for x in rs])-np.asarray([x['matched_identity_possible_errors'] for x in rs])
                if base is None:
                    base=raw;spans=np.asarray(rs[0]['frame_spans']);tref=np.asarray([x['t_ref'] for x in rs])
                else:assert np.array_equal(base,raw)
                values.append(np.asarray([x['impossible_errors'] for x in rs])-np.asarray([x['possible_errors'] for x in rs]));previous.append(j['eta'])
            etas=np.asarray([0.]+previous);curve=PchipInterpolator(etas,np.concatenate([base[None],np.asarray(values)]),axis=0)
            sample=np.linspace(0,1,8193)
            hit=[];pa=[]
            for start in range(0,len(sample),256):
                q=curve(sample[start:start+256]);k=len(q)
                aa,bb,_=decisions(q.transpose(2,1,0).reshape(q.shape[2],-1),spans,np.repeat(tref,k),2 if model=='videomaev2' else 1)
                pa.extend(aa.reshape(300,k).sum(0).tolist());hit.extend(bb.reshape(300,k).sum(0).tolist())
            hit=np.asarray(hit);pa=np.asarray(pa)
            edges=np.r_[0,np.flatnonzero((np.diff(hit)!=0)|(np.diff(pa)!=0))+1,len(sample)]
            count=0
            for first,last in zip(edges,edges[1:]):
                ideal=(int(hit[first]),int(pa[first]));eta=round(float((sample[first]+sample[last-1])/2),6)
                if ideal<=best_actual or eta in previous or eta<=0:continue
                s=c.setting(model,mode,eta,alpha)
                s.update(proposal_only_counts=list(ideal),proposal_interval=[float(sample[first]),float(sample[last-1])],
                    proposal_width=float(sample[last-1]-sample[first]),proposal_kind='PCHIP of saved IE error differences; not a result')
                proposals.append(s);count+=1
            notes.append(dict(mode=mode,alpha=alpha,knots=previous,max_surrogate_JH=int(hit.max()),max_surrogate_PA=int(pa.max()),useful_intervals=count))
        proposals.sort(key=lambda x:(-x['proposal_only_counts'][0],-x['proposal_only_counts'][1],-x['proposal_width'],x['config_id']))
        picks=[]
        for s in proposals:
            if s['config_id'] not in {x['config_id'] for x in picks}:picks.append(s)
            if len(picks)==3:break
        for s in picks:(video if model=='videomaev2' and s['mode']=='output_only' else neural).append(s)
        diagnostics[model]=dict(actual_best_counts=list(best_actual),groups=notes,selected_proposals=picks,
            limitation='Surrogate predicts candidate eta only; actual native nonlinear rollout / CUDA scoring remains mandatory.')
        print('BOUNDED REFINEMENT',model,[(x['mode'],x['alpha'],x['eta']) for x in picks],flush=True)
    c.save(c.HERE/'STAGE2.json',dict(at=time.time(),neural_new=neural,cached_video=video,diagnostics=diagnostics,
        maximum_new_global_candidates_per_model=3,stage1_complete=True,scope='Exposed-data adaptive proposals. Not actual performance.'))
    c.save(c.HERE/'STAGE2_FROZEN.json',dict(at=time.time(),files={str(c.HERE/'STAGE2.json'):c.digest(c.HERE/'STAGE2.json')},
        proposal_code_sha256=c.digest(c.HERE/'refine.py')))

if __name__=='__main__':main()
