"""Freeze a finite grid and identify all exact old settings, positives and negatives."""
import itertools
from pathlib import Path
import time
from . import common as c

def main():
    source=c.read(c.SOURCE);old=[];hashes={}
    for model in c.MODELS:
        for x in source['all_configurations'][model]:
            mode=x.get('mode','output_only' if model=='videomaev2' else 'feedback')
            if mode not in ('feedback','output_only','state_only','pre_final_block'):continue
            folder=x.get('source_directory') or x.get('directory')
            if not folder:
                from predictive_tracks.ppr_ie_eta_20261103_f.propose_slots import folder as resolve
                folder=str(resolve(model,x))
            row=dict(x,model=model,mode=mode,eta=x['parameters']['eta'],alpha=1.,directory=folder,origin='prior_M_radius1')
            old.append(row)
    restored=[]
    for model in c.MODELS:
        modes=('feedback','output_only') if model in c.SLOTS else ('output_only',)
        for mode,alpha in itertools.product(modes,(1.5,2.)):
            folder=c.ROOT/'predictive_tracks/ppr_alpha_20261031/runs/full_001'/model/mode/f'alpha_{round(alpha*100):03d}'
            s=c.read(folder/'summary.json');assert s['cases']==600
            row=dict(model=model,mode=mode,eta=1.,alpha=alpha,config_id='restored_'+c.setting_id(model,mode,1.,alpha),
                directory=str(folder),source_directory=str(folder),parameters=dict(eta=1.,alpha=alpha,q=None,presence=None,mass=None,min_history=None),
                metrics=s['metrics'],origin='restored_20261031_alpha_not_new',complete=True,completed_pairs=600,
                original_score_device='CPU_tensor_replay' if model=='videomaev2' else 'original_CUDA')
            restored.append(row)
            for p in [folder/'summary.json',folder/'config.json']+sorted((folder/'pairs').glob('*.json')):
                if p.exists():hashes[str(p)]=c.digest(p)
    old+=restored
    desired=[]
    for model in c.SLOTS:
        desired += [c.setting(model,mode,e,a) for mode,a,e in itertools.product(('feedback','output_only'),c.SLOT_ALPHAS,c.SLOT_ETAS)]
        desired += [c.setting(model,'state_only',e,a) for a,e in itertools.product(c.STATE_ALPHAS,c.STATE_ETAS)]
    desired += [c.setting('videomaev2','pre_final_block',.01,a) for a in (1.,2.)]
    new=[];reuse=[]
    for s in desired:
        matches=[x for x in old if (x['model'],x['mode'],x['eta'],x['alpha'])==(s['model'],s['mode'],s['eta'],s['alpha'])]
        if matches:
            assert all(x['metrics']==matches[0]['metrics'] for x in matches)
            reused=dict(s,source=matches[0]);reuse.append(reused)
        else:new.append(s)
    video=[c.setting('videomaev2','output_only',e,a) for a,e in itertools.product(c.VIDEO_ALPHAS,c.VIDEO_ETAS)]
    assert len(new)==114 and len(video)==126 and len(restored)==10
    c.save(c.HERE/'GRID.json',dict(at=time.time(),neural_new=new,neural_reuse=reuse,cached_video=video,
        restored=restored,source_report_sha256=c.digest(c.SOURCE),stage2_maximum_new_global_candidates_per_model=3))
    c.save(c.HERE/'RESTORED_HASHES.json',hashes)
    print('GRID',len(new),'new full-budget settings;',len(reuse),'exact settings reused;',len(video),'cached tensor settings;',len(restored),'restored controls',flush=True)

if __name__=='__main__':main()
