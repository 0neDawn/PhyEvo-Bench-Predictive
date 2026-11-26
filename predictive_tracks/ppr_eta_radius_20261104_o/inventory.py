"""Freeze exact points and identify complete identical historical settings."""
import itertools,sys,time
from pathlib import Path
from . import common as c
sys.path.insert(0,str(c.ROOT/'predictive_tracks/sp/code'))
def quality(p):return {k:p[k] for k in ('q','presence','mass','min_history')}
def signature(x):
    return (x['model'],x['mode'],float(x['eta']),float(x['alpha']),__import__('json').dumps(x.get('quality'),sort_keys=True))
def main():
    r=c.read(c.SOURCE);wanted=[];cache=[];prior=[]
    for m,rows in r['all_configurations'].items():
        for row in rows:
            p=row['parameters'];prior.append(dict(row,model=m,eta=p['eta'],alpha=p.get('alpha',1.),quality=quality(p) if m in c.OBJECTS else None))
    for m in c.CACHED:
        es=set(c.ETAS)
        if m=='slotformer':es.update((.875,.275))
        for task in ('sp','ie'):
            x=r['selected'][task][m]
            if x['mode']=='output_only' and x['parameters']['eta']>0:es.add(x['parameters']['eta'])
        cache += [c.job(m,'output_only',e,a) for a,e in itertools.product(c.ALPHAS,sorted(es))]
    for m in ('ocvp_seq','savi_transformer'):
        wanted += [c.job(m,'output_only',e,a) for e,a in itertools.product((.9,1.),(1.125,1.375,1.5))]
        mode=r['selected']['ie'][m]['mode']
        wanted += [c.job(m,mode,e,2.5) for e in (.1,.15,.25,.35,.5)]
        wanted += [c.job(m,mode,e,a) for a,e in itertools.product((3.,4.),(.15,.35,.65))]
    wanted += [c.job('slotformer','feedback',e,a) for e,a in itertools.product((.8,1.),(1.25,2.))]
    wanted += [c.job('slotformer','feedback',e,a) for e,a in itertools.product((.15,.35,.5),(1.25,2.,4.))]
    for m,task in itertools.product(c.OBJECTS,('sp','ie')):
        q=quality(r['selected'][task][m]['parameters'])
        wanted += [c.job(m,'feedback',1.,a,q,task) for a in (1.25,1.5,2.,3.,4.)]
        wanted.append(c.job(m,'feedback',.5,2.,q,task))
    for m in c.LARGE:
        wanted += [c.job(m,'pre_final_block',e,a) for e,a in ((.001,1.),(.003,2.),(.005,4.))]
    # These exact archived points predate N and remain read-only.
    for m,mode,a in itertools.product(('slotformer','ddlp','gswm'),('feedback','output_only'),(1.5,2.)):
        d=c.ROOT/'predictive_tracks/ppr_alpha_20261031/runs/full_001'/m/mode/f'alpha_{round(100*a):03d}'
        if not (d/'summary.json').exists():continue
        s=c.read(d/'summary.json');assert s['cases']==600
        p=dict(eta=1.,alpha=a,q=.5 if m in c.OBJECTS else None,presence=.5 if m in c.OBJECTS else None,mass=4. if m in c.OBJECTS else None,min_history=2 if m in c.OBJECTS else None)
        prior.append(dict(model=m,mode=mode,eta=1.,alpha=a,quality=quality(p) if m in c.OBJECTS else None,
            config_id=f'archived_alpha_{m}_{mode}_{a}',parameters=p,metrics=s['metrics'],complete=True,completed_pairs=600,
            directory=str(d),source_directory=str(d),origin='archived_20261031_not_new'))
    new=[];reuse=[];hashes={}
    for j in wanted:
        matches=[x for x in prior if signature(x)==signature(j)]
        if not matches:new.append(j);continue
        assert all(x['metrics']==matches[0]['metrics'] for x in matches)
        source=matches[0];d=source.get('directory') or source.get('source_directory')
        if not d:
            from predictive_tracks.ppr_ie_eta_20261103_f.propose_slots import folder
            d=str(folder(j['model'],source))
        source=dict(source,directory=d,source_directory=d);reuse.append(dict(j,source=source))
        for p in [Path(d)/'summary.json']+sorted((Path(d)/'pairs').glob('*.json')):
            hashes[str(p)]=c.digest(p)
    old_keys={signature(x) for x in prior}
    for j in cache:j['previous_parameter_point']=signature(j) in old_keys
    assert len(cache)==950 and len(wanted)==77 and len(new)==72 and len(reuse)==5,(len(cache),len(wanted),len(new),len(reuse))
    c.save(c.HERE/'GRID.json',dict(at=time.time(),native_new=new,native_reuse=reuse,cached=cache,
        independent_pairs=600,all_settings_full600=True,source_report_sha256=c.digest(c.SOURCE),stage2_max_per_model=2))
    c.save(c.HERE/'REUSED_HASHES.json',hashes)
    print('FROZEN GRID:',len(cache),'cached +',len(new),'new native;',len(reuse),'exact reuse',flush=True)
if __name__=='__main__':main()
