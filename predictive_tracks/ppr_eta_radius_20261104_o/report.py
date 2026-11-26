"""Original metrics, fixed task-level selection, all settings and diagnostics."""
import copy,functools,inspect,json,sys,time
from pathlib import Path
from . import common as c
sys.path.insert(0,str(c.ROOT/'predictive_tracks/sp/code'))
import numpy as np
from predictive_tracks.ppr_taskwise_best_20261102_001.select_report import rank,endpoint,ENDPOINTS,ORDER
from predictive_tracks.ppr_existing_20261102.metrics import make_summary
from predictive_tracks.controlled_tracks.registry import NATIVE_TOLERANCE,OUTPUT

def all_jobs():
    g=c.read(c.HERE/'GRID.json');s=c.read(c.HERE/'STAGE2.json') if (c.HERE/'STAGE2.json').exists() else dict(native_new=[],cached=[])
    return g['native_new']+g['cached']+s['native_new']+s['cached']
def parameters(j):return dict(eta=j['eta'],alpha=j['alpha'],**(j['quality'] or dict(q=None,presence=None,mass=None,min_history=None)))
def new_rows():
    out=[]
    for j in all_jobs():
        d=c.directory(j);p=d/'summary.json'
        if not p.exists():continue
        s=c.read(p);assert s['complete'] and s['cases']==600
        out.append(dict(j,parameters=parameters(j),metrics=s['metrics'],complete=True,completed_pairs=600,directory=str(d),source_directory=str(d),
            compact_case_source=s.get('compact_case_source'),summary_sha256=c.digest(p),origin='O_parameter_confirmation' if j.get('previous_parameter_point') else 'O_new_parameter_point',
            eligible_for_selection=not j.get('previous_parameter_point',False),eligible_tasks=[j['target_task']] if j['target_task'] else ['sp','ie']))
    return out
def native(m,metrics):
    return dict(model=m,mode='native',config_id='native_eta0',parameters=dict(eta=0.,alpha=1.,q=None,presence=None,mass=None,min_history=None),
        metrics={t:dict(native_baseline=metrics[t]['native_baseline'],corrected=metrics[t]['native_baseline']) for t in ('sp','ie')})
def assemble():
    old=c.read(c.SOURCE);rows=new_rows();g=c.read(c.HERE/'GRID.json');allrows=copy.deepcopy(old['all_configurations'])
    included={x['config_id'] for rs in allrows.values() for x in rs}
    restored=[]
    for j in g['native_reuse']:
        x=copy.deepcopy(j['source'])
        if x['config_id'] in included:continue
        x.update(eligible_tasks=[j['target_task']] if j['target_task'] else ['sp','ie'],eligible_for_selection=True)
        allrows[x['model']].append(x);restored.append(x);included.add(x['config_id'])
    for x in rows:allrows[x['model']].append(x)
    r=dict(at=time.time(),source_report=str(c.SOURCE),source_report_sha256=c.digest(c.SOURCE),all_configurations=allrows,
        selected={t:{} for t in ('sp','ie')},metric_specific_maxima={t:{} for t in ('sp','ie')},current_round_settings=rows,
        restored_matching_historical_settings=restored,completed_round_configurations=len(rows),planned_stage1=1022,
        completed_corrected_model_configs=sum(map(len,allrows.values())),radius_multiplier_search=[1,4],extra_neural_forwards_per_forecast=0,
        selection=old['selection'],scope='Same exposed SP300/IE300, adaptive eta/radius exploration; no heldout or continuous optimum claim.',
        dino=old.get('dino','Original timing issue retained; diagnostic only.'),
        current_round_note='950 cache replays include prior-parameter confirmations; these duplicate points are retained but not eligible as new main-table discoveries.',
        previous_selected=old['selected'],change_attribution_o={t:{} for t in ('sp','ie')})
    for m in ORDER:
        rs=allrows[m];b=native(m,rs[0]['metrics'])
        for t in ('sp','ie'):
            candidates=[x for x in rs if x.get('eligible_for_selection',True) and t in x.get('eligible_tasks',['sp','ie'])]+[b]
            for x in candidates:assert x['metrics'][t]['native_baseline']==b['metrics'][t]['native_baseline']
            best=min(candidates,key=lambda x:rank(x,t));r['selected'][t][m]=copy.deepcopy(best);previous=old['selected'][t][m]
            r['change_attribution_o'][t][m]=dict(origin=best.get('origin','native'),old_id=previous['config_id'],new_id=best['config_id'],
                differences_vs_N={k:endpoint(best,t,k)-endpoint(previous,t,k) for k in ENDPOINTS if endpoint(best,t,k) is not None and endpoint(previous,t,k) is not None})
            r['metric_specific_maxima'][t][m]={}
            for k in ENDPOINTS:
                x=min(candidates,key=lambda x:rank(x,t,k));bv=endpoint(b,t,k);v=endpoint(x,t,k)
                r['metric_specific_maxima'][t][m][k]=dict(config_id=x['config_id'],mode=x['mode'],parameters=x['parameters'],baseline=bv,best_value=v,
                    change=None if v is None or bv is None else v-bv,all_metrics_at_this_setting=x['metrics'][t])
    return r
@functools.lru_cache(maxsize=1)
def payloads(folder):return [c.read(p) for p in sorted(Path(folder).glob('*.json'))]
def records(x):
    if x.get('compact_case_source'):
        from .cache_worker import expand
        return [expand(p,x) for p in payloads(x['compact_case_source'])]
    d=x.get('directory') or x.get('source_directory')
    if not d:
        from predictive_tracks.ppr_ie_eta_20261103_f.propose_slots import folder
        d=folder(x['model'],x)
    return [c.read(p) for p in sorted((Path(d)/'pairs').glob('*.json'))]
def interim():
    r=assemble();d=c.HERE/'reports'/str(time.time_ns());c.save(d/'report.json',r)
    c.status(c.HERE/'monitor/latest_report.json',dict(at=time.time(),path=str(d),completed=r['completed_round_configurations']))
    print('COMPLETE SETTINGS',r['completed_round_configurations'],'/1022 Stage1',flush=True)
    for m in ORDER:
        x=r['selected']['ie'][m];print('IE BEST',m,x['parameters'],'JH',endpoint(x,'ie','overall_joint_hit'),flush=True)
    return r
def casewise(rs,m):
    from sp_benchmark.protocol_metrics import summarize
    from predictive_tracks.ppr_position_controls_20261103_k.casewise_audit_001 import flags
    from predictive_tracks.controlled_tracks.report import paired_stats
    out={};tol=NATIVE_TOLERANCE[m]
    for t in ('sp','ie'):
        rr=sorted((x for x in rs if x['benchmark']==t),key=lambda x:x['archive_index'])
        def calc(prefix):return summarize(np.asarray([x[prefix+'possible_errors'] for x in rr]),np.asarray([x[prefix+'impossible_errors'] for x in rr]),frame_spans=rr[0]['frame_spans'],violation_frames=[x['t_ref'] for x in rr],epsilon=1e-12,threshold=0.,persistence_frames=2,joint_tolerances=(tol,))
        bp,bm,bj,be=flags(calc('matched_identity_'),tol);ap,am,aj,ae=flags(calc(''),tol);common=bm&am
        ids=lambda f:[f'{t}_{rr[i]["archive_index"]:03d}' for i in np.flatnonzero(f)]
        out[t]=dict(PA_gain=ids(ap&~bp),PA_loss=ids(~ap&bp),JH_gain=ids(aj&~bj),JH_loss=ids(~aj&bj),native_MAE_n=int(bm.sum()),corrected_MAE_n=int(am.sum()),
            common_MAE_n=int(common.sum()),common_native_MAE=float(be[common].mean()) if common.any() else None,common_corrected_MAE=float(ae[common].mean()) if common.any() else None,
            onset_improved=ids(common&(ae<be)),onset_worsened=ids(common&(ae>be)),statistics={k:paired_stats(a,b,[x['category'] for x in rr],seed=20261104,repetitions=10000) for k,a,b in (('PA',ap,bp),('JH',aj,bj))})
    return out
def render(r,d):
    from predictive_tracks.ppr_eta_radius_20261104_n import report as previous
    source=inspect.getsource(previous.render);ns=dict(vars(previous),c=c)
    with (d/'renderer_source.txt').open('x') as f:f.write(source)
    exec(compile(source,'<O_original_N_renderer>','exec'),ns);ns['render'](r,d)
def finish():
    assert (c.HERE/'MODEL_RUNS_COMPLETE.json').exists();r=assemble();new=r['current_round_settings'];assert len(new)==len(all_jobs())
    d=c.HERE/'delivery_20261104_001';d.mkdir(exist_ok=False)
    from predictive_tracks.ppr_20261028.run import rows_and_references
    originals={m:{row['key']:(row,p,ref) for row,p,ref in rows_and_references(m)} for m in c.NATIVE}
    traces=c.read(OUTPUT/'deployment_audit_001.json')['details'];shared={}
    for m in c.NATIVE:
        ts={json.dumps(x['neural_trace'],sort_keys=True) for x in traces[m]};shared[m]=json.loads(next(iter(ts))) if len(ts)==1 else None
    hashes={};details=[];tests=[]
    for x in new:
        rr=records(x);s=make_summary(rr,x['model']);assert s['metrics']==x['metrics']
        for rec in rr:
            assert rec['eta']==x['eta'] and rec['radius_multiplier']==x['alpha']
            key=f'{rec["benchmark"]}_{rec["archive_index"]:03d}'
            if x.get('compact_case_source'):
                p=Path(x['compact_case_source'])/f'{key}.json'
                if str(p) not in hashes:
                    v=c.read(p);assert v['native_max_abs_error']==0 and v['neural_forwards']==0
                    assert c.digest(v['native_meta'])==v['source_meta_sha256'];meta=c.read(v['native_meta']);assert c.digest(meta['state'])==v['source_state_sha256']
                    hashes[str(p)]=c.digest(p)
            else:
                row,rp,ref=originals[x['model']][key]
                assert rec['pair_id']==row['pair_id'] and rec['frame_spans']==ref['frame_spans'] and rec['t_ref']==int(row['t_ref'])
                assert rec['reference_pair_sha256']==c.digest(rp) and rec['source_decoded_pixels_sha256']==ref['source_decoded_pixels_sha256']
                assert rec['trace_matches_original'] and rec['extra_forwards']==0 and rec['weight_identity_and_version_unchanged']
                expected=ref.get('neural_call_trace',shared[x['model']])
                if expected is None and row['split']=='development':expected=c.read(OUTPUT/f'{x["model"]}_boundary_dev_001/pairs'/f'{key}.json')['neural_call_trace']
                assert rec['neural_call_trace']==expected
                p=Path(x['directory'])/'pairs'/f'{key}.json';hashes[str(p)]=c.digest(p)
        details.append(dict(config_id=x['config_id'],cases=600,original_metrics_exact=True))
        for t,dd in s['exact_paired_tests'].items():
            for k,test in dd.items():tests.append(dict(test,task=t,endpoint=k,model=x['model'],config_id=x['config_id']))
        if len(details)%25==0:print('FINAL AUDIT',len(details),'/',len(new),flush=True)
    for p,w in c.read(c.HERE/'REUSED_HASHES.json').items():assert c.digest(p)==w
    for j in c.read(c.HERE/'GRID.json')['native_reuse']:
        s=make_summary(records(j['source']),j['model']);assert s['metrics']==j['source']['metrics']
    audits=[]
    for p in (c.HERE/'audit').glob('*/*/summary.json'):
        a=c.read(p);assert a['audit_passed'] and a['cases']==6 and a['no_op_max_abs_error']==0;audits.append(str(p));hashes[str(p)]=c.digest(p)
    stats={}
    for t in ('sp','ie'):
        for m,x in r['selected'][t].items():
            if x['config_id']=='native_eta0':continue
            k=m+':'+x['config_id']
            if k not in stats:stats[k]=casewise(records(x),m)
    r['exact_paired_tests_current_round']=tests
    c.save(d/'report.json',r);c.save(d/'COMPLETENESS_AUDIT.json',dict(at=time.time(),settings=details,configurations=len(new),case_records=600*len(new),independent_pairs=600,no_op_audits=audits,all_original_metrics_recomputed=True))
    c.save(d/'CASEWISE_AND_PAIRED_STATISTICS.json',stats);c.save(d/'input_hashes.json',hashes);c.save(d/'RESOURCE_PEAKS.json',c.read(c.HERE/'monitor/resource_peaks.json'))
    render(r,d)
    with (d/'all_settings.tsv').open('x') as f:
        f.write('model\ttask\tmode\teta\talpha\tquality\tPA_percent\tAUC_percent\tMAE\tJH_percent\teligible\torigin\tid\n')
        for m,rs in r['all_configurations'].items():
            for x in rs:
                x=dict(x,model=m)
                for t in ('sp','ie'):
                    pp=x['parameters'];eligible=x.get('eligible_for_selection',True) and t in x.get('eligible_tasks',['sp','ie'])
                    values=[m,t,x.get('mode'),pp['eta'],pp.get('alpha',1.),json.dumps({k:pp.get(k) for k in ('q','presence','mass','min_history')}),100*endpoint(x,t,'paired_accuracy'),100*endpoint(x,t,'auc'),endpoint(x,t,'localization_mae'),100*endpoint(x,t,'overall_joint_hit'),eligible,x.get('origin'),x['config_id']]
                    f.write('\t'.join(map(str,values))+'\n')
    with (d/'READ_ME.txt').open('x') as f:f.write('All settings use original SP300+IE300. One actual setting per row; all metrics/declines retained.\nOnly eta and radius multiplier changed; Euclidean norm,history,weights,native call budget,loss and data unchanged.\nOriginal DINO timing problem retained: diagnostic only. Cached native eta0 reproduced exactly; neural forwards forbidden.\nObject reference-quality constants fixed to the task-specific previous selected basis; cross-task scores retained but not used for a new quality search.\nDuplicate cached parameter points are confirmations, not eligible new main-table discoveries. Arithmetic expressions/devices fixed globally.\nSame-data adaptive selection, not independent validation. Projection bounds do not imply physical interactions or metrics improve. Conditional MAE needs common-member checks.\n')
    c.verify(full=True)
    c.save(d/'DELIVERY.json',dict(at=time.time(),files={str(p):c.digest(p) for p in d.iterdir() if p.is_file()},all_checks_passed=True))
    c.save(c.HERE/'FINAL_DELIVERY.json',dict(at=time.time(),directory=str(d),complete=True));print('FINAL DELIVERY',d,flush=True)
if __name__=='__main__':finish()
