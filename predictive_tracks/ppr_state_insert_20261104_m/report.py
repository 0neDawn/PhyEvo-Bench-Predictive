"""Original metrics, transparent all-setting tables, and independent final audit."""
import copy
import html
import inspect
from pathlib import Path
import sys
import time
from . import common as c
sys.path.insert(0,str(c.ROOT/'predictive_tracks/sp/code'))
import numpy as np
from predictive_tracks.ppr_taskwise_best_20261102_001.select_report import rank,endpoint,ENDPOINTS,NAMES,ORDER,TYPES
from predictive_tracks.ppr_existing_20261102.metrics import make_summary

def new_rows():
    rows=[]
    for model in c.MODELS:
        for eta in c.ETAS:
            d=c.directory(model,eta);p=d/'summary.json'
            if not p.exists():continue
            s=c.read(p)
            assert s['complete'] and s['cases']==600
            rows.append(dict(model=model,mode=c.mode(model),config_id=f'insert_m_{model}_{round(eta*1000):04d}',
                parameters=dict(eta=eta,q=None,presence=None,mass=None,min_history=None),complete=True,completed_pairs=600,
                directory=str(d),source_directory=str(d),metrics=s['metrics'],summary_sha256=c.digest(p)))
    return rows

def native(model,metrics):
    return dict(model=model,mode='native',config_id='native_eta0',parameters=dict(eta=0.,q=None,presence=None,mass=None,min_history=None),
        metrics={t:dict(native_baseline=metrics[t]['native_baseline'],corrected=metrics[t]['native_baseline']) for t in ('sp','ie')})

def interim():
    rows=new_rows();out=c.HERE/'reports'/str(time.time_ns())
    c.save(out/'new_completed.json',dict(at=time.time(),completed_settings=len(rows),expected_settings=12,rows=rows))
    c.status(c.HERE/'monitor/latest_report.json',dict(at=time.time(),path=str(out)))
    print('REPORT',len(rows),'/12 full settings',flush=True)

def casewise(records,model):
    from predictive_tracks.ppr_position_controls_20261103_k.casewise_audit_001 import summarize_records,flags
    from predictive_tracks.controlled_tracks.report import paired_stats
    result={}
    for task in ('sp','ie'):
        rs=sorted((r for r in records if r['benchmark']==task),key=lambda x:x['archive_index'])
        b=summarize_records(rs,model,'matched_identity_');a=summarize_records(rs,model)
        tol=2 if model not in c.SLOTS else 1
        bp,bm,bj,be=flags(b,tol);ap,am,aj,ae=flags(a,tol);common=bm&am
        ids=lambda mask:[f'{task}_{rs[i]["archive_index"]:03d}' for i in np.flatnonzero(mask)]
        result[task]=dict(paired_correct_gain=ids(ap&~bp),paired_correct_loss=ids(~ap&bp),
            joint_hit_gain=ids(aj&~bj),joint_hit_loss=ids(~aj&bj),
            same_PA_membership=bool(np.array_equal(bp,ap)),same_MAE_membership=bool(np.array_equal(bm,am)),
            native_MAE_n=int(bm.sum()),corrected_MAE_n=int(am.sum()),common_MAE_n=int(common.sum()),
            common_native_MAE=float(be[common].mean()) if common.any() else None,
            common_corrected_MAE=float(ae[common].mean()) if common.any() else None,
            common_onset_improved=ids(common&(ae<be)),common_onset_worsened=ids(common&(ae>be)),
            paired_statistics={key:paired_stats(now,before,[r['category'] for r in rs],seed=20261104,repetitions=10000)
                for key,now,before in (('paired_accuracy',ap,bp),('joint_hit',aj,bj))})
    return result

def audit(rows):
    from predictive_tracks.ppr_20261028.run import rows_and_references
    from predictive_tracks.controlled_tracks.registry import OUTPUT
    original={m:{r['key']:(r,p,ref) for r,p,ref in rows_and_references(m)} for m in c.MODELS}
    traces=c.read(OUTPUT/'deployment_audit_001.json')['details']
    hashes={};tests=[];cases={};details=[]
    for setting in rows:
        model=setting['model'];d=Path(setting['directory']);files=sorted((d/'pairs').glob('*.json'));assert len(files)==600
        records=[];ts={__import__('json').dumps(x['neural_trace'],sort_keys=True) for x in traces[model]}
        common=__import__('json').loads(next(iter(ts))) if len(ts)==1 else None
        for p in files:
            r=c.read(p);row,ref_path,ref=original[model][p.stem];hashes[str(p)]=c.digest(p)
            assert r['benchmark']==row['benchmark'] and r['pair_id']==row['pair_id'] and r['archive_index']==row['archive_index']
            assert r['frame_spans']==ref['frame_spans'] and r['t_ref']==int(row['t_ref'])
            assert r['reference_pair_sha256']==c.digest(ref_path) and r['source_decoded_pixels_sha256']==ref['source_decoded_pixels_sha256']
            assert r['matched_identity_possible_errors']==ref['matched_identity_possible_errors']
            assert r['matched_identity_impossible_errors']==ref['matched_identity_impossible_errors']
            expected=ref.get('neural_call_trace',common)
            if expected is None and row['split']=='development':
                expected=c.read(OUTPUT/f'{model}_boundary_dev_001/pairs'/f'{row["key"]}.json')['neural_call_trace']
            assert r['neural_call_trace']==expected and r['trace_matches_original'] and r['extra_forwards']==0
            assert r['eta']==setting['parameters']['eta'] and r['mode']==c.mode(model) and r['radius_multiplier']==1
            assert r['weight_identity_and_version_unchanged']
            assert len(r['projection_audits'])==1 and len(r['projection_audits'][0]['steps'])==(6 if model in c.SLOTS else 3)
            if model in c.SLOTS:
                anchor=c.read(c.HERE/'anchors'/model/(p.stem+'.json'))
                assert anchor['first_raw_prediction_sha256']==r['first_raw_prediction_sha256']
            else:
                assert r['active_hook_calls']==1 and r['inactive_target_hook_calls']==(6 if model=='videomaev2' else 0)
                assert r['projection_audits'][0]['history_steps']==4 and r['projection_audits'][0]['context_unchanged']
            records.append(r)
        s=make_summary(records,model);assert s['metrics']==setting['metrics']
        cases[setting['config_id']]=casewise(records,model)
        for task in ('sp','ie'):
            for key,test in s['exact_paired_tests'][task].items():
                tests.append(dict(test,model=model,benchmark=task,endpoint=key,config_id=setting['config_id'],mode=c.mode(model)))
        details.append(dict(model=model,eta=setting['parameters']['eta'],records=600,original_metrics_exact=True,native_budget_exact=True))
        print('AUDITED',setting['config_id'],flush=True)
    audits={}
    for model in c.MODELS:
        ds=c.directory(model,0.);s=c.read(ds/'summary.json');assert s['audit_passed'] and s['cases']==6
        paths=sorted((ds/'pairs').glob('*.json'));assert len(paths)==6
        for p in paths:hashes[str(p)]=c.digest(p)
        audits[model]=s
    return dict(at=time.time(),full_settings=len(rows),case_records=600*len(rows),independent_pairs=600,
        per_task_pairs=300,per_type_pairs=100,details=details,no_op_audits=audits,all_passed=True),tests,cases,hashes

def assemble(rows,tests):
    old=c.read(c.SOURCE);r=copy.deepcopy(old)
    r.update(at=time.time(),source_report=str(c.SOURCE),source_report_sha256=c.digest(c.SOURCE),
        scope='New author-authorized state-only/pre-final-block positions;12 full settings on previously exposed SP300/IE300. All old666 configurations retained. No independent validation or continuous optimum claim.',
        all_new_position_settings=rows,new_completed=len(rows),expected_new=12,
        completed_corrected_model_configs=old['completed_corrected_model_configs']+len(rows),
        extra_neural_forwards_per_forecast=0,radius_multiplier_fixed=1,
        previous_selected=copy.deepcopy(old['selected']),new_only_selected={'sp':{},'ie':{}})
    for x in rows:r['all_configurations'][x['model']].append(x)
    r['metric_specific_maxima']={t:{} for t in ('sp','ie')}
    for model in ORDER:
        default='feedback' if model in ('ocvp_seq','savi_transformer','ddlp','gswm','slotformer') else 'output_only'
        candidates=[dict(copy.deepcopy(x),model=model,mode=x.get('mode',default)) for x in r['all_configurations'][model]]
        base=native(model,candidates[0]['metrics']);candidates.append(base)
        for task in ('sp','ie'):
            r['selected'][task][model]=min(candidates,key=lambda x:rank(x,task))
            if model in c.MODELS:
                nrows=[x for x in rows if x['model']==model]
                r['new_only_selected'][task][model]=min(nrows+[base],key=lambda x:rank(x,task))
            r['metric_specific_maxima'][task][model]={}
            for key in ENDPOINTS:
                best=min(candidates,key=lambda x:rank(x,task,key));v=endpoint(best,task,key);b=endpoint(base,task,key)
                o=best['metrics'][task]['corrected']['overall']
                r['metric_specific_maxima'][task][model][key]=dict(parameters=best['parameters'],mode=best['mode'],config_id=best['config_id'],
                    baseline=b,best_value=v,change=None if v is None or b is None else v-b,coverage=o['detection_coverage'],
                    localization_MAE_denominator=o['localization_mae_denominator'],all_metrics_at_this_setting=best['metrics'][task])
    alltests=copy.deepcopy(old['paired_tests_all_settings'])+tests
    last=0.
    for i,x in enumerate(sorted(alltests,key=lambda x:x['exact_mcnemar_p'])):
        last=max(last,min(1.,(len(alltests)-i)*x['exact_mcnemar_p']));x['holm_all_positions_m']=last
    r['paired_tests_all_settings']=alltests
    r['multiple_comparison_family_including_previous_numerical_versions']=len(alltests)
    r['positive_holm_significant']=sum(x['net_cases']>0 and x['holm_all_positions_m']<.05 for x in alltests)
    r['change_attribution_m']={t:{} for t in ('sp','ie')}
    for task in ('sp','ie'):
        for model in c.MODELS:
            a=r['selected'][task][model];b=old['selected'][task][model]
            r['change_attribution_m'][task][model]=dict(old_id=b['config_id'],new_id=a['config_id'],new_mode=a['mode'],parameters=a['parameters'],
                is_new_position_gain=a['config_id'].startswith('insert_m_'),changes_vs_previous={key:endpoint(a,task,key)-endpoint(dict(b,model=model),task,key)
                    for key in ENDPOINTS if endpoint(a,task,key) is not None and endpoint(dict(b,model=model),task,key) is not None})
    return r

def full_tables(r,out):
    """Every new setting, not just the favourable selected rows."""
    from matplotlib import pyplot as plt
    from predictive_tracks.ppr_existing_20261102.render import fields
    from predictive_tracks.ppr_taskwise_best_20261102_001.select_report import fmt
    plt.switch_backend('Agg')
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10})
    fig,ax=plt.subplots(figsize=(16,16));ax.set(xlim=(0,1),ylim=(0,40));ax.axis('off')
    xs=(.01,.30,.43,.54,.73,.94)
    heads=('Model / eta','Paired Acc. ↑','AUC ↑','Loc. MAE ↓','Joint Hit by Type ↑','Overall JH ↑')
    y=39;lines=[r['scope'],'Each absolute row is one actual setting. PA/AUC/JH percent; MAE original frames.','']
    web=['<!doctype html><meta charset="utf-8"><style>table{border-collapse:collapse}td,th{padding:5px 13px;text-align:center}td:first-child{text-align:left}</style><table>']
    allmetrics=[]
    for task in ('sp','ie'):
        ax.text(.01,y,'State Persistence' if task=='sp' else 'Interaction Evolution',style='italic');y-=.7
        for x,h in zip(xs,heads):ax.text(x,y,h,ha='left' if x==xs[0] else 'center',weight='bold')
        web.append('<tr><th colspan="6">'+task.upper()+'</th></tr><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in heads)+'</tr>')
        lines.extend([task.upper(),' | '.join(heads)]);y-=.65
        for model in c.MODELS:
            raw=[x for x in r['all_new_position_settings'] if x['model']==model]
            configs=[native(model,raw[0]['metrics'])]+raw
            for x in configs:
                value=x['metrics'][task]['corrected'];v=fields(value,model,task)
                text=[' / '.join(fmt(z,type_value=True) for z in a) if j==3 else fmt(a,mae=j==2) for j,a in enumerate(v)]
                label=NAMES[model]+(' native' if x['parameters']['eta']==0 else f' + eta={x["parameters"]["eta"]:g}')
                ax.text(xs[0],y,label,weight='bold' if x['parameters']['eta'] else 'normal')
                for xp,s in zip(xs[1:],text):ax.text(xp,y,s,ha='center')
                lines.append(' | '.join([label,*text]));web.append('<tr><td>'+html.escape(label)+'</td>'+''.join('<td>'+s+'</td>' for s in text)+'</tr>')
                allmetrics.append(dict(task=task,model=model,mode=x['mode'],eta=x['parameters']['eta'],config_id=x['config_id'],metrics=value))
                y-=.78
            ax.hlines(y+.4,.005,.995,lw=.4,color='gray');y-=.2
        y-=.65
    ax.text(.01,.8,'Slots: state_only. Features: before the last original block. Euclidean distance, radius multiplier1; no extra neural forwards.',fontsize=9)
    ax.text(.01,.3,'SP types: disappearance / identity / reappearance. IE: motion anomaly / penetration / contact response. Same-data exploration.',fontsize=9)
    fig.tight_layout();fig.savefig(out/'all_new_settings.png',dpi=160,facecolor='white');plt.close(fig)
    (out/'all_new_settings.txt').write_text('\n'.join(lines)+'\n')
    (out/'all_new_settings.html').write_text('\n'.join(web+['</table>'])+'\n')
    c.save(out/'all_new_settings.json',allmetrics)

def render_cumulative(r,out):
    from predictive_tracks.ppr_position_controls_20261103_k.finish_001 import render
    render(r,out)

def finish():
    if not (c.HERE/'MODEL_RUNS_COMPLETE.json').exists():raise RuntimeError('Model queue incomplete')
    c.verify(full=True);rows=new_rows();assert len(rows)==12
    a,tests,cs,hashes=audit(rows);r=assemble(rows,tests)
    out=c.HERE/'delivery_20261104_001';out.mkdir(exist_ok=False)
    c.save(out/'report.json',r);c.save(out/'COMPLETENESS_AUDIT.json',a);c.save(out/'CASEWISE_AND_PAIRED_STATISTICS.json',cs)
    c.save(out/'input_hashes.json',hashes);c.save(out/'CHANGE_ATTRIBUTION.json',r['change_attribution_m'])
    c.save(out/'RESOURCE_PEAKS.json',c.read(c.HERE/'monitor/resource_peaks.json'))
    render_cumulative(r,out);full_tables(r,out)
    lines=['New insertion positions, fixed before outcomes. Euclidean distance and radius multiplier1.','',
        'Each model/task best is selected from the declared new .1/.5/1 grid plus native; cumulative table also includes all old666 settings.','']
    for task in ('sp','ie'):
        for model in c.MODELS:
            x=r['new_only_selected'][task][model];v=x['metrics'][task]
            lines.append(f'{task.upper()} {NAMES[model]}: {x["mode"]}, eta={x["parameters"]["eta"]:.17g}, id={x["config_id"]}')
            for name in ('native_baseline','corrected'):lines.append(f'  {name}: {v[name]["overall"]}')
    (out/'NEW_POSITION_BEST.txt').write_text('\n'.join(lines)+'\n')
    (out/'READ_ME.txt').write_text(
        '12 new settings, each SP300+IE300, original100 pairs/type; independent sample count600. All negative results retained.\n'
        'New insertion positions explicitly authorized2026-11-04. Original files/weights/data/metrics untouched. Euclidean projection/radius multiplier1/all history preserved.\n'
        'No extra model forward within any forecast. Separate24 audit cases are eta0 implementation checks. Native call/order/input-shape hashes and original pixels/time spans all verified.\n'
        'all_new_settings gives every setting; NEW_POSITION_BEST is new-track-only. table.png is cumulative9-model table including all old results. Each row is one real configuration.\n'
        'Selection is same-data adaptive exploration. PA/JH confidence intervals are marginal, not selection-adjusted; conservative Holm family includes earlier numerical versions.\n'
        'Conditional MAE requires common-member checks, provided in CASEWISE. A single-step projection bound does not imply improved physical interactions or benchmark metrics.\n'
        'State-only still feeds back full position/appearance-entangled slots. Feature mean offsets at native hidden layers are not validated pure-object attribute corrections.\n')
    c.verify(full=True)
    files={str(p):c.digest(p) for p in out.iterdir() if p.is_file()}
    c.save(out/'DELIVERY.json',dict(at=time.time(),files=files,new_full_settings=12,all_audits_passed=True,
        source_report_sha256=c.digest(c.SOURCE),old_sources_unchanged=True))
    c.save(c.HERE/'FINAL_DELIVERY.json',dict(at=time.time(),directory=str(out),all_models_and_audits_complete=True))
    print('FINAL DELIVERY',out,flush=True)

if __name__=='__main__':finish()
