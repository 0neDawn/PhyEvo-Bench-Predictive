"""Single-setting rows, all outcomes retained, original metrics and paired audits."""
import copy
import html
import json
from pathlib import Path
import sys
import time
from . import common as c
sys.path.insert(0,str(c.ROOT/'predictive_tracks/sp/code'))
from predictive_tracks.ppr_taskwise_best_20261102_001.select_report import rank,endpoint,ENDPOINTS,NAMES,ORDER,TYPES
from predictive_tracks.ppr_existing_20261102.metrics import make_summary

def params(s):return dict(eta=s['eta'],alpha=s['alpha'],q=None,presence=None,mass=None,min_history=None)
def jobs():
    g=c.read(c.HERE/'GRID.json');s=c.read(c.HERE/'STAGE2.json') if (c.HERE/'STAGE2.json').exists() else dict(neural_new=[],cached_video=[])
    return g['neural_new']+g['cached_video']+s['neural_new']+s['cached_video']
def new_rows():
    out=[]
    for s in jobs():
        d=c.path(s);p=d/'summary.json'
        if not p.exists():continue
        summary=c.read(p);assert summary['complete'] and summary['cases']==600
        out.append(dict(s,parameters=params(s),metrics=summary['metrics'],complete=True,completed_pairs=600,
            directory=str(d),source_directory=str(d),summary_sha256=c.digest(p),origin='new_N',
            eligible_for_selection=True,storage='compact' if summary.get('compact_case_source') else 'individual',
            compact_case_source=summary.get('compact_case_source')))
    return out
def native(model,metrics):
    return dict(model=model,mode='native',config_id='native_eta0',parameters=dict(eta=0.,alpha=1.,q=None,presence=None,mass=None,min_history=None),
        metrics={t:dict(native_baseline=metrics[t]['native_baseline'],corrected=metrics[t]['native_baseline']) for t in ('sp','ie')})

def assemble():
    old=c.read(c.SOURCE);new=new_rows();restored=c.read(c.HERE/'GRID.json')['restored']
    all_rows=copy.deepcopy(old['all_configurations'])
    for model,rows in all_rows.items():
        for x in rows:
            x.setdefault('model',model);x.setdefault('mode','feedback' if model in ('ocvp_seq','savi_transformer','slotformer','ddlp','gswm') else 'output_only')
            x['parameters'].setdefault('alpha',1.);x.setdefault('origin','prior_M_radius1');x['eligible_for_selection']=True
    for x in restored:
        x=copy.deepcopy(x);x['eligible_for_selection']=x['model']!='videomaev2';all_rows[x['model']].append(x)
    for x in new:all_rows[x['model']].append(x)
    r=dict(at=time.time(),source_report=str(c.SOURCE),source_report_sha256=c.digest(c.SOURCE),all_configurations=all_rows,
        selection=old['selection'],scope='All exposed data, same-data adaptive eta/radius exploration. No independent validation or continuous optimum claim.',
        author_authorization='Radius multiplier adjustment reauthorized in latest user instruction; Euclidean metric/forward budget/old code unchanged.',
        radius_multiplier_search=[1,2],original_history_unchanged=True,extra_neural_forwards_per_forecast=0,
        selected={t:{} for t in ('sp','ie')},metric_specific_maxima={t:{} for t in ('sp','ie')},
        current_round_settings=new,restored_historical_radius_controls=restored,
        old_radius1_selected=old['selected'],change_attribution_n={t:{} for t in ('sp','ie')},
        completed_new_configurations=len(new),planned_stage1=240,stage2_max_new_per_model=3,
        completed_corrected_model_configs=sum(len(x) for x in all_rows.values()),
        dino=old.get('dino','Original adapter timing issue retained: diagnostic only.'),
        interpretation='One actual setting per row. Green/bold changes are descriptive; physical interaction repair and significance are not established.')
    for model in ORDER:
        rows=[x for x in all_rows[model] if x.get('eligible_for_selection',True)]
        b=native(model,rows[0]['metrics']);choices=rows+[b]
        for t in ('sp','ie'):
            for x in choices:assert x['metrics'][t]['native_baseline']==b['metrics'][t]['native_baseline']
            best=min(choices,key=lambda x:rank(x,t));r['selected'][t][model]=copy.deepcopy(best)
            prior=dict(old['selected'][t][model],model=model)
            r['change_attribution_n'][t][model]=dict(origin=best.get('origin','native'),old_id=prior['config_id'],new_id=best['config_id'],
                delta_vs_prior={k:endpoint(best,t,k)-endpoint(prior,t,k) for k in ENDPOINTS if endpoint(best,t,k) is not None and endpoint(prior,t,k) is not None})
            for k in ENDPOINTS:
                x=min(choices,key=lambda x:rank(x,t,k));bv=endpoint(b,t,k);v=endpoint(x,t,k);o=x['metrics'][t]['corrected']['overall']
                r['metric_specific_maxima'][t][model][k]=dict(config_id=x['config_id'],parameters=x['parameters'],mode=x['mode'],baseline=bv,best_value=v,
                    change=None if bv is None or v is None else v-bv,coverage=o['detection_coverage'],localization_MAE_denominator=o['localization_mae_denominator'],all_metrics_at_this_setting=x['metrics'][t])
    return r

def records(setting):
    if setting.get('compact_case_source'):
        from .cache_worker import expand
        return [expand(c.read(p),setting) for p in sorted(Path(setting['compact_case_source']).glob('*.json'))]
    return [c.read(p) for p in sorted((Path(setting['directory'])/'pairs').glob('*.json'))]

def interim():
    r=assemble();out=c.HERE/'reports'/str(time.time_ns());c.save(out/'report.json',r)
    c.status(c.HERE/'monitor/latest_report.json',dict(at=time.time(),path=str(out),new_complete=r['completed_new_configurations']))
    for m in c.MODELS:
        x=r['selected']['ie'][m]
        print('IE BEST',m,x['mode'],x['parameters'],'JH',endpoint(x,'ie','overall_joint_hit'),'PA',endpoint(x,'ie','paired_accuracy'),flush=True)
    return r

def render(r,out):
    from matplotlib import pyplot as plt
    from predictive_tracks.ppr_taskwise_best_20261102_001.select_report import fmt,color
    from predictive_tracks.ppr_existing_20261102.render import fields,difference
    plt.switch_backend('Agg');plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10.5})
    fig,ax=plt.subplots(figsize=(16,14));ax.axis('off');ax.set(xlim=(0,1),ylim=(0,43))
    xs=[.01,.29,.43,.55,.75,.93];headers=['Model','Paired Acc. ↑','AUC ↑','Loc. MAE ↓','Joint Hit by Type ↑','Overall Joint Hit ↑']
    ax.text(.5,42.5,'Historical state correction: eta / radius refinement',ha='center',weight='bold',fontsize=13)
    for x,h in zip(xs,headers):ax.text(x,41.5,h,ha='left' if x==xs[0] else 'center',weight='bold')
    ax.hlines(41,.005,.995,lw=1);y=40.2
    lines=[r['selection'],r['scope'],'Every row is one actual setting. SP and IE can differ. All declines retained.','']
    parameters=[];web=['<!doctype html><meta charset="utf-8"><style>table{border-collapse:collapse}td,th{padding:5px 12px;text-align:center}td:first-child{text-align:left}</style><table><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr>']
    tex=['% booktabs,xcolor. Exposed-data exploratory maxima.','\\begin{tabular}{lrrrrr}','\\toprule','Model & Paired Acc. & AUC & Loc. MAE & Joint Hit by Type & Overall Joint Hit \\\\','\\midrule']
    for task,title in (('sp','State Persistence'),('ie','Interaction Evolution')):
        ax.text(.01,y,title,style='italic');lines.extend([title,' | '.join(headers)]);web.append('<tr><td colspan="6"><i>'+title+'</i></td></tr>');tex.append('\\multicolumn{6}{l}{\\textit{'+title+'}} \\\\');y-=.85
        for model in ORDER:
            row=r['selected'][task][model];before=fields(row['metrics'][task]['native_baseline'],model,task);after=fields(row['metrics'][task]['corrected'],model,task);delta=difference(after,before)
            bs=[' / '.join(fmt(z,type_value=True) for z in v) if j==3 else fmt(v,mae=j==2) for j,v in enumerate(before)]
            ds=[' / '.join(fmt(z,True,type_value=True) for z in v) if j==3 else fmt(v,True,mae=j==2) for j,v in enumerate(delta)]
            ax.text(.01,y,NAMES[model]);lines.append(' | '.join([NAMES[model],*bs]));web.append('<tr><td>'+html.escape(NAMES[model])+'</td>'+''.join('<td>'+v+'</td>' for v in bs)+'</tr>')
            tex.append(NAMES[model].replace('†','$^{\\dagger}$')+' & '+' & '.join(z.replace('%','\\%') for z in bs)+' \\\\')
            for x,v in zip(xs[1:],bs):ax.text(x,y,v,ha='center')
            y-=.72
            p=row['parameters'];alpha=p.get('alpha',1.);label='+our' if row['config_id']!='native_eta0' else '+our (native)'
            ax.text(.022,y,label,weight='bold',fontsize=9);webcells=[];texcells=[]
            for j,v in enumerate(delta):
                vals=v if j==3 else [v];coords=[.684,.75,.816] if j==3 else [xs[j+1]]
                if j==3:
                    for xp in (.717,.783):ax.text(xp,y,'/',ha='center',color='#777777')
                w=[];tt=[]
                for xp,z in zip(coords,vals):
                    txt=fmt(z,True,mae=j==2,type_value=j==3);col=color(z,j==2);good=col=='#00852e'
                    ax.text(xp,y,txt,ha='center',color=col,weight='bold' if good else 'normal')
                    w.append('<span style="color:'+col+(';font-weight:bold' if good else '')+'">'+txt+'</span>')
                    cc='green!50!black' if good else 'red' if col=='#b00020' else 'gray';tt.append('\\textcolor{'+cc+'}{'+('\\textbf{'+txt+'}' if good else txt)+'}')
                webcells.append(' / '.join(w));texcells.append(' / '.join(tt))
            lines.append(' | '.join([label,*ds]));web.append('<tr><td><b>'+label+'</b></td>'+''.join('<td>'+z+'</td>' for z in webcells)+'</tr>');tex.append('\\quad +our & '+' & '.join(texcells)+' \\\\')
            parameters.append(dict(task=task,model=model,mode=row['mode'],parameters=p,origin=row.get('origin','native'),config_id=row['config_id'],directory=row.get('directory'),delta=delta))
            y-=1.08
        ax.hlines(y+.25,.005,.995,lw=.9);y-=.6;lines.append('');tex.append('\\midrule')
    notes=['Same-data adaptive selection: max Overall Joint Hit, then PA, AUC, min conditional MAE. No significance claim.',
        'Euclidean projection; all original history; no extra neural forwards. Selected eta, radius and position are in best_parameters.json.',
        'SP: disappearance / identity / reappearance. IE: motion anomaly / penetration / contact response. DINO dagger: original timing issue retained.']
    for yy,s in zip((1.35,.82,.28),notes):ax.text(.01,yy,s,fontsize=8.2)
    fig.subplots_adjust(left=.02,right=.99,bottom=.02,top=.99);fig.savefig(out/'table.png',dpi=170,facecolor='white');plt.close(fig)
    tex[-1]='\\bottomrule';tex.append('\\end{tabular}');web.append('</table>'+' '.join('<p>'+html.escape(n)+'</p>' for n in notes))
    for name,value in [('table.txt','\n'.join(lines)),('table.html','\n'.join(web)),('table.tex','\n'.join(tex))]:
        with (out/name).open('x') as f:f.write(value+'\n')
    c.save(out/'best_parameters.json',parameters)
    with (out/'best_parameters.txt').open('x') as f:
        for x in parameters:f.write(json.dumps(x,ensure_ascii=False)+'\n')

def finish():
    assert (c.HERE/'MODEL_RUNS_COMPLETE.json').exists();c.verify(full=True)
    r=assemble();new=r['current_round_settings'];assert len(new)==len(jobs())
    out=c.HERE/'delivery_20261104_001';out.mkdir(exist_ok=False)
    details=[];hashes={};tests=[]
    from predictive_tracks.ppr_20261028.run import rows_and_references
    from predictive_tracks.controlled_tracks.registry import OUTPUT
    originals={m:{r['key']:(r,p,ref) for r,p,ref in rows_and_references(m)} for m in c.MODELS}
    traces=c.read(OUTPUT/'deployment_audit_001.json')['details']
    shared={}
    for m in c.MODELS:
        ts={json.dumps(x['neural_trace'],sort_keys=True) for x in traces[m]}
        shared[m]=json.loads(next(iter(ts))) if len(ts)==1 else None
    ref_hashes={str(p):c.digest(p) for m in c.MODELS for _,p,_ in originals[m].values()}
    for setting in new:
        rs=records(setting);assert len(rs)==600
        s=make_summary(rs,setting['model']);assert s['metrics']==setting['metrics']
        for rec in rs:
            assert rec['eta']==setting['eta'] and rec['radius_multiplier']==setting['alpha']
            key=f'{rec["benchmark"]}_{rec["archive_index"]:03d}'
            row,ref_path,ref=originals[setting['model']][key]
            assert rec['pair_id']==row['pair_id'] and rec['frame_spans']==ref['frame_spans'] and rec['t_ref']==int(row['t_ref'])
            for branch in ('possible','impossible'):assert rec['matched_identity_'+branch+'_errors']==ref['matched_identity_'+branch+'_errors']
            if setting['storage']=='individual':
                assert rec['trace_matches_original'] and rec['extra_forwards']==0 and rec['weight_identity_and_version_unchanged']
                assert rec['reference_pair_sha256']==ref_hashes[str(ref_path)] and rec['source_decoded_pixels_sha256']==ref['source_decoded_pixels_sha256']
                expected=ref.get('neural_call_trace',shared[setting['model']])
                if expected is None and row['split']=='development':expected=c.read(OUTPUT/f'{setting["model"]}_boundary_dev_001/pairs'/f'{key}.json')['neural_call_trace']
                assert rec['neural_call_trace']==expected
                p=Path(setting['directory'])/'pairs'/f'{key}.json';hashes[str(p)]=c.digest(p)
            else:
                assert rec['neural_forwards']==0 and rec['score_device']=='original_cuda'
                p=Path(setting['compact_case_source'])/f'{key}.json'
                if str(p) not in hashes:
                    payload=c.read(p);assert payload['native_max_abs_error']==0
                    meta=c.read(payload['native_meta']);assert c.digest(payload['native_meta'])==payload['source_meta_sha256']
                    assert c.digest(meta['state'])==payload['source_state_sha256']
                    hashes[str(p)]=c.digest(p)
        for task,v in s['exact_paired_tests'].items():
            for key,test in v.items():tests.append(dict(test,task=task,endpoint=key,model=setting['model'],config_id=setting['config_id']))
        details.append(dict(config_id=setting['config_id'],cases=600,original_metrics_exact=True,budget_unchanged=True))
        sp=Path(setting['directory'])/'summary.json';hashes[str(sp)]=c.digest(sp)
    for p,sha in c.read(c.HERE/'RESTORED_HASHES.json').items():assert c.digest(p)==sha
    audits=[]
    for p in sorted((c.HERE/'audit').glob('*/*/summary.json')):
        a=c.read(p);assert a['audit_passed'] and a['cases']==6 and a['no_op_max_abs_error']==0
        audits.append(dict(path=str(p),cases=6,bit_exact_noop=True));hashes[str(p)]=c.digest(p)
    assert len(audits)==7
    # All historical radius controls are checked, including the excluded CPU numerical versions.
    for setting in c.read(c.HERE/'GRID.json')['restored']:
        s=make_summary(records(setting),setting['model']);assert s['metrics']==setting['metrics']
    from predictive_tracks.ppr_state_insert_20261104_m.report import casewise
    paired={}
    for m in c.MODELS:
        x=r['selected']['ie'][m]
        if x['config_id']!='native_eta0':paired[m]=dict(config_id=x['config_id'],details=casewise(records(x),m))
    r['exact_paired_tests_new']=tests
    c.save(out/'report.json',r);c.save(out/'CASEWISE_AND_PAIRED_STATISTICS.json',paired)
    c.save(out/'COMPLETENESS_AUDIT.json',dict(at=time.time(),configurations=len(new),case_records=len(new)*600,independent_pairs=600,
        settings=details,no_op_audits=audits,all_original_metrics_recomputed=True,original_sources_unchanged=True,all_negative_results_retained=True))
    c.save(out/'input_hashes.json',hashes);c.save(out/'RESOURCE_PEAKS.json',c.read(c.HERE/'monitor/resource_peaks.json'))
    render(r,out)
    with (out/'all_settings.tsv').open('x') as f:
        f.write('origin\tmodel\ttask\tmode\teta\talpha\tPA_percent\tAUC_percent\tMAE\tJH_percent\tMAE_n\tcoverage\tconfig_id\n')
        for model,rows in r['all_configurations'].items():
            for x in rows:
                x=dict(x,model=model)
                for task in ('sp','ie'):
                    v=x['metrics'][task]['corrected']['overall'];p=x['parameters']
                    a=[x.get('origin'),model,task,x['mode'],p['eta'],p.get('alpha',1.),100*endpoint(x,task,'paired_accuracy'),100*endpoint(x,task,'auc'),endpoint(x,task,'localization_mae'),100*endpoint(x,task,'overall_joint_hit'),v['localization_mae_denominator'],v['detection_coverage'],x['config_id']]
                    f.write('\t'.join(map(str,a))+'\n')
    text='All full600 settings and original metrics retained. Original SP300/IE300,100 pairs/type. Same-data adaptive exploration; no independent validation.\n'
    text+='Radius multiplier reauthorized; only eta/alpha varied at existing positions. Alpha>=1 retains original historical references.\n'
    text+='Restored alpha1.5/2 archived controls are explicitly old; Video CPU-scored versions are retained but excluded from selection in favor of declared original CUDA replay.\n'
    text+='One actual setting per model/task row; all metrics on that row share its parameters. Conditional MAE is audited on common members.\n'
    text+='Native module count/order/input-shape, decoded inputs, weights identity/version and eta0 reproduction are checked by native runner. Cache replay globally forbids NN forwards.\n'
    text+='Projection does not establish pure attribute semantics, recover missing objects, add collision rules, prove improved physics, or guarantee improved metrics.\n'
    with (out/'READ_ME.txt').open('x') as f:f.write(text)
    c.verify(full=True)
    c.save(out/'DELIVERY.json',dict(at=time.time(),files={str(p):c.digest(p) for p in out.iterdir() if p.is_file()},all_checks_passed=True))
    c.save(c.HERE/'FINAL_DELIVERY.json',dict(at=time.time(),directory=str(out),complete=True))
    print('FINAL DELIVERY',out,flush=True)

if __name__=='__main__':finish()
