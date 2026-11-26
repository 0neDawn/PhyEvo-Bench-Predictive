"""Select one actual shared SP/IE setting per model; retain every grid row."""
import fcntl
import sys
from pathlib import Path
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'predictive_tracks/sp/code'))
from .common import HERE,MODELS,configs,config_id,result_folder,old_folder,read,save,status,digest,primary,verify
from .metrics import make_summary
from predictive_tracks.controlled_tracks.registry import NATIVE_TOLERANCE

def rank(entry,model):
    tol=str(NATIVE_TOLERANCE[model]);m=entry['metrics']
    joint=sum(round(300*m[t]['corrected']['overall']['joint_hit'][tol]) for t in ('sp','ie'))
    paired=sum(round(300*m[t]['corrected']['overall']['paired_accuracy']) for t in ('sp','ie'))
    return (-joint,-paired,entry['config_id'])

def obtain(model,c):
    folder=result_folder(model,c);source=folder/'summary.json'
    if not source.exists():return None
    if old_folder(model,c) is None:return read(source)
    cached=HERE/'imports'/model/f'{config_id(c)}.json'
    if cached.exists():
        value=read(cached)
        if value['source_summary_sha256']!=digest(source):raise RuntimeError('Old result summary changed')
        return value
    rs=[read(p) for p in sorted((folder/'pairs').glob('*.json'))]
    value=make_summary(rs,model);previous=read(source)
    if previous['metrics']!=value['metrics']:raise RuntimeError('Reused original metrics changed')
    value.update(source_directory=str(folder),source_summary_sha256=digest(source),parameters=c)
    save(cached,value);return value

def main(final=False):
    (HERE/'monitor').mkdir(exist_ok=True)
    lock=(HERE/'monitor/report.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    verify()
    out=HERE/'reports'/str(time.time_ns());out.mkdir(parents=True)
    report={'scope':'Exploratory model-specific selection on previously exposed SP300+IE300, not independent test',
        'selection':'Max total Overall JointHit cases over SP+IE; tie max total paired-correct cases; tie lexicographic configuration ID',
        'alpha_fixed':1,'same_parameters_for_SP_IE_and_all_columns':True,'expected_corrected_model_configs':220,
        'selected':{},'all_configurations':{},'completion':{},'paired_tests_all_settings':[],
        'dino':'Original temporal misalignment retained; diagnostic only',
        'pending_mechanism_controls':['random same-norm changes','shuffled object references']}
    completed=0
    for model in MODELS:
        entries=[];jobs=[]
        for c in configs(model):
            cid=config_id(c);folder=result_folder(model,c);summary=obtain(model,c)
            if summary is None:
                progress=folder/'progress.json'
                jobs.append({'config_id':cid,'parameters':c,'complete':False,
                    'completed_pairs':read(progress).get('completed',0) if progress.exists() else 0})
                continue
            entry={'config_id':cid,'parameters':c,'mode':primary(model),'source_directory':str(folder),
                'reused':old_folder(model,c) is not None,'metrics':summary['metrics']}
            entries.append(entry);completed+=1
            jobs.append({'config_id':cid,'parameters':c,'complete':True,'completed_pairs':600,'reused':entry['reused'],'metrics':summary['metrics']})
            for task,tests in summary['exact_paired_tests'].items():
                for endpoint,value in tests.items():report['paired_tests_all_settings'].append(dict(value,model=model,config_id=cid,benchmark=task,endpoint=endpoint))
        report['all_configurations'][model]=jobs
        report['completion'][model]={'complete':len(entries)==len(configs(model)),'complete_configs':len(entries),'expected_configs':len(configs(model))}
        if entries:
            selected=min(entries,key=lambda x:rank(x,model));selected['model_search_complete']=report['completion'][model]['complete']
            selected['best_including_native_is_disabled']=False
            tol=str(NATIVE_TOLERANCE[model]);m=selected['metrics']
            native_key=(-sum(round(300*m[t]['native_baseline']['overall']['joint_hit'][tol]) for t in ('sp','ie')),
                -sum(round(300*m[t]['native_baseline']['overall']['paired_accuracy']) for t in ('sp','ie')))
            if native_key<=rank(selected,model)[:2]:selected['best_including_native_is_disabled']=True
            report['selected'][model]=selected
    report['completed_corrected_model_configs']=completed
    report['all_complete']=completed==220
    if final and not report['all_complete']:raise RuntimeError('Cannot label partial grid complete')
    # Remaining tests have p=1 until complete; family size is fixed in advance.
    maximum=0.
    for i,test in enumerate(sorted(report['paired_tests_all_settings'],key=lambda x:x['exact_mcnemar_p'])):
        maximum=max(maximum,min(1.,(880-i)*test['exact_mcnemar_p']));test['holm_all_880']=maximum
    if final:
        from predictive_tracks.ppr_reports_20261028.report import comparison
        report['selected_descriptive_bootstrap']=[]
        for model,entry in report['selected'].items():
            rs=[read(p) for p in sorted((Path(entry['source_directory'])/'pairs').glob('*.json'))]
            for task in ('sp','ie'):
                for endpoint,value in comparison([r for r in rs if r['benchmark']==task],model).items():
                    report['selected_descriptive_bootstrap'].append(dict(value,model=model,config_id=entry['config_id'],benchmark=task,endpoint=endpoint,
                        caveat='Post-selection marginal descriptive CI; does not account for parameter selection or imply independent validation'))
    from .render import render
    save(out/'report.json',report);render(report,out)
    status(HERE/'monitor/latest_report.json',{'directory':str(out),'all_complete':report['all_complete'],
        'completed_configs':completed,'expected_configs':220,'at':time.time()})
    print('REPORT',out,'complete',completed,'/220',flush=True)
    return report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args();main(a.final)
