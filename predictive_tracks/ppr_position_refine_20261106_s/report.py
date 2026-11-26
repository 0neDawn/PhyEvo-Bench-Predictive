"""Retain all results, one actual setting per row, original fixed metric rules."""
import copy
import sys
import time
from pathlib import Path
from . import common as c
sys.path.insert(0,str(c.ROOT/'predictive_tracks/sp/code'))
from predictive_tracks.ppr_taskwise_best_20261102_001.select_report import ORDER,ENDPOINTS,rank,endpoint
from predictive_tracks.ppr_existing_20261102.metrics import make_summary


def jobs():
    out=list(c.read(c.HERE/'GRID.json')['cached'])
    for phase in ('coarse','fine'):
        for model in ('dino_foresight',)+c.SLOTS+c.FEATURES:
            path=c.HERE/'grids'/f'{phase}_{model}.json'
            if path.exists():
                out.extend(c.read(path)['jobs'])
    return out


def parameters(item):
    result=dict(eta=item['eta'],alpha=item['alpha'],q=None,presence=None,mass=None,min_history=None)
    if item.get('position'):
        result['position']=item['position']
    return result


def new_rows():
    rows=[]
    for item in jobs():
        native=c.directory(item)
        search=c.HERE/'search_settings'/item['model']/item['config_id']
        if item['model']=='dino_foresight':
            path=native/'summary.json'
        else:
            path=search/'summary.json'
        if not path.exists():
            continue
        summary=c.read(path)
        assert summary['complete'] and summary['cases']==600
        validated=(native/'summary.json').exists() and item['model']!='dino_foresight'
        if validated:
            assert c.read(native/'summary.json')['metrics']==summary['metrics'],item['config_id']
        rows.append(dict(item,parameters=parameters(item),metrics=summary['metrics'],complete=True,completed_pairs=600,
            directory=str(native if validated or item['model']=='dino_foresight' else search),source_directory=str(search),
            compact_case_source=summary['compact_case_source'],full600_native_validated=validated,
            partial_graph_parameter_search=item['model']!='dino_foresight',
            eligible_for_selection=(validated or item['model']=='dino_foresight') and not item.get('confirmation_only',False),
            eligible_tasks=['sp','ie'],origin='S_explicit_placement_or_eta_radius',
            diagnostic_only=item['model']=='dino_foresight'))
    return rows


def native(model,row):
    return dict(model=model,mode='native',config_id='native_eta0',parameters=parameters(dict(eta=0.,alpha=1.)),
        metrics={t:dict(native_baseline=row['metrics'][t]['native_baseline'],corrected=row['metrics'][t]['native_baseline']) for t in ('sp','ie')})


def assemble(provisional=False):
    parent=c.read(c.source_path())
    rows=new_rows()
    result=dict(at=time.time(),source_report=str(c.source_path()),source_report_sha256=c.digest(c.source_path()),
        all_configurations=copy.deepcopy(parent['all_configurations']),current_round_settings=rows,
        completed_round_configurations=len(rows),completed_corrected_model_configs=0,
        previous_selected=parent['selected'],selected={t:{} for t in ('sp','ie')},metric_specific_maxima={t:{} for t in ('sp','ie')},
        selection=parent['selection'],scope='Same exposed SP300/IE300; exploratory eta/radius and explicitly declared original-layer placement comparisons. Full original600 validates each new primary partial-graph setting.',
        dino=parent.get('dino'),extra_neural_forwards_per_forecast=0,provisional_search_ranking=provisional)
    for row in rows:
        result['all_configurations'][row['model']].append(row)
    result['completed_corrected_model_configs']=sum(map(len,result['all_configurations'].values()))
    for model in ORDER:
        values=result['all_configurations'][model]
        baseline=native(model,values[0])
        for task in ('sp','ie'):
            eligible=[x for x in values if (x.get('eligible_for_selection',True) or
                (provisional and x['config_id'].startswith('S_') and not x.get('confirmation_only',False))) and task in x.get('eligible_tasks',['sp','ie'])]+[baseline]
            assert all(x['metrics'][task]['native_baseline']==baseline['metrics'][task]['native_baseline'] for x in eligible)
            result['selected'][task][model]=copy.deepcopy(min(eligible,key=lambda x:rank(x,task)))
            for key in ENDPOINTS:
                best=min(eligible,key=lambda x:rank(x,task,key))
                bv,value=endpoint(baseline,task,key),endpoint(best,task,key)
                result['metric_specific_maxima'][task][model][key]=dict(config_id=best['config_id'],mode=best['mode'],parameters=best['parameters'],
                    baseline=bv,best_value=value,change=None if value is None or bv is None else value-bv,
                    all_metrics_at_this_setting=best['metrics'][task])
    return result


def interim():
    result=assemble()
    dest=c.HERE/'reports'/str(time.time_ns())
    c.save(dest/'report.json',result)
    c.status(c.HERE/'monitor/latest_report.json',dict(at=time.time(),directory=str(dest),settings=result['completed_round_configurations']))
    print('S COMPLETE SETTINGS',result['completed_round_configurations'],flush=True)


def make_fine():
    rows=new_rows()
    for model in ('dino_foresight',)+c.SLOTS+c.FEATURES:
        path=c.HERE/'grids'/f'fine_{model}.json'
        if path.exists():
            continue
        values=[x for x in rows if x['model']==model and not x.get('confirmation_only',False)]
        occupied={(j.get('position'),j['eta'],j['alpha']) for j in jobs() if j['model']==model}
        proposed=[]
        for task in ('sp','ie'):
            for key in ('overall_joint_hit','auc'):
                if not values:
                    continue
                anchor=min(values,key=lambda x:rank(x,task,key))
                group=[x for x in values if x.get('position')==anchor.get('position')]
                es=sorted({v['eta'] for v in group})
                aa=sorted({v['alpha'] for v in group})
                e,a=anchor['eta'],anchor['alpha']
                ei,ai=es.index(e),aa.index(a)
                ev=[round((e+es[k])/2,8) for k in (ei-1,ei+1) if 0<=k<len(es)]
                av=[round((a+aa[k])/2,8) for k in (ai-1,ai+1) if 0<=k<len(aa)]
                for eta,alpha in [(x,a) for x in ev]+[(e,x) for x in av]+[(x,y) for x in ev for y in av]:
                    token=anchor.get('position'),eta,alpha
                    if token in occupied:
                        continue
                    occupied.add(token)
                    proposed.append(c.job(model,anchor['mode'],eta,alpha,anchor.get('position'),
                        proposal='global task/endpoint midpoint refinement, no per-case tuning',anchor=anchor['config_id']))
        c.freeze_grid(f'fine_{model}',dict(at=time.time(),model=model,jobs=proposed,settings=len(proposed),all_cases_each=600))
        if model=='dino_foresight':
            c.save(c.HERE/'STAGE2.json',dict(at=time.time(),cached=proposed,native_new=[],native_reuse=[],settings=len(proposed)))


def make_validation():
    path=c.HERE/'grids/validation.json'
    if path.exists():
        return
    result=assemble(provisional=True)
    wanted={}
    for task in ('sp','ie'):
        for model in c.SLOTS+c.FEATURES:
            selected=result['selected'][task][model]
            if selected['config_id'].startswith('S_'):
                wanted[selected['config_id']]=selected
    # Also validate at least one best searched setting per model, even if no gain.
    rows=new_rows()
    for model in c.SLOTS+c.FEATURES:
        values=[x for x in rows if x['model']==model and not x.get('confirmation_only',False)]
        if values and not any(x['model']==model for x in wanted.values()):
            best=min(values,key=lambda x:rank(x,'ie'))
            wanted[best['config_id']]=best
    # Keep Video's localization alternative distinct from AUC/primary settings.
    v=result['metric_specific_maxima']['ie']['videomaev2']['localization_mae']
    if v['config_id'].startswith('S_'):
        selected=next(x for x in rows if x['config_id']==v['config_id'])
        wanted[selected['config_id']]=selected
    originals={j['config_id']:j for j in jobs()}
    c.freeze_grid('validation',dict(at=time.time(),jobs=[originals[k] for k in wanted],settings=len(wanted),
        fixed_primary_rank=True,no_per_case_parameters=True,extra_video_MAE_candidate_not_mixed=True))


def records(item):
    from predictive_tracks.ppr_eta_radius_20261104_o.path_runtime_001 import install
    original=install()
    if item.get('full600_native_validated'):
        return [c.read(p) for p in sorted((Path(item['directory'])/'pairs').glob('*.json'))]
    if item['config_id'].startswith('S_'):
        if item['model']=='dino_foresight':
            from predictive_tracks.ppr_eta_radius_20261104_o.cache_worker import expand
        else:
            from .worker import expand
        return [expand(c.read(p),item) for p in sorted(Path(item['compact_case_source']).glob('*.json'))]
    if item.get('partial_graph_parameter_search'):
        from predictive_tracks.ppr_internal_activation_20261106_r.tail_worker import expand
        return [expand(c.read(p),item) for p in sorted(Path(item['compact_case_source']).glob('*.json'))]
    return original.records(item)


def finish():
    c.verify(full=True)
    result=assemble()
    current=result['current_round_settings']
    assert len(current)==len(jobs())
    assert all(x['complete'] and x['completed_pairs']==600 for x in current)
    d=c.HERE/'delivery_20261106_001'
    if d.exists():
        if (d/'DELIVERY.json').exists():
            return
        n=2
        while d.with_name(f'delivery_20261106_{n:03d}').exists():
            n+=1
        d=d.with_name(f'delivery_20261106_{n:03d}')
    d.mkdir()
    hashes={}
    for model in c.SLOTS+c.FEATURES:
        paths=sorted((c.HERE/'capture'/model/'pairs').glob('*.json'))
        assert len(paths)==600
        for path in paths:
            meta=c.read(path)
            assert meta['original_prediction_bitwise_equal'] and meta['weights_identity_version_unchanged']
            for name,sha in ((str(path),c.digest(path)),(meta['tensor'],meta['tensor_sha256']),
                             (meta['source_meta'],meta['source_meta_sha256'])):
                if name not in hashes:
                    assert c.digest(name)==sha,name
                    hashes[name]=sha
            original=c.read(meta['source_meta'])
            if original['state'] not in hashes:
                assert c.digest(original['state'])==meta['source_state_sha256']
                hashes[original['state']]=meta['source_state_sha256']
    details=[]
    payload_cache={}
    for item in current:
        folder=item['compact_case_source']
        if folder not in payload_cache:
            payload_cache[folder]=[c.read(p) for p in sorted(Path(folder).glob('*.json'))]
        if item['model']=='dino_foresight':
            from predictive_tracks.ppr_eta_radius_20261104_o.cache_worker import expand
        else:
            from .worker import expand
        rr=[expand(p,item) for p in payload_cache[folder]]
        measured=make_summary(rr,item['model'])
        assert measured['metrics']==item['metrics']
        for payload in payload_cache[folder]:
            if item['model']=='dino_foresight':
                assert payload['native_max_abs_error']==0 and payload['neural_forwards']==0
                original=c.read(payload['native_meta'])
                assert c.digest(payload['native_meta'])==payload['source_meta_sha256']
                if original['state'] not in hashes:
                    assert c.digest(original['state'])==payload['source_state_sha256']
                    hashes[original['state']]=payload['source_state_sha256']
            else:
                assert payload['original_native_errors_exact'] and payload['original_suffix_output_bitwise_equal']
        if item.get('full600_native_validated'):
            actual=records(item)
            assert len(actual)==600
            index={f'{x["benchmark"]}_{x["archive_index"]:03d}':x for x in rr}
            for row in actual:
                assert row['trace_matches_original'] and row['extra_forwards']==0 and row['weight_identity_and_version_unchanged']
                key=f'{row["benchmark"]}_{row["archive_index"]:03d}'
                assert row['possible_errors']==index[key]['possible_errors'] and row['impossible_errors']==index[key]['impossible_errors']
                path=Path(item['directory'])/'pairs'/f'{key}.json'
                hashes[str(path)]=c.digest(path)
        details.append(dict(config_id=item['config_id'],cases=600,original_metrics_exact=True,full600_native_validated=item.get('full600_native_validated')))
        if len(details)%25==0:
            print('S FINAL AUDIT',len(details),'/',len(current),flush=True)
    required={x['config_id'] for x in c.read(c.HERE/'grids/validation.json')['jobs']}
    assert {x['config_id'] for x in current if x.get('full600_native_validated')}==required
    preflight=list((c.HERE/'preflight').glob('*/*/*.json'))
    assert preflight
    for path in preflight:
        item=c.read(path)
        assert item['original_trace_exact'] and item['full_vs_suffix_max_abs_error']==0
        hashes[str(path)]=c.digest(path)
    from predictive_tracks.ppr_eta_radius_20261104_o import report as old
    stats={}
    for task in ('sp','ie'):
        for model,item in result['selected'][task].items():
            if item['config_id']=='native_eta0':
                continue
            if item['config_id'].startswith('S_') and model!='dino_foresight':
                assert item['full600_native_validated']
            key=model+':'+item['config_id']
            if key not in stats:
                stats[key]=old.casewise(records(item),model)
    for folder,payloads in payload_cache.items():
        for path in sorted(Path(folder).glob('*.json')):
            hashes[str(path)]=c.digest(path)
    c.save(d/'report.json',result)
    c.save(d/'COMPLETENESS_AUDIT.json',dict(at=time.time(),configurations=len(details),settings=details,independent_pairs=600,
        configuration_case_evaluations=600*len(details),all_original_metrics_recomputed=True,
        original_full600_validated_configurations=len(required),partial_original_graph_search_disclosed=True))
    c.save(d/'CASEWISE_AND_PAIRED_STATISTICS.json',stats)
    c.save(d/'input_hashes.json',hashes)
    c.save(d/'RESOURCE_PEAKS.json',c.read(c.HERE/'monitor/resource_peaks.json'))
    c.install()
    old.render(result,d)
    with (d/'all_settings.tsv').open('x') as stream:
        stream.write('model\ttask\tmode\tposition\teta\talpha\tPA_percent\tAUC_percent\tMAE\tJH_percent\teligible\torigin\tid\n')
        for model,values in result['all_configurations'].items():
            for x in values:
                for task in ('sp','ie'):
                    p=x['parameters']
                    row=[model,task,x['mode'],p.get('position',''),p['eta'],p.get('alpha',1.),
                        100*endpoint(x,task,'paired_accuracy'),100*endpoint(x,task,'auc'),endpoint(x,task,'localization_mae'),
                        100*endpoint(x,task,'overall_joint_hit'),x.get('eligible_for_selection',True),x.get('origin'),x['config_id']]
                    stream.write('\t'.join(map(str,row))+'\n')
    c.verify(full=True)
    c.save(d/'DELIVERY.json',dict(at=time.time(),files={str(p):c.digest(p) for p in d.iterdir() if p.is_file()},all_checks_passed=True))
    c.save(c.HERE/'FINAL_DELIVERY.json',dict(at=time.time(),directory=str(d),complete=True))
