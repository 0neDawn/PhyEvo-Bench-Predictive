"""One actual setting per row; never mix best endpoints across settings."""
import copy
import time
from pathlib import Path
from . import common as c
from predictive_tracks.ppr_taskwise_best_20261102_001.select_report import ORDER,ENDPOINTS,rank,endpoint


def jobs():
    values = []
    for phase in ('coarse','fine'):
        for model in c.MODELS:
            path = c.HERE/'grids'/f'{phase}_{model}.json'
            if path.exists():values.extend(c.read(path)['jobs'])
    assert len({v['config_id'] for v in values}) == len(values)
    return values


def parameters(item):
    return dict(eta=item['eta'],alpha=item['alpha'],q=None,presence=None,mass=None,min_history=None)


def new_rows():
    result = []
    for item in jobs():
        source = c.HERE/'search_settings'/item['model']/item['config_id']/'summary.json'
        if not source.exists():continue
        summary = c.read(source)
        assert summary['complete'] and summary['cases'] == 600
        native = c.directory(item)/'summary.json'
        validated = native.exists()
        if validated:assert c.read(native)['metrics'] == summary['metrics']
        result.append(dict(item,parameters=parameters(item),metrics=summary['metrics'],complete=True,completed_pairs=600,
            directory=str(native.parent if validated else source.parent),source_directory=str(source.parent),
            compact_case_source=summary['compact_case_source'],full600_native_validated=validated,
            partial_graph_parameter_search=True,eligible_for_selection=validated,eligible_tasks=['sp','ie'],
            origin='AC_existing_eta_radius_and_original_loop',diagnostic_only=item['model']=='dino_foresight'))
    return result


def assemble(provisional=False):
    parent = c.read(c.source_path())
    current = new_rows()
    result = dict(at=time.time(),source_report=str(c.source_path()),source_report_sha256=c.digest(c.source_path()),
        all_configurations=copy.deepcopy(parent['all_configurations']),current_round_settings=current,
        completed_round_configurations=len(current),previous_selected=parent['selected'],selected={t:{} for t in ('sp','ie')},
        metric_specific_maxima={t:{} for t in ('sp','ie')},selection=parent['selection'],dino=parent.get('dino'),
        scope='Same exposed SP300/IE300. Existing slot eta/radius refinement and unchanged DINO-loop projection. DINO remains diagnostic; existing cached scoring is fixed and native per-frame scalars are separately retained.',
        extra_neural_forwards_per_forecast=0,provisional_search_ranking=provisional)
    for row in current:result['all_configurations'][row['model']].append(row)
    result['completed_corrected_model_configs'] = sum(map(len,result['all_configurations'].values()))
    for model in ORDER:
        values = result['all_configurations'][model]
        baseline = dict(model=model,mode='native',config_id='native_eta0',parameters=parameters(dict(eta=0.,alpha=1.)),
            metrics={t:dict(native_baseline=values[0]['metrics'][t]['native_baseline'],corrected=values[0]['metrics'][t]['native_baseline']) for t in ('sp','ie')})
        for task in ('sp','ie'):
            eligible = [v for v in values if (v.get('eligible_for_selection',True) or
                (provisional and v['config_id'].startswith('AC_'))) and task in v.get('eligible_tasks',['sp','ie'])]+[baseline]
            assert all(v['metrics'][task]['native_baseline'] == baseline['metrics'][task]['native_baseline'] for v in eligible)
            result['selected'][task][model] = copy.deepcopy(min(eligible,key=lambda v:rank(v,task)))
            result['metric_specific_maxima'][task][model] = {}
            for key in ENDPOINTS:
                best = min(eligible,key=lambda v:rank(v,task,key))
                before,after = endpoint(baseline,task,key),endpoint(best,task,key)
                result['metric_specific_maxima'][task][model][key] = dict(config_id=best['config_id'],mode=best['mode'],parameters=best['parameters'],
                    baseline=before,best_value=after,change=None if before is None or after is None else after-before,
                    all_metrics_at_this_setting=best['metrics'][task])
    return result


def make_fine():
    parent = c.read(c.source_path())
    current = new_rows()
    for model in c.MODELS:
        path = c.HERE/'grids'/f'fine_{model}.json'
        if path.exists():continue
        old = [dict(v,eta=v['parameters']['eta'],alpha=v['parameters'].get('alpha',1.))
            for v in parent['all_configurations'][model] if v.get('eligible_for_selection',True)]
        own = [v for v in current if v['model']==model]
        occupied = {(v['mode'],v['eta'],v['alpha']) for v in old+own}
        proposed,anchors = [],[]
        modes = ('feedback',) if model == 'dino_foresight' else ('feedback','state_only')
        for mode in modes:
            group = [v for v in old+own if v['mode']==mode]
            es = sorted({v['eta'] for v in group});aa = sorted({v['alpha'] for v in group})
            for task in (('sp','ie') if model == 'dino_foresight' else ('ie',)):
                for metric in ('overall_joint_hit','paired_accuracy','auc'):
                    anchor = min(group,key=lambda v:rank(v,task,metric))
                    anchors.append(dict(task=task,mode=mode,metric=metric,config_id=anchor['config_id']))
                    e,a = anchor['eta'],anchor['alpha'];ei,ai = es.index(e),aa.index(a)
                    ee = [e]+[round((e+es[k])/2,8) for k in (ei-1,ei+1) if 0<=k<len(es)]
                    av = [a]+[round((a+aa[k])/2,8) for k in (ai-1,ai+1) if 0<=k<len(aa)]
                    for eta in ee:
                        for alpha in av:
                            token = mode,eta,alpha
                            if token in occupied:continue
                            occupied.add(token)
                            proposed.append(c.job(model,mode,eta,alpha,anchor=anchor['config_id'],proposal='One additional global-endpoint neighbour midpoint round; same existing eta/radius only'))
        c.freeze_grid(f'fine_{model}',dict(at=time.time(),model=model,jobs=proposed,settings=len(proposed),cases_each=600,anchors=anchors,
            source_generator=str(Path(__file__).resolve()),source_generator_sha256=c.digest(__file__)))


def make_validation():
    path = c.HERE/'grids/validation.json'
    if path.exists():return
    result = assemble(provisional=True)
    available = {v['config_id']:v for v in jobs()}
    wanted = set()
    for task in ('sp','ie'):
        for model in c.MODELS:
            wanted.add(result['selected'][task][model]['config_id'])
            wanted.add(result['metric_specific_maxima'][task][model]['paired_accuracy']['config_id'])
    for model in c.MODELS:
        if not any(v in available and available[v]['model']==model for v in wanted):
            wanted.add(min([v for v in new_rows() if v['model']==model],key=lambda v:rank(v,'ie'))['config_id'])
    required = [available[k] for k in sorted(wanted) if k in available]
    c.freeze_grid('validation',dict(at=time.time(),jobs=required,settings=len(required),cases_each=600,
        new_primary_and_PA_maxima_native600_required=True,no_case_specific_parameters=True))


def records(item):
    if item.get('full600_native_validated'):
        return [c.read(p) for p in sorted((Path(item['directory'])/'pairs').glob('*.json'))]
    if item['config_id'].startswith('AC_'):
        from predictive_tracks.ppr_position_refine_20261106_s.worker import expand
        return [expand(c.read(p),item) for p in sorted(Path(item['compact_case_source']).glob('*.json'))]
    from predictive_tracks.ppr_eta_radius_20261104_o.path_runtime_001 import install
    return install().records(item)


def finish():
    import sys
    sys.path.insert(0,str(c.ROOT/'predictive_tracks/sp/code'))
    from predictive_tracks.ppr_existing_20261102.metrics import make_summary
    from predictive_tracks.ppr_eta_radius_20261104_o.report import casewise
    c.verify(full=True)
    result = assemble()
    assert len(result['current_round_settings']) == len(jobs())
    cache,hashes,checked = {},{},[]
    for row in result['current_round_settings']:
        folder = row['compact_case_source']
        if folder not in cache:
            paths = sorted(Path(folder).glob('*.json'))
            assert len(paths) == 600
            cache[folder] = [c.read(p) for p in paths]
            hashes.update({str(p):c.digest(p) for p in paths})
        from predictive_tracks.ppr_position_refine_20261106_s.worker import expand
        expected = [expand(p,row) for p in cache[folder]]
        assert make_summary(expected,row['model'])['metrics'] == row['metrics']
        assert all(p['original_native_errors_exact'] and p['original_suffix_output_bitwise_equal'] for p in cache[folder])
        if row['full600_native_validated']:
            actual = records(row)
            assert len(actual) == 600
            lookup = {f'{v["benchmark"]}_{v["archive_index"]:03d}':v for v in expected}
            for v in actual:
                key = f'{v["benchmark"]}_{v["archive_index"]:03d}'
                assert v['possible_errors'] == lookup[key]['possible_errors'] and v['impossible_errors'] == lookup[key]['impossible_errors']
                assert v['extra_forwards'] == 0 and v['trace_matches_original'] and v['weight_identity_and_version_unchanged']
            for p in (Path(row['directory'])/'pairs').glob('*.json'):hashes[str(p)]=c.digest(p)
        checked.append(dict(config_id=row['config_id'],cases=600,native600=row['full600_native_validated']))
    previous_stats = c.read(c.source_path().parent/'CASEWISE_AND_PAIRED_STATISTICS.json')
    stats = {}
    for task in ('sp','ie'):
        for model,item in result['selected'][task].items():
            if item['config_id']=='native_eta0':continue
            key = model+':'+item['config_id']
            if key not in stats:
                stats[key] = copy.deepcopy(previous_stats[key]) if key in previous_stats else casewise(records(item),model)
            if item['config_id'].startswith('AC_'):assert item['full600_native_validated']
    out = c.HERE/'delivery_20261107_001'
    out.mkdir(exist_ok=False)
    c.save(out/'report.json',result)
    c.save(out/'COMPLETENESS_AUDIT.json',dict(at=time.time(),configurations=len(checked),settings=checked,independent_pairs=600,
        configuration_case_evaluations=len(checked)*600,all_original_metrics_recomputed=True,new_primary_native600_required=True,
        same_exposed_data_exploratory=True,dino_alignment_unchanged_diagnostic=True,existing_DINO_cache_scoring_expression_unchanged=True))
    c.save(out/'CASEWISE_AND_PAIRED_STATISTICS.json',stats)
    c.save(out/'input_hashes.json',hashes)
    from predictive_tracks.ppr_eta_radius_20261104_n import report as renderer
    renderer.c = c
    renderer.render(result,out)
    alternatives = []
    for task in ('sp','ie'):
        for model,values in result['metric_specific_maxima'][task].items():
            for metric,row in values.items():alternatives.append(dict(task=task,model=model,metric=metric,**row))
    c.save(out/'METRIC_ALTERNATIVES.json',alternatives)
    with (out/'READ_ME.txt').open('x') as stream:
        stream.write('每行只取一个真实参数设置，主表排序沿用原规则；各端点完整候选见METRIC_ALTERNATIVES.json。\n'
            'SP300、IE300，每类型100；同数据探索选参，不是留出验证。所有旧、负、零及失败记录均保留。\n'
            '只调原η/半径与原循环位置。原权重、欧氏距离、历史选择、前向次数与评分不变。\n'
            'DINO†保留原时间问题，仅作诊断；使用既有NativeCache.errors，原适配器逐帧标量另存。\n'
            'DINO循环仍6次forecaster、5帧历史+12帧候选编码；CPU投影与旧算子逐值相同，传输增加墙钟时间。\n'
            '条件MAE需同时看统计成员与共同成员；定位/AUC下降均不隐藏。\n')
    c.save(out/'DELIVERY.json',dict(at=time.time(),files={str(p):c.digest(p) for p in out.iterdir() if p.is_file()},all_checks_passed=True))
    c.save(c.HERE/'FINAL_DELIVERY.json',dict(at=time.time(),directory=str(out),complete=True))
    c.save(c.HERE/'FINAL_STATE_001.json',dict(at=time.time(),all_declared_configurations_complete=True,all_lineage_audits_complete=True,
        every_new_primary_original_full600_validated=True,dino_diagnostic_only=True,directory=str(out)))
    print('AC complete full nine-model table',out,flush=True)
