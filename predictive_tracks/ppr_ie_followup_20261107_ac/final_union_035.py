"""Merge completed AE and AC/U/W/Z branches without changing selection or metrics."""
import copy
import os
import time
from pathlib import Path
from . import common as c


def verify_manifest(path):
    for file, wanted in c.read(path)['files'].items():
        assert c.digest(file) == wanted, file
    return c.digest(path)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    for path, wanted in c.read(c.HERE/'UNION_READY_035.json')['files'].items():
        assert c.digest(path) == wanted, path
    ae = c.ROOT/'predictive_tracks/ppr_buffer_read_20261107_ae'
    review = c.ROOT/'predictive_tracks/ppr_completion_review_20261106_v/FINAL_HANDOFF_001.json'
    required = [c.HERE/'MERGED_FINAL_DELIVERY_019.json', ae/'COMPLETED_001.json',
        c.HERE/'ALL_REVIEWS_COMPLETE_023.json', review]
    while not all(p.exists() for p in required):
        c.status(c.HERE/'monitor/final_union_035.json', dict(at=time.time(), state='waiting',
            missing=[str(p) for p in required if not p.exists()], no_GPU=True))
        time.sleep(30)
    inputs = [Path(c.read(c.HERE/'MERGED_FINAL_DELIVERY_019.json')['directory']),
        Path(c.read(ae/'FINAL_DELIVERY.json')['directory'])]
    manifests = {str(d/'DELIVERY.json'):verify_manifest(d/'DELIVERY.json') for d in inputs}
    priority = c.read(ae/'SCHEDULER_READY_004.json')
    for path, wanted in priority['files'].items():
        assert c.digest(path) == wanted, path
    from predictive_tracks.ppr_taskwise_best_20261102_001.select_report import ORDER, ENDPOINTS, rank, endpoint
    result = c.read(inputs[0]/'report.json')
    addition = c.read(inputs[1]/'report.json')
    result['previous_selected'] = copy.deepcopy(result['selected'])
    for model in ORDER:
        known = {v['config_id']:v for v in result['all_configurations'][model]}
        for row in addition['all_configurations'][model]:
            if row['config_id'] in known:
                old = known[row['config_id']]
                assert old['parameters'] == row['parameters'] and old['metrics'] == row['metrics']
            else:
                result['all_configurations'][model].append(row)
                known[row['config_id']] = row
    result['selected'] = {t:{} for t in ('sp','ie')}
    result['metric_specific_maxima'] = {t:{} for t in ('sp','ie')}
    for model in ORDER:
        values = result['all_configurations'][model]
        baseline = dict(model=model, mode='native', config_id='native_eta0', parameters=dict(eta=0., alpha=1.),
            metrics={t:dict(native_baseline=values[0]['metrics'][t]['native_baseline'],
                corrected=values[0]['metrics'][t]['native_baseline']) for t in ('sp','ie')})
        for task in ('sp','ie'):
            eligible = [v for v in values if v.get('eligible_for_selection',True)
                and task in v.get('eligible_tasks',['sp','ie'])] + [baseline]
            assert all(v['metrics'][task]['native_baseline'] == baseline['metrics'][task]['native_baseline'] for v in eligible)
            chosen = min(eligible, key=lambda v:rank(v,task))
            if chosen['config_id'].startswith(('AC_', 'AE_', 'Z_')):
                assert chosen['full600_native_validated']
                assert not chosen.get('confirmation_only',False)
            result['selected'][task][model] = copy.deepcopy(chosen)
            alternatives = result['metric_specific_maxima'][task][model] = {}
            for metric in ENDPOINTS:
                item = min(eligible, key=lambda v:rank(v,task,metric))
                before, after = endpoint(baseline,task,metric), endpoint(item,task,metric)
                alternatives[metric] = dict(config_id=item['config_id'],mode=item['mode'],parameters=item['parameters'],
                    baseline=before,best_value=after,change=None if before is None or after is None else after-before,
                    all_metrics_at_this_setting=item['metrics'][task])
    current = {v['config_id']:v for v in result['current_round_settings']}
    current.update({v['config_id']:v for v in addition['current_round_settings']})
    result.update(at=time.time(), current_round_settings=list(current.values()),
        completed_round_configurations=len(current),
        completed_corrected_model_configs=sum(map(len,result['all_configurations'].values())),
        source_reports={str(d/'report.json'):c.digest(d/'report.json') for d in inputs},
        scope='All completed AC/AE/U/W/Z branches; original ranking, Euclidean correction, same exposed SP300/IE300. One setting per row. DINO original alignment remains diagnostic.',
        no_new_experiment_or_metric=True)
    stats = {}
    for directory in inputs:
        for key, value in c.read(directory/'CASEWISE_AND_PAIRED_STATISTICS.json').items():
            if key in stats:
                assert stats[key] == value
            else:
                stats[key] = value
    selected_stats = {}
    for task in ('sp','ie'):
        for model, row in result['selected'][task].items():
            if row['config_id'] != 'native_eta0':
                key = model+':'+row['config_id']
                assert key in stats, key
                selected_stats[key] = stats[key]
    index = 1
    while (c.HERE/f'final_delivery_035_{index:03d}').exists():
        index += 1
    out = c.HERE/f'final_delivery_035_{index:03d}'
    out.mkdir()
    c.save(out/'report.json',result)
    c.save(out/'CASEWISE_AND_PAIRED_STATISTICS.json',selected_stats)
    from predictive_tracks.ppr_eta_radius_20261104_n import report as renderer
    renderer.c = c
    renderer.render(result,out)
    alternatives = [dict(task=t,model=m,metric=k,**v)
        for t,models in result['metric_specific_maxima'].items()
        for m,metrics in models.items() for k,v in metrics.items()]
    c.save(out/'METRIC_ALTERNATIVES.json',alternatives)
    c.save(out/'UNION_AUDIT_035.json',dict(at=time.time(),input_manifests=manifests,
        source_report_hashes=result['source_reports'],original_selection_and_metrics=True,
        one_actual_setting_per_row=True,all_negative_and_parent_results_retained=True,
        all_new_primary_native600_verified=True,independent_AC_AE_and_legacy_reviews_present=True,
        same_exposed_data_exploratory=True,DINO_diagnostic_only=True,no_model_calls=True,
        root_visual_inspection_pending=True))
    with (out/'READ_ME.txt').open('x') as f:
        f.write('完整九模型总表：AC、AE与后续U/W/Z都完成并审核后合并。每行一个实际设置，SP/IE分别选全局参数；各指标单独最优完整行另存。\n'
            '全部为曝光SP300/IE300上的探索选参，非留出验证。原权重、欧氏距离、评分、前向次数不变；所有负/零/失败结果保留。\n'
            'AE为已声明的新读取位置，匹配单次写回对照见AE源交付；eta1读取极限控制不参与新位置选优。\n'
            'DINO†保留原时间错位，只作诊断；缓存评分与原生标量的数值归约差分开留档。\n'
            '定位MAE下降或升高都报告，并提供共同成员统计。根代理必须实际查看table.png后才能称目视完成。\n')
    c.save(out/'DELIVERY.json',dict(at=time.time(),files={str(p):c.digest(p) for p in out.iterdir() if p.is_file()},all_union_checks_passed=True))
    c.save(c.HERE/'FINAL_ALL_DELIVERY_035.json',dict(at=time.time(),directory=str(out),complete=True,visual_inspection_pending=True))
    c.status(c.HERE/'monitor/final_union_035.json',dict(at=time.time(),state='complete',directory=str(out),no_GPU=True,visual_inspection_pending=True))


if __name__ == '__main__':
    main()
