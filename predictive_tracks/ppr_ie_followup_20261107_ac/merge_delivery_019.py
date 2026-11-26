"""Combine completed independent branches using the original selection rule only."""
import json
import os
import time
from pathlib import Path
from . import common as c


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    c.verify_sources('MERGE_READY_019.json')
    z = c.ROOT/'predictive_tracks/ppr_after_norm_20261106_z'
    while not (c.HERE/'COMPLETED_012.json').exists() or not (z/'COMPLETED_001.json').exists():
        c.status(c.HERE/'monitor/merge_019.json',dict(at=time.time(),state='waiting_for_both_AC_and_Z',no_GPU=True,
            AC_complete=(c.HERE/'COMPLETED_012.json').exists(),Z_complete=(z/'COMPLETED_001.json').exists()))
        time.sleep(30)
    ac_dir = Path(c.read(c.HERE/'FINAL_DELIVERY.json')['directory'])
    z_dir = Path(c.read(z/'FINAL_DELIVERY.json')['directory'])
    inputs = {}
    for directory in (ac_dir,z_dir):
        manifest = directory/'DELIVERY.json'
        for path,wanted in c.read(manifest)['files'].items():assert c.digest(path)==wanted,path
        inputs[str(manifest)] = c.digest(manifest)
    from predictive_tracks.ppr_taskwise_best_20261102_001.select_report import ORDER,ENDPOINTS,rank,endpoint
    result = c.read(z_dir/'report.json')
    own = c.read(ac_dir/'report.json')
    ac_rows = own['current_round_settings']
    del own
    original_z_selected = result['selected']
    for model in ORDER:
        identifiers = {v['config_id']:v for v in result['all_configurations'][model]}
        for row in (v for v in ac_rows if v['model']==model):
            if row['config_id'] in identifiers:
                previous = identifiers[row['config_id']]
                assert previous['parameters']==row['parameters'] and previous['metrics']==row['metrics']
            else:
                result['all_configurations'][model].append(row)
                identifiers[row['config_id']]=row
    result['previous_selected'] = original_z_selected
    result['selected'] = {t:{} for t in ('sp','ie')}
    result['metric_specific_maxima'] = {t:{} for t in ('sp','ie')}
    for model in ORDER:
        values = result['all_configurations'][model]
        baseline = dict(model=model,mode='native',config_id='native_eta0',parameters=dict(eta=0.,alpha=1.),
            metrics={t:dict(native_baseline=values[0]['metrics'][t]['native_baseline'],corrected=values[0]['metrics'][t]['native_baseline']) for t in ('sp','ie')})
        for task in ('sp','ie'):
            eligible = [v for v in values if v.get('eligible_for_selection',True) and task in v.get('eligible_tasks',['sp','ie'])]+[baseline]
            assert all(v['metrics'][task]['native_baseline']==baseline['metrics'][task]['native_baseline'] for v in eligible)
            chosen = min(eligible,key=lambda v:rank(v,task))
            if chosen['config_id'].startswith('AC_'):assert chosen['full600_native_validated']
            if chosen['config_id'].startswith('Z_'):assert chosen['full600_native_validated']
            result['selected'][task][model]=chosen
            result['metric_specific_maxima'][task][model]={}
            for metric in ENDPOINTS:
                item = min(eligible,key=lambda v:rank(v,task,metric))
                before,after = endpoint(baseline,task,metric),endpoint(item,task,metric)
                result['metric_specific_maxima'][task][model][metric]=dict(config_id=item['config_id'],mode=item['mode'],parameters=item['parameters'],
                    baseline=before,best_value=after,change=None if before is None or after is None else after-before,
                    all_metrics_at_this_setting=item['metrics'][task])
    result.update(at=time.time(),completed_corrected_model_configs=sum(map(len,result['all_configurations'].values())),
        current_round_settings=ac_rows+result['current_round_settings'],
        source_reports={str(ac_dir/'report.json'):c.digest(ac_dir/'report.json'),str(z_dir/'report.json'):c.digest(z_dir/'report.json')},
        scope='Completed AC and Z branches merged with original ranking; same exposed SP300/IE300, one actual setting per row, no held-out or significance claim. DINO original alignment remains diagnostic.',
        no_new_experiment_or_metric=True)
    result['completed_round_configurations']=len(result['current_round_settings'])
    stats = c.read(ac_dir/'CASEWISE_AND_PAIRED_STATISTICS.json')
    for key,value in c.read(z_dir/'CASEWISE_AND_PAIRED_STATISTICS.json').items():stats.setdefault(key,value)
    chosen_stats = {}
    for task in ('sp','ie'):
        for model,row in result['selected'][task].items():
            if row['config_id']!='native_eta0':
                key = model+':'+row['config_id']
                assert key in stats,key
                chosen_stats[key]=stats[key]
    out = c.HERE/'merged_delivery_20261107_001'
    if out.exists():
        assert (out/'DELIVERY.json').exists(),'Incomplete old merge preserved; use a new recovery output'
        return
    out.mkdir()
    c.save(out/'report.json',result)
    c.save(out/'CASEWISE_AND_PAIRED_STATISTICS.json',chosen_stats)
    from predictive_tracks.ppr_eta_radius_20261104_n import report as renderer
    renderer.c = c
    renderer.render(result,out)
    alternatives = [dict(task=t,model=m,metric=k,**v) for t,models in result['metric_specific_maxima'].items()
        for m,metrics in models.items() for k,v in metrics.items()]
    c.save(out/'METRIC_ALTERNATIVES.json',alternatives)
    c.save(out/'MERGE_AUDIT.json',dict(at=time.time(),original_selection_rule_unchanged=True,
        baseline_identity_checked_for_all_eligible_settings=True,all_parent_and_negative_results_preserved=True,
        every_new_selected_AC_or_Z_setting_native600_verified=True,all_metrics_from_one_setting_per_row=True,
        same_exposed_data_exploratory=True,no_new_model_calls=True,DINO_diagnostic_only=True,
        input_manifests=inputs,root_visual_inspection_still_required=True))
    with (out/'READ_ME.txt').open('x') as stream:
        stream.write('AC与Z全部完成后的九模型总表和对应参数；每行来自同一实际设置，SP/IE可分别使用全局参数。\n'
            '增益加粗绿色、下降红色，原欧氏/权重/数据/指标/神经前向预算保持，全部旧和失败结果保留。\n'
            '同数据探索选参，非独立验证；DINO†保留原时间问题，仅作诊断。\n'
            '原生DINO标量与缓存评分的归约差另存，不用于制造提升。条件MAE同时看共同成员。\n'
            '图已自动生成，但不能将此文件当作根代理已目视检查的证明。\n')
    c.save(out/'DELIVERY.json',dict(at=time.time(),files={str(p):c.digest(p) for p in out.iterdir() if p.is_file()},all_merge_checks_passed=True))
    c.save(c.HERE/'MERGED_FINAL_DELIVERY_019.json',dict(at=time.time(),directory=str(out),complete=True,visual_inspection_pending=True))
    c.status(c.HERE/'monitor/merge_019.json',dict(at=time.time(),state='complete',directory=str(out),no_GPU=True))


if __name__=='__main__':main()
