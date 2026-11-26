"""Package the existing S checkpoint; no model execution or result selection."""
import hashlib
import inspect
import os
import time
from pathlib import Path
from . import common as c


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    latest = c.read(c.HERE/'monitor/latest_report.json')
    source = Path(latest['directory'])/'report.json'
    result = c.read(source)
    audit_path = c.HERE/'monitor/native_comparison_013.json'
    audit = c.read(audit_path)
    assert audit['all_completed_records_exact']
    verified = {x['config_id']: x for x in audit['results']}
    inputs = {str(source): c.digest(source), str(audit_path): c.digest(audit_path)}
    selected = []
    for task in ('sp', 'ie'):
        assert len(result['selected'][task]) == 9
        for model, row in result['selected'][task].items():
            if not row['config_id'].startswith('S_') or model == 'dino_foresight':
                continue
            assert row['full600_native_validated']
            assert verified[row['config_id']]['verified'] == 600
            path = c.directory(row)/'summary.json'
            summary = c.read(path)
            assert summary['complete'] and summary['cases'] == 600
            assert row['metrics'] == summary['metrics']
            inputs[str(path)] = c.digest(path)
            selected.append(dict(task=task, model=model, config_id=row['config_id']))
    result['interim_only'] = True
    result['all_declared_experiments_complete'] = False
    result['scope'] = ('INTERIM: S model runs complete; S final lineage audit and T/U/W/Z '
                       'are pending. No result reselection. '+result['scope'])
    out = c.HERE/'verified_snapshot_044'/str(time.time_ns())
    c.save(out/'report.json', result)
    from predictive_tracks.ppr_eta_radius_20261104_n import report as renderer
    original = inspect.getsource(renderer.render)
    title = 'Historical state correction: eta / radius refinement'
    assert original.count(title) == 1
    transformed = original.replace(title, 'Historical state correction: verified results (interim)')
    with (out/'renderer_source_044.txt').open('x') as stream:
        stream.write(transformed)
    namespace = dict(vars(renderer), c=c)
    exec(compile(transformed, '<verified_snapshot_044>', 'exec'), namespace)
    namespace['render'](result, out)
    with (out/'READ_ME.txt').open('x') as stream:
        stream.write('原有检查点的完整九模型阶段表；没有重新选参或新增模型调用。\n'
                     'S所有模型实验已结束，最终来源审计仍在进行；T/U/W/Z尚未完成。\n'
                     '每个模型/任务整行来自同一设置，旧、负和零结果均保留。\n'
                     '新S主点已经完整600原推理与后缀误差逐例精确核验。DINO仍为旧适配器诊断。\n'
                     '原Video IE条件MAE和共同成员MAE均变差，不因准确率/JH上升而省略。\n'
                     'L SP共同287成员MAE不变；新增一例使整体条件MAE升高。\n'
                     'L IE共同135成员MAE从.118518519降至.088888889；整体条件MAE仍升至.125。\n'
                     '全部为同一曝光数据的探索选参结果，不是留出集验证。\n')
    c.save(out/'VERIFIED_SNAPSHOT_MANIFEST_044.json', dict(
        at=time.time(), source_checkpoint=str(source), input_hashes=inputs,
        current_S_primary_rows_checked=selected, no_result_reselection=True,
        no_model_calls=True, interim_only=True,
        S_final_lineage_audit_pending=True, remaining_stages=['T','U','W','Z'],
        original_renderer_sha256=hashlib.sha256(original.encode()).hexdigest(),
        title_only_render_change=True, source_exporter_sha256=c.digest(__file__),
        files={str(p):c.digest(p) for p in out.iterdir() if p.is_file()}))
    c.status(c.HERE/'monitor/verified_snapshot_044.json', dict(at=time.time(), directory=str(out)))
    print(out, flush=True)


if __name__ == '__main__':
    main()
