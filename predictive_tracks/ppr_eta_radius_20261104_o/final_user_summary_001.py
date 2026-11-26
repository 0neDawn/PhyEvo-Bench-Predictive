"""Package completed, audited nine-model results; never run a model or rerank."""
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from predictive_tracks.ppr_existing_20261102.render import fields, difference
from predictive_tracks.ppr_taskwise_best_20261102_001.select_report import ORDER, NAMES, fmt


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    assert read(HERE / "PERSISTED.json")["all_experiments_and_audits_complete"]
    assert read(HERE / "FINAL_STATE_001.json")["all_lineage_audits_complete"]
    delivery = Path(read(HERE / "FINAL_DELIVERY.json")["directory"])
    for name in ("DELIVERY.json", "SUPPLEMENT_MANIFEST_001.json"):
        for path, sha256 in read(delivery / name)["files"].items():
            assert digest(path) == sha256, path
    completeness = read(delivery / "COMPLETENESS_AUDIT.json")
    assert completeness["all_original_metrics_recomputed"] and completeness["independent_pairs"] == 600
    report = read(delivery / "report.json")
    assert len(report["current_round_settings"]) == completeness["configurations"]
    assert set(report["selected"]["sp"]) == set(ORDER) == set(report["selected"]["ie"])
    out = delivery / "user_summary_001"
    out.mkdir(exist_ok=False)
    rows = []
    lines = [f'已完成本轮 {completeness["configurations"]} 组全量配置及原指标、来源核验。',
             '每组均为原 SP 300 对、IE 300 对，每类型 100 对。SP、IE 分别选参，同一行全部指标使用同一组真实参数。',
             '选择顺序：Overall Joint Hit → Paired Acc. → AUC → 更低的条件 Loc. MAE；完全并列保留原模型。',
             '本轮仅调整 η 和半径倍率 α，原模型权重、欧氏距离、历史选择、已授权插入位置、原前向预算及指标保持不变。',
             '', f'![全部模型结果]({delivery / "table.png"})', '']
    headers = '| Model | Paired Acc. ↑ | AUC ↑ | Loc. MAE ↓ | Joint Hit by Type ↑ | Overall Joint Hit ↑ |'
    for task, title in (("sp", "State Persistence"), ("ie", "Interaction Evolution")):
        lines.extend([title, '', headers, '|---|---:|---:|---:|---:|---:|'])
        for model in ORDER:
            setting = report["selected"][task][model]
            if setting["config_id"] != "native_eta0":
                assert any(x["config_id"] == setting["config_id"] and x["parameters"] == setting["parameters"] and x["metrics"] == setting["metrics"]
                           for x in report["all_configurations"][model])
            else:
                assert setting["metrics"][task]["corrected"] == setting["metrics"][task]["native_baseline"]
            before = fields(setting["metrics"][task]["native_baseline"], model, task)
            after = fields(setting["metrics"][task]["corrected"], model, task)
            delta = difference(after, before)

            def cells(values, changed=False):
                result = []
                for j, value in enumerate(values):
                    entries = value if j == 3 else [value]
                    text = []
                    for entry in entries:
                        formatted = fmt(entry, changed, mae=j == 2, type_value=j == 3)
                        good = entry is not None and abs(entry) >= 1e-12 and (entry < 0 if j == 2 else entry > 0)
                        text.append(f'**{formatted}**' if changed and good else formatted)
                    result.append(' / '.join(text))
                return result

            lines.append('| ' + ' | '.join([NAMES[model], *cells(before)]) + ' |')
            label = '**+our**' if setting["config_id"] != "native_eta0" else '+our（保留原模型）'
            lines.append('| ' + ' | '.join([label, *cells(delta, True)]) + ' |')
            rows.append(dict(task=task, model=model, mode=setting["mode"], parameters=setting["parameters"],
                             config_id=setting["config_id"], origin=setting.get("origin", "native"), baseline=before, corrected=after, delta=delta))
        lines.append('')
    lines.extend(['参数（SP、IE 可不同）：', '', '| Model | SP：η / α / 位置 | IE：η / α / 位置 |', '|---|---|---|'])
    for model in ORDER:
        text = []
        for task in ("sp", "ie"):
            setting = report["selected"][task][model]
            parameters = setting["parameters"]
            text.append(f'{parameters["eta"]} / {parameters.get("alpha", 1.0)} / {setting["mode"]}')
        lines.append('| ' + ' | '.join([NAMES[model], *text]) + ' |')
    lines.extend(['', '百分比指标的增量单位为百分点，Loc. MAE 使用原单位。正收益加粗，所有下降照实保留。',
                  'SP 类型顺序：突然消失 / 身份改变 / 遮挡后错误重现；IE：异常运动 / 穿透 / 接触响应。',
                  'DINO† 保留原适配器时间错位，仅作诊断。所有选参基于已曝光数据，属于探索性最优结果，不构成独立验证或物理交互修复证明。',
                  '条件 Loc. MAE 的统计成员可能变化；共同成员与起点变化详见 CASEWISE_AND_PAIRED_STATISTICS.json。',
                  '对象模型既定的参考筛选常数及每个设置的完整参数，见 best_parameters.json 与 all_settings.tsv。'])
    with (out / '完整汇总.md').open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    save(out / 'SUMMARY.json', dict(at=time.time(), all_models=9, all_tasks=2, rows=rows,
                                   new_configurations=completeness['configurations'], independent_pairs=600,
                                   source_report_sha256=digest(delivery / 'report.json'),
                                   selection=report['selection'], no_reranking_or_new_model_calls=True))
    extras = ['watch_status_001.py', 'RESUME_AFTER_POWER_001.txt', 'POWER_RESUME_CHECKPOINT_002.json',
              'SLOTFORMER_IE_FIRST_GAIN_CASEWISE_001.json', 'discussion_164_confirmed_001.txt',
              'discussion_165.txt', 'SHARED_DISCUSSION_165.json']
    diagnostics = out / 'diagnostics'
    diagnostics.mkdir()
    for name in extras:
        path = HERE / name
        if path.exists():
            with (diagnostics / name).open('xb') as stream:
                stream.write(path.read_bytes())
    monitor_files = sorted((HERE / 'author_monitor_001').glob('*.json'))
    save(diagnostics / 'continuous_monitor_manifest.json', dict(
        at=time.time(), snapshots=len(monitor_files), files={str(p): digest(p) for p in monitor_files}))
    save(delivery / 'USER_SUMMARY_MANIFEST_001.json', dict(at=time.time(),
         files={str(p): digest(p) for p in out.rglob('*') if p.is_file()},
         core_delivery_sha256=digest(delivery / 'DELIVERY.json'),
         original_report_and_scores_untouched=True, one_actual_configuration_per_row=True))
    print(out, flush=True)


if __name__ == '__main__':
    main()
