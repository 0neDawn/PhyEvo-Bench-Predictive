"""Read-only supplementary export: show each endpoint's actual trade-offs."""
import json
import time
from pathlib import Path
from . import common as c


def main():
    assert c.read(c.HERE / "FINAL_STATE_001.json")["all_lineage_audits_complete"]
    delivery = Path(c.read(c.HERE / "FINAL_DELIVERY.json")["directory"])
    report = c.read(delivery / "report.json")
    from predictive_tracks.ppr_taskwise_best_20261102_001.select_report import ORDER, NAMES
    out = delivery / "metric_alternatives_001"
    out.mkdir(exist_ok=False)
    rows = []
    lines = ["以下是各指标对应的独立实测设置，不拼成一个方法的成绩行。", "",
             "定位误差为条件均值，须同时检查其统计成员数、覆盖率、PA与JH的变化；更低MAE不自动表示整体改善。", "",
             "同一曝光数据上的探索性选参，所有设置及下降仍保留。", ""]
    endpoints = [("overall_joint_hit", "Overall JH"), ("paired_accuracy", "PA"), ("auc", "AUC"), ("localization_mae", "MAE")]
    for task in ("sp", "ie"):
        lines += [task.upper(), "", "| 模型 | 优先指标 | η / α / 位置 | PA % | AUC % | MAE (成员数) | Coverage % | JH % |",
                  "|---|---|---|---:|---:|---:|---:|---:|"]
        for model in ORDER:
            for endpoint, label in endpoints:
                x = report["metric_specific_maxima"][task][model][endpoint]
                if x["config_id"] != "native_eta0":
                    actual = [v for v in report["all_configurations"][model] if v["config_id"] == x["config_id"]]
                    assert any(v["parameters"] == x["parameters"] and v["metrics"][task] == x["all_metrics_at_this_setting"] for v in actual)
                both = x["all_metrics_at_this_setting"]
                z = both["corrected"]["overall"]
                assert z["num_pairs"] == 300 and len(z["joint_hit"]) == 1
                pa, auc, cov, jh = [100*v for v in (z["paired_accuracy"], z["auc"], z["detection_coverage"], next(iter(z["joint_hit"].values()))) ]
                mae = "NA" if z["localization_mae"] is None else f'{z["localization_mae"]:.6f}'
                pp = x["parameters"]
                lines.append(f'| {NAMES[model]} | {label} | {pp["eta"]} / {pp["alpha"]} / {x["mode"]} {pp.get("position", "")} | {pa:.4f} | {auc:.4f} | {mae} ({z["localization_mae_denominator"]}) | {cov:.4f} | {jh:.4f} |')
                rows.append(dict(task=task, model=model, priority_endpoint=endpoint, **x))
        lines.append("")
    (out / "各指标独立设置与代价.md").write_text("\n".join(lines)+"\n")
    c.save(out / "rows.json", dict(at=time.time(), source_report_sha256=c.digest(delivery / "report.json"),
        rows=rows, no_new_scores_or_selection=True, conditional_MAE_denominator_included=True))
    c.save(delivery / "METRIC_ALTERNATIVES_MANIFEST_001.json", dict(at=time.time(),
        files={str(p): c.digest(p) for p in out.iterdir()}, source_exporter_sha256=c.digest(Path(__file__)),
        source_report_sha256=c.digest(delivery / "report.json"), one_actual_setting_per_row=True))
    print(out, flush=True)


if __name__ == "__main__":
    main()
