"""Show both fully validated VideoMAE IE settings without mixing metrics."""
import time
from pathlib import Path
from . import common as c


def main():
    assert c.read(c.HERE/"FINAL_STATE_001.json")["all_lineage_audits_complete"]
    d=Path(c.read(c.HERE/"FINAL_DELIVERY.json")["directory"])
    report=c.read(d/"report.json")
    primary=report["selected"]["ie"]["videomaev2"]
    identifier="R_videomaev2_pre_final_block_e0.75_a11.997855_570a391c5728"
    locator=next(x for x in report["all_configurations"]["videomaev2"] if x["config_id"]==identifier)
    assert primary["full600_native_validated"] and locator["full600_native_validated"]
    casewise=c.read(c.HERE/"CANDIDATE_CASEWISE_SNAPSHOT_001.json")["results"]["videomaev2:ie"]
    assert casewise["config_id"]==identifier and casewise["same_MAE_members"]
    assert casewise["native_MAE_n"]==casewise["candidate_MAE_n"]==145
    assert not casewise["PA_gain"] and not casewise["PA_loss"] and not casewise["JH_gain"] and not casewise["JH_loss"]
    assert casewise["onset_improved"]==[dict(key="ie_003",category="motion_anomaly",native_error=2.,candidate_error=0.)]
    assert not casewise["onset_worsened"]
    native=primary["metrics"]["ie"]["native_baseline"]
    lines=["VideoMAE IE 的两套完整实测参数", "",
        "两套参数均通过原完整流程SP300+IE300验证。主排序规则不变，每一行的全部指标来自同一个参数配置。", "",
        "| 设置 | η | 半径倍率 | PA % | AUC % | Loc. MAE（成员数） | Coverage % | Joint Hit by Type % | Overall JH % |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    rows=[("原模型",None,native),("原主排序选择",primary,primary["metrics"]["ie"]["corrected"]),
          ("另列定位设置",locator,locator["metrics"]["ie"]["corrected"])]
    for label,setting,metrics in rows:
        o=metrics["overall"]
        pp=setting["parameters"] if setting else dict(eta=0.,alpha=1.)
        types=[100*next(iter(metrics["by_type"][key]["joint_hit"].values())) for key in ("motion_anomaly","penetration","contact_response")]
        lines.append(f'| {label} | {pp["eta"]} | {pp["alpha"]} | {100*o["paired_accuracy"]:.4f} | {100*o["auc"]:.6f} | {o["localization_mae"]:.9f} ({o["localization_mae_denominator"]}) | {100*o["detection_coverage"]:.4f} | '+" / ".join(f"{v:.0f}" for v in types)+f' | {100*next(iter(o["joint_hit"].values())):.4f} |')
    lines += ["", "IE类型顺序：异常运动 / 穿透 / 接触响应。定位设置的完整指标单独列出，不与主排序设置的AUC拼接。",
              "定位设置的145个统计成员与原模型完全相同，ie_003异常运动的定位误差从2帧降到0帧，其余共同成员不变；PA/JH集合未改变。此项由原评分及逐例完整流程比对核验。",
              "全部属于已曝光数据上的探索性选参，不是独立验证或物理交互修复证明。"]
    path=d/"VideoMAE_IE_两套参数的完整结果.md"
    with path.open("x") as stream:
        stream.write("\n".join(lines)+"\n")
    c.save(d/"VIDEO_LOCALIZATION_REPORT_MANIFEST_001.json",dict(at=time.time(),files={str(path):c.digest(path)},
        source_report_sha256=c.digest(d/"report.json"),source_exporter_sha256=c.digest(Path(__file__)),
        no_scores_reranked_or_mixed=True))
    print(path,flush=True)


if __name__=="__main__":
    main()
