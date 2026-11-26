"""Review completed search and original-pipeline validation; no model calls."""
import time
from pathlib import Path
from . import common as c


def main():
    assert c.read(c.HERE / "FINAL_STATE_001.json")["all_lineage_audits_complete"]
    assert c.read(c.HERE / "PERSISTED.json")["all_experiments_and_audits_complete"]
    c.verify(full=True)
    for path, wanted in c.read(c.HERE / "RUNTIME_ADDENDUM_002.json")["files"].items():
        assert c.digest(path) == wanted
    if (c.HERE/"VALIDATION_EXTENSION_003.json").exists():
        for path,wanted in c.read(c.HERE/"VALIDATION_EXTENSION_003.json")["files"].items():
            assert c.digest(path)==wanted
    if (c.HERE/"REPORT_ROUTING_ADDENDUM_004.json").exists():
        for path,wanted in c.read(c.HERE/"REPORT_ROUTING_ADDENDUM_004.json")["files"].items():
            assert c.digest(path)==wanted
    d=Path(c.read(c.HERE / "FINAL_DELIVERY.json")["directory"])
    checked={}
    for name in ("DELIVERY.json","SUPPLEMENT_MANIFEST_001.json","USER_SUMMARY_MANIFEST_001.json","METRIC_ALTERNATIVES_MANIFEST_001.json"):
        manifest=c.read(d/name)
        for path,wanted in manifest["files"].items():
            assert c.digest(path)==wanted,path
        checked[str(d/name)]=c.digest(d/name)
    r=c.read(d/"report.json")
    parent=c.read(c.source_path())
    for model,rows in parent["all_configurations"].items():
        assert r["all_configurations"][model][:len(rows)]==rows
    expected=c.read(c.HERE/"GRID.json")["settings"]+c.read(c.HERE/"STAGE2.json")["settings"]
    assert r["completed_round_configurations"]==expected==len(r["current_round_settings"])
    assert r["completed_corrected_model_configs"]==parent["completed_corrected_model_configs"]+expected
    native_hashes={}
    for x in r["current_round_settings"]:
        assert x["complete"] and x["completed_pairs"]==600
        assert not x["eligible_for_selection"] or x["full600_native_validated"]
        if x.get("confirmation_only",False):
            assert not x["eligible_for_selection"]
        if x.get("full600_native_validated"):
            files=list((Path(x["directory"])/"pairs").glob("*.json"))
            assert len(files)==600
            for path in files:
                row=c.read(path)
                assert row["trace_matches_original"] and row["extra_forwards"]==0
                assert row["weight_identity_and_version_unchanged"]
                native_hashes[str(path)]=c.digest(path)
    summary=c.read(d/"user_summary_001/SUMMARY.json")
    assert summary["all_models"]==9 and summary["all_tasks"]==2
    for task in ("sp","ie"):
        assert len(r["selected"][task])==9
        for model,x in r["selected"][task].items():
            if x["config_id"].startswith("R_"):
                assert x["full600_native_validated"] and x["eligible_for_selection"]
                actual=[v for v in r["all_configurations"][model] if v==x]
                assert actual
    validation_path=c.HERE/"VALIDATION_GRID_EXPANDED_003.json"
    if not validation_path.exists():
        validation_path=c.HERE/"VALIDATION_GRID.json"
    expected_native_ids={x["config_id"] for x in c.read(validation_path)["native_new"]+c.read(c.HERE/"GRID.json")["native_new"]}
    actual_native_ids={x["config_id"] for x in r["current_round_settings"] if x.get("full600_native_validated")}
    assert actual_native_ids==expected_native_ids,(actual_native_ids,expected_native_ids)
    counts=dict(search_configurations=expected,
        original_full600_validated_new_configurations=sum(x.get("full600_native_validated",False) for x in r["current_round_settings"]),
        original_full600_neural_validation_runs=c.read(validation_path)["settings"]+len(c.read(c.HERE/"GRID.json")["native_new"]),
        known_configuration_confirmations=sum(x.get("confirmation_only",False) for x in r["current_round_settings"]),
        independent_pairs=600,configuration_case_evaluations=600*expected,
        cumulative_retained_versions=r["completed_corrected_model_configs"])
    c.save(d/"NATIVE_RECORD_HASHES_001.json",dict(at=time.time(),files=native_hashes))
    c.save(d/"FINAL_REVIEW_001.json",dict(at=time.time(),counts=counts,verified_manifests=checked,
        native_record_hash_manifest_sha256=c.digest(d/"NATIVE_RECORD_HASHES_001.json"),
        original_sources_unchanged=True,all_parent_versions_retained=True,
        partial_graph_search_and_full_neural_forecasts_distinguished=True,
        every_new_main_row_original_full600_validated=True,one_actual_setting_per_main_row=True,
        same_exposed_data_exploratory_selection=True))
    with (d/"实验范围说明.md").open("x") as stream:
        stream.write(f'本轮完成{expected}组原算子参数搜索，每组原SP300对+IE300对。原完整流程另外验证了{counts["original_full600_neural_validation_runs"]}个参数配置；重复验证不增加独立样本。\n\n')
        stream.write('离线搜索复用与η、半径倍率无关的原模型前缀，串行重放原后缀，不是NN0。所有新入选主表设置均已通过原完整600对流程、原调用序列和预算检查，且逐例误差与搜索完全一致；未完成原完整推理的搜索点保留并标为不可入选。\n\n')
        stream.write('欧氏投影、原历史、原权重、原指标、原数据及正式推理前向次数均不变。所有下降与零收益保留；同一主表行使用一套真实参数。所有选参基于已曝光数据，属于探索性结果，DINO仅作诊断。条件MAE需检查成员变化和共同成员误差。\n')
    c.save(d/"FINAL_REVIEW_MANIFEST_001.json",dict(at=time.time(),files={str(p):c.digest(p) for p in
        (d/"NATIVE_RECORD_HASHES_001.json",d/"FINAL_REVIEW_001.json",d/"实验范围说明.md")},
        source_reviewer_sha256=c.digest(Path(__file__)),core_reports_and_scores_unchanged=True))
    print(counts,flush=True)


if __name__=="__main__":
    main()
