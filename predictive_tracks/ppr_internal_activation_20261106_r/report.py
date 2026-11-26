"""Original metrics, fixed ranking and full disclosure of partial-graph search."""
import copy
import time
from pathlib import Path
from . import common as c
from predictive_tracks.ppr_taskwise_best_20261102_001.select_report import ORDER, ENDPOINTS, rank, endpoint


def parameters(item):
    return dict(eta=item["eta"], alpha=item["alpha"], q=None, presence=None, mass=None, min_history=None)


def search_rows():
    rows = []
    for model in c.FOCUS:
        for item in c.all_tail_jobs(model):
            folder = c.HERE / "search_settings" / model / item["config_id"]
            if not (folder / "summary.json").exists():
                continue
            summary = c.read(folder / "summary.json")
            validation = c.directory(item)
            measured = (validation / "summary.json").exists()
            if measured:
                full = c.read(validation / "summary.json")
                assert full["metrics"] == summary["metrics"], item["config_id"]
            rows.append(dict(item, parameters=parameters(item), complete=True, completed_pairs=600,
                metrics=summary["metrics"], compact_case_source=summary["compact_case_source"],
                directory=str(validation) if measured else str(folder), source_directory=str(folder),
                partial_graph_parameter_search=True, full600_native_validated=measured,
                eligible_for_selection=measured and not item.get("confirmation_only", False),
                eligible_tasks=["sp", "ie"], origin="R_internal_activation_scale"))
    return rows


def native_rows():
    result = []
    for item in c.read(c.HERE / "GRID.json")["native_new"]:
        path = c.directory(item)
        if not (path / "summary.json").exists():
            continue
        summary = c.read(path / "summary.json")
        assert summary["complete"] and summary["cases"] == 600
        result.append(dict(item, parameters=parameters(item), metrics=summary["metrics"],
            complete=True, completed_pairs=600, directory=str(path), source_directory=str(path),
            full600_native_validated=True, eligible_for_selection=True, eligible_tasks=["sp", "ie"],
            origin="R_existing_state_only_radius"))
    return result


def native(model, row):
    return dict(model=model, mode="native", config_id="native_eta0", parameters=parameters(dict(eta=0., alpha=1.)),
        metrics={t:dict(native_baseline=row["metrics"][t]["native_baseline"], corrected=row["metrics"][t]["native_baseline"]) for t in ("sp", "ie")})


def assemble():
    parent = c.read(c.source_path())
    rows = search_rows()+native_rows()
    r = dict(at=time.time(), source_report=str(c.source_path()), source_report_sha256=c.digest(c.source_path()),
        all_configurations=copy.deepcopy(parent["all_configurations"]), selected={t:{} for t in ("sp", "ie")},
        metric_specific_maxima={t:{} for t in ("sp", "ie")}, previous_selected=parent["selected"], current_round_settings=rows,
        completed_round_configurations=len(rows), completed_corrected_model_configs=0,
        selection=parent["selection"], scope="Same exposed SP300/IE300; exploratory parameter search. Partial-graph search is declared and only original full600-validated new settings enter the main table.",
        dino=parent.get("dino"), extra_neural_forwards_per_forecast=0,
        radius_multiplier_search=[1., max(j["alpha"] for m in c.FOCUS for j in c.all_tail_jobs(m))])
    for row in rows:
        r["all_configurations"][row["model"]].append(row)
    r["completed_corrected_model_configs"] = sum(map(len, r["all_configurations"].values()))
    for model in ORDER:
        allrows = r["all_configurations"][model]
        baseline = native(model, allrows[0])
        for task in ("sp", "ie"):
            candidates = [x for x in allrows if x.get("eligible_for_selection", True) and task in x.get("eligible_tasks", ["sp", "ie"])]+[baseline]
            assert all(x["metrics"][task]["native_baseline"] == baseline["metrics"][task]["native_baseline"] for x in candidates)
            r["selected"][task][model] = copy.deepcopy(min(candidates, key=lambda x:rank(x, task)))
            r["metric_specific_maxima"][task][model] = {}
            for key in ENDPOINTS:
                best = min(candidates, key=lambda x:rank(x, task, key))
                bv, value = endpoint(baseline, task, key), endpoint(best, task, key)
                r["metric_specific_maxima"][task][model][key] = dict(config_id=best["config_id"], mode=best["mode"], parameters=best["parameters"],
                    baseline=bv, best_value=value, change=None if value is None or bv is None else value-bv,
                    all_metrics_at_this_setting=best["metrics"][task])
    return r


def interim():
    r = assemble()
    dest = c.HERE / "reports" / str(time.time_ns())
    c.save(dest / "report.json", r)
    c.status(c.HERE / "monitor/latest_report.json", dict(at=time.time(), path=str(dest), completed=r["completed_round_configurations"]))
    print("R COMPLETE SETTINGS", r["completed_round_configurations"], flush=True)


def make_stage2():
    if (c.HERE / "STAGE2.json").exists():
        return
    parent = c.read(c.source_path())
    rows = search_rows()
    existing = {(m, x["mode"], x["parameters"]["eta"], x["parameters"].get("alpha", 1.))
                for m, values in parent["all_configurations"].items() for x in values}
    existing.update((x["model"], x["mode"], x["eta"], x["alpha"]) for x in rows)
    jobs = []
    for model, task in (("vjepa2_vitl", "sp"), ("videomaev2", "ie")):
        candidates = [x for x in rows if x["model"] == model and not x.get("confirmation_only", False)]
        anchors = [min(candidates, key=lambda x:rank(x, task)), min(candidates, key=lambda x:rank(x, task, "auc"))]
        es = sorted({x["eta"] for x in candidates})
        alphas = sorted({x["alpha"] for x in candidates})
        for anchor in anchors:
            eta, alpha = anchor["eta"], anchor["alpha"]
            ei, ai = es.index(eta), alphas.index(alpha)
            ev = [round((eta+es[k])/2,8) for k in (ei-1,ei+1) if 0 <= k < len(es)]
            av = [round((alpha+alphas[k])/2,8) for k in (ai-1,ai+1) if 0 <= k < len(alphas)]
            for e, a in [(e,alpha) for e in ev]+[(eta,a) for a in av]+[(e,a) for e in ev for a in av]:
                key = (model, "pre_final_block", e, a)
                if key in existing:
                    continue
                existing.add(key)
                jobs.append(c.job(model, "pre_final_block", e, a, confirmation_only=False,
                                  proposal="midpoint refinement around global actual primary/AUC grid anchors", anchor_id=anchor["config_id"], target_analysis_task=task))
    grid = dict(at=time.time(), tail=jobs, native_new=[], cached=[], native_reuse=[], settings=len(jobs),
        no_per_case_or_per_type_parameters=True, history_distance_and_formula_unchanged=True)
    c.save(c.HERE / "STAGE2.json", grid)
    c.save(c.HERE / "STAGE2_FROZEN.json", dict(at=time.time(), sha256=c.digest(c.HERE / "STAGE2.json")))


def make_validation():
    if (c.HERE / "VALIDATION_GRID.json").exists():
        return
    parent = c.read(c.source_path())
    rows = search_rows()
    wanted = {}
    for model in c.FOCUS:
        for task in ("sp", "ie"):
            best = min([parent["selected"][task][model]]+[x for x in rows if x["model"] == model and not x.get("confirmation_only",False)], key=lambda x:rank(x,task))
            if best["config_id"].startswith("R_"):
                wanted[best["config_id"]] = {key:best[key] for key in ("model", "mode", "eta", "alpha", "quality", "target_task", "config_id")}
    # At least one original full600 check of the best searched point for each
    # focus model, even when none beats the previous main-table selection.
    for model, task in (("vjepa2_vitl", "sp"), ("videomaev2", "ie")):
        if not any(x["model"] == model for x in wanted.values()):
            best = min([x for x in rows if x["model"] == model and not x.get("confirmation_only",False)], key=lambda x:rank(x,task))
            wanted[best["config_id"]] = {key:best[key] for key in ("model", "mode", "eta", "alpha", "quality", "target_task", "config_id")}
    c.save(c.HERE / "VALIDATION_GRID.json", dict(at=time.time(), native_new=list(wanted.values()), cached=[], native_reuse=[],
        selection="Global main-table rank per model/task; fallback best searched point for each focus model; no per-case selection", settings=len(wanted)))
    c.save(c.HERE / "VALIDATION_FROZEN.json", dict(at=time.time(), sha256=c.digest(c.HERE / "VALIDATION_GRID.json")))


def finish():
    import sys
    sys.path.insert(0, str(c.ROOT / "predictive_tracks/sp/code"))
    from predictive_tracks.ppr_existing_20261102.metrics import make_summary
    from predictive_tracks.ppr_eta_radius_20261104_o import report as old
    from .tail_worker import expand
    c.install()
    r = assemble()
    expected = c.read(c.HERE / "GRID.json")["settings"]+c.read(c.HERE / "STAGE2.json")["settings"]
    assert len(r["current_round_settings"]) == expected
    for model in c.FOCUS:
        audits = list((c.HERE / "full_pipeline_preflight" / model).glob("*.json"))
        assert len(audits) == 12
        assert all(c.read(path)["original_trace_exact"] and c.read(path)["full_vs_tail_max_abs_error"] == 0 for path in audits)
    d = c.HERE / "delivery_20261106_001"
    if d.exists():
        if (d / "DELIVERY.json").exists():
            return
        index = 2
        while (c.HERE / f"delivery_20261106_{index:03d}").exists():
            index += 1
        d = c.HERE / f"delivery_20261106_{index:03d}"
    d.mkdir()
    payload_cache = {}
    def records(x):
        if x.get("partial_graph_parameter_search"):
            if x.get("full600_native_validated"):
                return [c.read(p) for p in sorted((Path(x["directory"]) / "pairs").glob("*.json"))]
            folder = x["compact_case_source"]
            if folder not in payload_cache:
                payload_cache[folder] = [c.read(p) for p in sorted(Path(folder).glob("*.json"))]
            return [expand(p,x) for p in payload_cache[folder]]
        return old.records(x)
    details, hashes = [], {}
    for model in c.FOCUS:
        for phase in ("tail", "tail_stage2"):
            root = c.HERE / phase / model / "pairs"
            for p in sorted(root.glob("*.json")):
                v = c.read(p)
                assert v["original_native_prediction_bitwise_equal"] and v["original_native_errors_exact"] and v["original_suffix_output_bitwise_equal"]
                for name, wanted in v["confirmed_full_pipeline_settings"].items():
                    job = next(j for j in c.all_tail_jobs(model) if j["config_id"] == name)
                    source = Path(job["confirmation_source"]) / "pairs" / p.name
                    assert c.digest(source) == wanted
                hashes[str(p)] = c.digest(p)
                cm = Path(v["prefix_meta"])
                assert c.digest(cm) == v["prefix_meta_sha256"]
                meta = c.read(cm)
                tensor_file = cm.with_suffix(".pt")
                if str(tensor_file) not in hashes:
                    assert c.digest(tensor_file) == meta["sha256"]
                    assert c.digest(v["source_meta"]) == v["source_meta_sha256"]
                    original_meta = c.read(v["source_meta"])
                    assert c.digest(original_meta["state"]) == v["source_state_sha256"]
                    hashes[str(tensor_file)] = meta["sha256"]
                    hashes[str(original_meta["state"])] = v["source_state_sha256"]
    for i, x in enumerate(r["current_round_settings"],1):
        rr = records(x)
        measured = make_summary(rr, x["model"])
        assert measured["metrics"] == x["metrics"]
        if x.get("full600_native_validated"):
            for rec in rr:
                assert rec["trace_matches_original"] and rec["extra_forwards"] == 0 and rec["weight_identity_and_version_unchanged"]
                assert rec["eta"] == x["eta"] and rec["radius_multiplier"] == x["alpha"]
            if x.get("partial_graph_parameter_search"):
                pp = [c.read(p) for p in sorted(Path(x["compact_case_source"]).glob("*.json"))]
                replay = {f'{p["native"]["benchmark"]}_{p["native"]["archive_index"]:03d}':p for p in pp}
                for rec in rr:
                    expected = replay[f'{rec["benchmark"]}_{rec["archive_index"]:03d}']["settings"][x["config_id"]]
                    assert rec["possible_errors"] == expected["possible_errors"] and rec["impossible_errors"] == expected["impossible_errors"]
        details.append(dict(config_id=x["config_id"], cases=600, original_metrics_exact=True,
                            full600_native_validated=x.get("full600_native_validated",False), confirmation_only=x.get("confirmation_only",False)))
        if i % 25 == 0:
            print("R FINAL AUDIT", i, "/", len(r["current_round_settings"]), flush=True)
    stats = {}
    for task in ("sp", "ie"):
        for model, x in r["selected"][task].items():
            if x["config_id"] == "native_eta0":
                continue
            key = model+":"+x["config_id"]
            if key not in stats:
                stats[key] = old.casewise(records(x),model)
    c.save(d / "report.json", r)
    c.save(d / "COMPLETENESS_AUDIT.json", dict(at=time.time(), configurations=len(details), settings=details,
        independent_pairs=600, case_records=600*len(details), all_original_metrics_recomputed=True,
        partial_graph_search_disclosed=True, every_new_main_row_original_full600_validated=True))
    c.save(d / "CASEWISE_AND_PAIRED_STATISTICS.json", stats)
    c.save(d / "input_hashes.json", hashes)
    c.save(d / "RESOURCE_PEAKS.json", c.read(c.HERE / "monitor/resource_peaks.json"))
    old.render(r,d)
    with (d / "all_settings.tsv").open("x") as stream:
        stream.write("model\ttask\tmode\teta\talpha\tPA_percent\tAUC_percent\tMAE\tJH_percent\teligible\tfull600_native_validated\torigin\tid\n")
        for model, rows in r["all_configurations"].items():
            for row in rows:
                for task in ("sp", "ie"):
                    pp = row["parameters"]
                    values = [model,task,row["mode"],pp["eta"],pp.get("alpha",1.),100*endpoint(row,task,"paired_accuracy"),
                        100*endpoint(row,task,"auc"),endpoint(row,task,"localization_mae"),100*endpoint(row,task,"overall_joint_hit"),
                        row.get("eligible_for_selection",True) and task in row.get("eligible_tasks",["sp","ie"]),
                        row.get("full600_native_validated", "inherited_audited_parent"),row.get("origin"),row["config_id"]]
                    stream.write("\t".join(map(str,values))+"\n")
    with (d / "READ_ME.txt").open("x") as stream:
        stream.write(r["scope"]+"\nOnly eta/alpha tuned; original Euclidean metric, history, original weights, batch1/float32 device/expression and deployed neural budget preserved. Offline search reuses original parameter-independent prefix and the unchanged suffix; this is not NN0. Raw partial-graph results and original full600 validation are distinguished in all_settings.tsv. No unvalidated new setting enters the main table. DINO is diagnostic only. Conditional MAE includes changing membership; common-member checks saved.\n")
    c.verify(full=True)
    c.save(d / "DELIVERY.json", dict(at=time.time(), files={str(p):c.digest(p) for p in d.iterdir() if p.is_file()}, all_checks_passed=True))
    c.save(c.HERE / "FINAL_DELIVERY.json", dict(at=time.time(), directory=str(d), complete=True))
    print("R FINAL DELIVERY", d, flush=True)
