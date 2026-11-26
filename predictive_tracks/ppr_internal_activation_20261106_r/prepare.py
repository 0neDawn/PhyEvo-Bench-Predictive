"""Choose internal radii from existing native history/prediction audits only."""

from release_paths import project_path as _release_path
import collections
import shutil
import time
from pathlib import Path
from . import common as c


def quantile(values, probability):
    values = sorted(values)
    position = (len(values)-1)*probability
    lo = int(position)
    hi = min(lo+1, len(values)-1)
    return values[lo] if hi == lo else values[lo]*(hi-position)+values[hi]*(position-lo)


def main():
    assert c.read(c.PARENT / "FINAL_STATE_001.json")["all_lineage_audits_complete"]
    source = c.read(c.source_path())
    c.save(c.HERE / "PARENT_MANIFESTS.json", {str(p): c.digest(p) for p in (
        c.source_path().parent / name for name in ("DELIVERY.json", "SUPPLEMENT_MANIFEST_001.json", "USER_SUMMARY_MANIFEST_001.json"))})
    parent_annotation = c.ROOT / "predictive_tracks/ppr_joint_fine_20261105_p/annotation.json"
    a = c.read(parent_annotation)
    st = Path(a["rollout_path"]).stat()
    assert a["task_complete_verified"] and a["turn_status"] == "completed"
    assert st.st_size == a["rollout_size"] and st.st_mtime_ns == a["rollout_mtime_ns"]
    c.save(c.HERE / "annotation.json", a)
    all_jobs, analyses, confirmations = [], {}, {}
    existing = {(m, x["mode"], x["parameters"]["eta"], x["parameters"].get("alpha", 1.))
                for m, rows in source["all_configurations"].items() for x in rows}
    etas = [.00005, .00025, .001, .003, .01, .03, .075, .15, .3, .5, .75, 1.]
    constants = {"vjepa2_vitl": [1, 2, 4, 4.5, 5, 5.5, 6, 6.5, 6.8, 7, 7.25, 7.5],
                 "videomaev2": [1, 2, 4, 8, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.25, 12.5, 13]}
    for model in c.FOCUS:
        previous = [x for x in source["all_configurations"][model] if x["mode"] == "pre_final_block"]
        old = next(x for x in previous if x["parameters"]["eta"] == .1 and x["parameters"].get("alpha", 1.) == 1.)
        folder = Path(old.get("directory") or old["source_directory"]) / "pairs"
        ratios, files = [], {}
        assert len(list(folder.glob("*.json"))) == 600
        for path in sorted(folder.glob("*.json")):
            row = c.read(path)
            audit = row["projection_audits"][0]
            radius = audit.get("base_radii", audit["radii"])[0]
            assert radius > 0 and len(audit["steps"]) == 3
            for step in audit["steps"]:
                assert step["units"] == 1
                ratios.append(1.+step["change_l2"]/radius)
            files[str(path)] = c.digest(path)
        quantiles = {str(q): quantile(ratios, q) for q in [.0, .1, .25, .5, .75, .9, 1.]}
        alphas = sorted({round(float(v), 8) for v in constants[model]+[quantiles[str(q)] for q in [.1, .25, .5, .75, .9]]})
        jobs = {(eta, alpha): c.job(model, "pre_final_block", eta, alpha,
                    proposal="existing internal operator, activation-scale eta/radius grid",
                    confirmation_only=(model, "pre_final_block", eta, alpha) in existing)
                for eta in etas for alpha in alphas}
        confirmations[model] = []
        for x in previous:
            pp = x["parameters"]
            eta, alpha = pp["eta"], pp.get("alpha", 1.)
            key = (eta, alpha)
            if key not in jobs:
                jobs[key] = c.job(model, "pre_final_block", eta, alpha, confirmation_only=True,
                                  proposal="complete original full600 parameter confirmation")
            jobs[key]["confirmation_source"] = x.get("directory") or x["source_directory"]
            confirmations[model].append(dict(config_id=jobs[key]["config_id"], source=jobs[key]["confirmation_source"],
                                             summary_sha256=x.get("summary_sha256")))
        all_jobs.extend(jobs.values())
        analyses[model] = dict(quantiles=quantiles, alphas=alphas, source_directory=str(folder),
            files=files, ratios_use_only_history_and_raw_prediction=True,
            distance_over_radius_from_original_projection_audit=True)
    native = []
    for model in ("ocvp_seq", "savi_transformer"):
        # The same previously authorized state-only insertion, no new location.
        for alpha in (1.5, 2.5, 3.):
            for eta in (.25, .75):
                if (model, "state_only", eta, alpha) not in existing:
                    native.append(c.job(model, "state_only", eta, alpha, proposal="existing state-only insertion radius neighborhood"))
    grid = dict(at=time.time(), tail=all_jobs, native_new=native, cached=[], native_reuse=[],
        settings=len(all_jobs)+len(native), confirmations=confirmations,
        original_parameters_only=["eta", "alpha"], no_extra_forward_in_deployed_forecast=True,
        search_acceleration="Parameter-independent original prefix captured once; exact original suffix replayed serially at batch1. Final selected new settings receive original full600 validation.",
        data_pairs_per_configuration=600, independent_pairs=600,
        scope="Same exposed SP300/IE300; adaptive global parameter search, not independent validation.")
    c.save(c.HERE / "INTERNAL_ACTIVATION_ANALYSIS.json", analyses)
    c.save(c.HERE / "GRID.json", grid)
    c.save(c.HERE / "PLAN.json", dict(at=time.time(), parent=str(c.source_path()), parent_sha256=c.digest(c.source_path()),
        user_constraints="Old files, data, metrics, weights, Euclidean projection and per-forecast neural budget unchanged. Only eta/radius and already authorized positions.",
        inference_formula="raw + eta*(Pi_B(center,alpha*R)(raw)-raw); eta0/raw and eta1/hard retain original branches",
        prefix_capture_is_search_only=True, original_batch_and_precision=True,
        final_validation="At most one new primary best for each model/task, original complete pipeline, all600, same call trace. No partial-graph result silently claimed as a full neural forecast.",
        other_existing_choices="Only already authorized insertion modes. Lambda is eta/(1-eta), not independent. History window, distance, representation, visibility thresholds and weights fixed."))
    with (c.HERE / "PLAN.txt").open("x") as f:
        f.write("R: Existing Euclidean state correction, eta/radius activation-scale search. Original history and all600 pairs. Search reuses parameter-independent prefix; original full600 verifies selected settings. All outputs/negative settings retained.\n")
    paths = list(c.HERE.glob("*.py"))+[c.HERE/n for n in ("GRID.json", "PLAN.txt", "PLAN.json", "INTERNAL_ACTIVATION_ANALYSIS.json", "PARENT_MANIFESTS.json")]
    old_files = [c.OLD/n for n in ("native_worker.py", "hooks.py", "report.py", "common.py")]
    old_files += [c.ROOT/"predictive_tracks"/n for n in ("ppr_eta_radius_20261104_n/hooks.py", "ppr_state_insert_20261104_m/hooks.py", "ppr_all_20261029/projection.py", "ppr_20261028/projection.py", "ppr_existing_20261102/metrics.py", "controlled_tracks/full_metrics_report.py", "ppr_taskwise_best_20261102_001/select_report.py")]
    old_files += [c.ROOT/"predictive_tracks/sp/code"/n for n in ("sp_vjepa2_vitl/adapter.py", "sp_vjepa2_vith/adapter.py", "sp_videomae/adapter.py", "sp_benchmark/protocol_metrics.py")]
    old_files += [Path(_release_path('third_party/vjepa2/src/models/predictor.py'))]
    for path, wanted in c.read(c.OLD / "frozen.json")["files"].items():
        if path.endswith(".py"):
            assert c.digest(path) == wanted, path
            old_files.append(Path(path))
    old_files = list(dict.fromkeys(old_files))
    c.save(c.HERE / "frozen.json", dict(at=time.time(), files={str(p):c.digest(p) for p in paths+old_files}))
    c.verify(full=True)
    print({"tail":len(all_jobs), "native_other_IE":len(native), "models":dict(collections.Counter(j["model"] for j in all_jobs))}, flush=True)


if __name__ == "__main__":
    main()
