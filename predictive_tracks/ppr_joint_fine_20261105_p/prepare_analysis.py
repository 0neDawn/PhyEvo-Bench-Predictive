"""Freeze a new read-only proposal stage, retaining every previous result."""
import json
import time
from pathlib import Path
from . import common as c


def main():
    from predictive_tracks.ppr_eta_radius_20261104_o.common import verify as original_verify
    original_verify(full=False)
    assert c.read(c.OLD / "FINAL_STATE_001.json")["all_lineage_audits_complete"]
    assert c.read(c.OLD / "PERSISTED.json")["all_experiments_and_audits_complete"]
    delivery = c.OLD / "delivery_20261104_001"
    for manifest in ("DELIVERY.json", "SUPPLEMENT_MANIFEST_001.json", "USER_SUMMARY_MANIFEST_001.json"):
        for path, wanted in c.read(delivery / manifest)["files"].items():
            assert c.digest(path) == wanted, path
    annotation = c.read(c.OLD / "annotation.json")
    assert annotation["task_complete_verified"] and annotation["turn_status"] == "completed"
    stat = Path(annotation["rollout_path"]).stat()
    assert stat.st_size == annotation["rollout_size"] and stat.st_mtime_ns == annotation["rollout_mtime_ns"]
    # The app was freshly queried in this turn: same latest completed turn,
    # error=null. This record does not infer completion from resource idleness.
    c.save(c.HERE / "PRIORITY_APP_VERIFIED_001.json", dict(at=time.time(), tool="codex_app.read_thread", thread_id="019f7e4b-0e84-7991-8dd6-4a9a2fba7e27",
        host_id="remote-ssh-discovered:codex-server", latest_turn="01a18ac6-b945-7ea1-8f19-11c3b62c4d1b", status="completed", error=None,
        rollout_size=stat.st_size, rollout_mtime_ns=stat.st_mtime_ns))
    c.save(c.HERE / "annotation.json", annotation)
    names = ("native_run_runtime_004.txt", "native_execute_runtime_004.txt", "PARAMETER_AUDIT_RUNTIME_ADDENDUM_003.json", "PARAMETER_AUDIT_RUNTIME_ADDENDUM_004.json")
    copied = {}
    for name in names:
        dest = c.HERE / name
        with dest.open("xb") as stream:
            stream.write((c.OLD / name).read_bytes())
        copied[str(dest)] = dict(original=str(c.OLD / name), sha256=c.digest(dest))
    c.save(c.HERE / "COPIED_ORIGINAL_RUNTIME.json", copied)
    files = [c.HERE / name for name in ("__init__.py", "common.py", "analyze.py", "PLAN.txt", "prepare_analysis.py")]
    files += [c.SOURCE, c.OLD / "frozen.json", c.OLD / "FINAL_STATE_001.json", delivery / "DELIVERY.json", delivery / "SUPPLEMENT_MANIFEST_001.json"]
    files += [c.ROOT / "predictive_tracks" / name for name in (
        "ppr_alpha_20261031/projection.py", "ppr_20261028/projection.py", "ppr_eta_20261030/hooks.py",
        "ppr_eta_radius_20261104_o/hooks.py", "ppr_eta_radius_20261104_o/native_worker.py", "ppr_eta_radius_20261104_o/cache_worker.py",
        "ppr_eta_radius_20261104_o/parameter_runtime_004.py", "ppr_eta_radius_20261104_n/hooks.py")]
    files += [Path(path) for path in copied]
    c.save(c.HERE / "ANALYSIS_FROZEN.json", dict(at=time.time(), files={str(p): c.digest(p) for p in files},
        source_report_configurations=c.source()["completed_corrected_model_configs"], original_operator_and_score_reused=True,
        allowed_eta=[0., 1.], allowed_alpha=[1., 4.], target_tasks=["sp", "ie"], no_extra_neural_forwards_per_forecast=True))
    print("P proposal stage frozen; all O core and supplement file hashes passed", flush=True)


if __name__ == "__main__":
    main()
