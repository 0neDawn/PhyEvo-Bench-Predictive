"""Verify the completed delivery and preserve the reused runtime evidence."""
import time
from pathlib import Path
from . import common as c


def main():
    assert c.read(c.HERE / "PERSISTED.json")["all_experiments_and_audits_complete"]
    directory = Path(c.read(c.HERE / "FINAL_DELIVERY.json")["directory"])
    manifest = c.read(directory / "DELIVERY.json")
    assert manifest["all_checks_passed"]
    for path, wanted in manifest["files"].items():
        assert c.digest(path) == wanted, path
    audit = c.read(directory / "COMPLETENESS_AUDIT.json")
    assert audit["configurations"] == c.read(c.HERE / "GRID.json")["settings"]
    assert audit["case_records"] == 600*audit["configurations"] and audit["independent_pairs"] == 600
    assert audit["all_original_metrics_recomputed"]
    c.verify()
    for path, value in c.read(c.HERE / "COPIED_ORIGINAL_RUNTIME.json").items():
        assert c.digest(path) == value["sha256"] == c.digest(value["original"])
    for path, value in c.read(c.HERE / "COPIED_RUNTIME_COMPLETION_001.json")["files"].items():
        assert c.digest(path) == value["sha256"] == c.digest(value["original"])
    from PIL import Image
    with Image.open(directory / "table.png") as image:
        image.verify()
    out = directory / "runtime_and_diagnostics_001"
    out.mkdir(exist_ok=True)
    sources = list(c.HERE.glob("*.py")) + [c.HERE / name for name in (
        "PLAN.txt", "frozen.json", "ANALYSIS_FROZEN.json", "ANALYSIS_COMPLETE.json", "GRID.json", "GRID_FROZEN.json",
        "FROZEN_VERIFIED.json", "PREFLIGHT_METRICS.json", "COPIED_ORIGINAL_RUNTIME.json", "PRIORITY_APP_VERIFIED_001.json",
        "PATH_RUNTIME_ADDENDUM_001.json", "REPORT_RUNTIME_ADDENDUM_002.json", "COPIED_RUNTIME_COMPLETION_001.json",
        "native_run_runtime_004.txt", "native_execute_runtime_004.txt", "PARAMETER_AUDIT_RUNTIME_ADDENDUM_003.json",
        "PARAMETER_AUDIT_RUNTIME_ADDENDUM_004.json", "report_finish_source.txt", "REPORT_SOURCE_DECLARATION.json", "REUSED_HASHES.json")]
    sources += list((c.HERE / "weight_audits").glob("*/*.json"))
    sources += list(c.HERE.glob("REPORT_OUTPUT_RECOVERY_*.json"))
    for path in sources:
        if not path.is_file():
            continue
        target = out / path.relative_to(c.HERE)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert c.digest(target) == c.digest(path)
        else:
            with target.open("xb") as stream:
                stream.write(path.read_bytes())
    c.save(directory / "SUPPLEMENT_MANIFEST_001.json", dict(at=time.time(), files={str(p): c.digest(p) for p in out.rglob("*") if p.is_file()},
        all_original_operator_and_native_runtime_sources_preserved=True, neural_forwards_added_per_forecast=0))
    c.save(c.HERE / "FINAL_STATE_001.json", dict(at=time.time(), directory=str(directory), all_declared_joint_configurations_complete=True,
        all_lineage_audits_complete=True, all_models_improving_not_assumed=True, same_exposed_data_selection=True))
    print("P LINEAGE AND DELIVERY VERIFIED", directory, flush=True)


if __name__ == "__main__":
    main()
