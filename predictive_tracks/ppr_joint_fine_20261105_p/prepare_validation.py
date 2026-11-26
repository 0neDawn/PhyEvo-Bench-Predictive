"""Seal all actual settings and operational routing before starting a worker."""
import inspect
import time
from pathlib import Path
from . import common as c


def main():
    from .prepare_grid import main as grid
    grid()
    assert c.read(c.HERE / "PREFLIGHT_METRICS.json")["exact_PA_JH_AUC"]
    c.install_original_runtime()
    from predictive_tracks.ppr_eta_radius_20261104_o.parameter_runtime_004 import install_worker, worker_sources, verify_addendum
    verify_addendum()
    run, execute = worker_sources()
    assert run == (c.HERE / "native_run_runtime_004.txt").read_text()
    assert execute == (c.HERE / "native_execute_runtime_004.txt").read_text()
    install_worker()
    from .report import finish_source
    directory = c.HERE / "delivery_20261105_001"
    source = finish_source(directory)
    compile(source, "<P_metric_finish_preflight>", "exec")
    with (c.HERE / "report_finish_source.txt").open("x") as stream:
        stream.write(source)
    c.save(c.HERE / "REPORT_SOURCE_DECLARATION.json", dict(at=time.time(), directory=str(directory), source_sha256=c.digest(c.HERE / "report_finish_source.txt")))
    retained = {str(c.SOURCE): c.digest(c.SOURCE), str(c.OLD / "delivery_20261104_001/DELIVERY.json"): c.digest(c.OLD / "delivery_20261104_001/DELIVERY.json")}
    c.save(c.HERE / "REUSED_HASHES.json", retained)
    files = list(c.HERE.glob("*.py")) + [c.HERE / name for name in (
        "PLAN.txt", "ANALYSIS_FROZEN.json", "ANALYSIS_COMPLETE.json", "PREFLIGHT_METRICS.json", "GRID.json", "GRID_FROZEN.json",
        "COPIED_ORIGINAL_RUNTIME.json", "native_run_runtime_004.txt", "native_execute_runtime_004.txt",
        "PATH_RUNTIME_ADDENDUM_001.json", "REPORT_RUNTIME_ADDENDUM_002.json", "COPIED_RUNTIME_COMPLETION_001.json",
        "PARAMETER_AUDIT_RUNTIME_ADDENDUM_003.json", "PARAMETER_AUDIT_RUNTIME_ADDENDUM_004.json", "report_finish_source.txt", "REPORT_SOURCE_DECLARATION.json", "REUSED_HASHES.json")]
    files += list((c.HERE / "proposals").glob("*.json"))
    files += [Path(path) for path in c.read(c.HERE / "ANALYSIS_FROZEN.json")["files"]]
    c.save(c.HERE / "frozen.json", dict(at=time.time(), files={str(p): c.digest(p) for p in set(files)},
        methods_unchanged=True, original_O_worker_sources_byte_identical=True, changed_method_parameters=["eta", "alpha"],
        selection=c.source()["selection"], source_report=str(c.SOURCE)))
    c.verify()
    c.save(c.HERE / "FROZEN_VERIFIED.json", dict(at=time.time(), passed=True, original_runtime_compiles=True, extra_neural_forwards=0))
    print("P VALIDATION FROZEN", c.read(c.HERE / "GRID.json")["settings"], "full600 settings", flush=True)


if __name__ == "__main__":
    main()
