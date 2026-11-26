"""The original metrics and ranking, with new-round provenance labels."""
import inspect
import time
from pathlib import Path
from . import common as c

_BASE = None


def base():
    global _BASE
    if _BASE is None:
        c.install_original_runtime()
        from predictive_tracks.ppr_eta_radius_20261104_o.path_runtime_001 import install
        _BASE = install()
    return _BASE


def assemble():
    old = base()
    r = old.assemble()
    ids = {x["config_id"] for x in r["current_round_settings"]}
    for x in r["current_round_settings"]:
        x["origin"] = "P_joint_eta_radius"
    for values in r["all_configurations"].values():
        for x in values:
            if x["config_id"] in ids:
                x["origin"] = "P_joint_eta_radius"
    for task in ("sp", "ie"):
        for x in r["selected"][task].values():
            if x["config_id"] in ids:
                x["origin"] = "P_joint_eta_radius"
    r["change_attribution_p"] = r.pop("change_attribution_o")
    for values in r["change_attribution_p"].values():
        for value in values.values():
            value["differences_vs_O"] = value.pop("differences_vs_N")
            if value["new_id"] in ids:
                value["origin"] = "P_joint_eta_radius"
    r["planned_stage1"] = c.read(c.HERE / "GRID.json")["settings"]
    r["current_round_note"] = "Only existing eta/alpha, fixed original operators and numerical expression. Joint search proposals are not measurements; every reported new configuration is actual full600."
    r["scope"] = "Same exposed SP300/IE300; exploratory joint parameter selection, no heldout or continuous global optimum claim."
    return r


def interim():
    r = assemble()
    dest = c.HERE / "reports" / str(time.time_ns())
    c.save(dest / "report.json", r)
    c.status(c.HERE / "monitor/latest_report.json", dict(at=time.time(), path=str(dest), completed=r["completed_round_configurations"]))
    print("P COMPLETE SETTINGS", r["completed_round_configurations"], "/", r["planned_stage1"], flush=True)
    return r


def finish_source(directory):
    old = base()
    source = inspect.getsource(old.finish)
    before = "c.HERE/'delivery_20261104_001'"
    assert source.count(before) == 1
    source = source.replace(before, repr(Path(directory)), 1)
    before = "assert rec['trace_matches_original'] and rec['extra_forwards']==0 and rec['weight_identity_and_version_unchanged']"
    assert source.count(before) == 1
    source = source.replace(before, "assert rec['trace_matches_original'] and rec['extra_forwards']==0;_verify_weight_record(rec)", 1)
    # Use a plain Path expression in the saved/compiled source.
    source = source.replace("PosixPath(", "Path(")
    return source


def finish():
    old = base()
    plan = c.read(c.HERE / "REPORT_SOURCE_DECLARATION.json")
    dest = Path(plan["directory"])
    source = finish_source(dest)
    assert source == (c.HERE / "report_finish_source.txt").read_text()
    if dest.exists():
        if (dest / "DELIVERY.json").exists():
            for path, wanted in c.read(dest / "DELIVERY.json")["files"].items():
                assert c.digest(path) == wanted
            c.verify(full=True)
            pointer = c.HERE / "FINAL_DELIVERY.json"
            if pointer.exists():
                assert c.read(pointer)["directory"] == str(dest)
            else:
                c.save(pointer, dict(at=time.time(), directory=str(dest), complete=True))
            return
        # A power loss during an audit must not overwrite a published partial
        # directory or cause another neural forecast. Only the output path moves.
        number = 2
        while (c.HERE / f"delivery_20261105_{number:03d}").exists():
            number += 1
        preserved = dest
        dest = c.HERE / f"delivery_20261105_{number:03d}"
        source = finish_source(dest)
        c.save(c.HERE / f"REPORT_OUTPUT_RECOVERY_{time.time_ns()}.json", dict(at=time.time(), preserved=str(preserved), new_directory=str(dest),
            original_declared_source_sha256=c.digest(c.HERE / "report_finish_source.txt"), output_path_only_changed=True,
            new_compiled_source=source, no_new_forecasts=True))
    from predictive_tracks.ppr_eta_radius_20261104_o.parameter_runtime_004 import verify_record
    namespace = dict(vars(old), assemble=assemble, _verify_weight_record=verify_record)
    exec(compile(source, "<P_original_metric_finish>", "exec"), namespace)
    namespace["finish"]()

