"""Freeze global candidates before any new full600 validation."""
import json
import struct
import time
from . import common as c


def effective_key(model, mode, eta, alpha, quality):
    # Identical float32 parameter values are confirmations, not new trials.
    return (model, mode, struct.pack("f", eta).hex(), struct.pack("f", alpha).hex(), json.dumps(quality, sort_keys=True))


def main():
    if (c.HERE / "GRID.json").exists():
        assert (c.HERE / "GRID_FROZEN.json").exists()
        for path, wanted in c.read(c.HERE / "GRID_FROZEN.json")["files"].items():
            assert c.digest(path) == wanted
        return
    assert c.read(c.HERE / "ANALYSIS_COMPLETE.json")["neural_forwards"] == 0
    occupied = set()
    for model, rows in c.source()["all_configurations"].items():
        for x in rows:
            p = x["parameters"]
            quality = {k: p.get(k) for k in ("q", "presence", "mass", "min_history")} if model in c.OBJECTS else None
            occupied.add(effective_key(model, x["mode"], p["eta"], p.get("alpha", 1.), quality))
    cached, native, reasons = [], [], []

    def add(job, reason):
        key = effective_key(job["model"], job["mode"], job["eta"], job["alpha"], job["quality"])
        if key in occupied:
            return False
        occupied.add(key)
        job = dict(job, search_reason=reason)
        (cached if job["model"] in c.CACHED and job["mode"] == "output_only" else native).append(job)
        return True

    def candidates(model, task, is_cached, maximum):
        if maximum <= 0:
            return 0
        values = c.read(c.HERE / "proposals" / (model + ".json"))["candidates"][task]
        selected = []
        # The published selection stays fixed. These extra objectives merely
        # propose separately labelled full configurations for possible tradeoffs.
        for role, limit in (("primary", max(1, maximum-2)), ("auc", 1), ("mae", 1)):
            if len(selected) >= maximum:
                break
            used = 0
            for job in values[role]:
                if (job["model"] in c.CACHED and job["mode"] == "output_only") != is_cached:
                    continue
                if any(job["mode"] == x["mode"] and abs(job["eta"]-x["eta"]) < max(1e-5, .025*max(job["eta"], x["eta"])) and abs(job["alpha"]-x["alpha"]) < .02 for x in selected):
                    continue
                if add(job, "joint_surrogate_" + task + "_" + role):
                    selected.append(job)
                    used += 1
                if used >= limit or len(selected) >= maximum:
                    break
        return len(selected)

    for model in c.CACHED:
        for task in ("sp", "ie"):
            reasons.append(dict(model=model, task=task, cached_candidates=candidates(model, task, True, 8)))
    # Explicit, data-wide neighborhood grid from PLAN.txt. Both tasks are
    # evaluated for every point, independent of which task motivated the point.
    for i in range(21):
        for j in range(21):
            add(c.job("slotformer", "output_only", .95+.0025*i, 1.+.005*j), "declared_SP_neighborhood")
    for i in range(16):
        for j in range(41):
            add(c.job("slotformer", "output_only", .45+.01*i, 1.30+.01*j), "declared_IE_neighborhood")
    probes = {
        "ocvp_seq": {"sp": [("output_only", .975, 1.225), ("output_only", 1., 1.20)],
                     "ie": [("output_only", .225, 2.05), ("output_only", .175, 1.95), ("feedback", .125, 1.30), ("feedback", .25, 2.10)]},
        "savi_transformer": {"sp": [("output_only", .975, 1.225), ("output_only", 1., 1.20)],
                             "ie": [("feedback", .525, 2.05), ("feedback", .475, 1.95), ("output_only", .525, 2.05), ("feedback", .55, 2.125)]},
        "ddlp": {"sp": [("feedback", .975, 1.05), ("feedback", 1., 1.10)],
                 "ie": [("feedback", .975, 1.05), ("feedback", 1., 1.10)]},
        "gswm": {"sp": [("feedback", .975, 1.20), ("feedback", 1., 1.15)],
                 "ie": [("feedback", .975, 1.05), ("feedback", 1., 1.10)]},
        "vjepa2_vitl": {"sp": [("pre_final_block", .00025, 1.05), ("pre_final_block", .00075, 1.25)],
                        "ie": [("pre_final_block", .1, 1.05), ("pre_final_block", .1, 1.125)]},
        "videomaev2": {"sp": [("pre_final_block", .005, 1.125)],
                       "ie": [("pre_final_block", .00005, 1.05), ("pre_final_block", .00025, 1.125), ("pre_final_block", .00075, 1.375)]},
        "slotformer": {"sp": [], "ie": []},
    }
    for model in c.NATIVE:
        for task in ("sp", "ie"):
            maximum = (4 if task == "sp" else 8) if model in ("ocvp_seq", "savi_transformer") else (1 if task == "sp" else 3) if model == "videomaev2" else 2
            # Reserve the declared local probes, especially for the zero-gain
            # models. A surrogate is not evidence of an impossibility result.
            local = probes[model][task]
            used = candidates(model, task, False, max(0, maximum-len(local))) if maximum > len(local) else 0
            for mode, eta, alpha in local:
                if used >= maximum:
                    break
                quality = c.quality(model, task)
                job = c.job(model, mode, eta, alpha, quality, task if model in c.OBJECTS else None,
                            proposal_task=task, proposal_only=True, proposal_kind="declared local actual-validation probe; no interpolated score asserted")
                used += int(add(job, "declared_local_probe_" + task))
            reasons.append(dict(model=model, task=task, native_candidates=used, maximum=maximum))
    assert len(native) <= 44
    for model in c.CACHED:
        if model != "slotformer":
            assert sum(j["model"] == model for j in cached) <= 16
    assert all(0 < j["eta"] <= 1 and 1 <= j["alpha"] <= 4 for j in cached+native)
    declaration = dict(at=time.time(), cached=cached, native_new=native, native_reuse=[], settings=len(cached)+len(native),
        counts={m: dict(cached=sum(x["model"] == m for x in cached), native=sum(x["model"] == m for x in native)) for m in c.ORDER},
        reasons=reasons, source_report_sha256=c.digest(c.SOURCE), independent_pairs=600, each_configuration_pairs=600,
        selection=c.source()["selection"], all_proposals_require_actual_original_scoring=True, method_parameters_only=["eta", "alpha"])
    c.save(c.HERE / "GRID.json", declaration)
    c.save(c.HERE / "GRID_FROZEN.json", dict(at=time.time(), files={str(c.HERE / "GRID.json"): c.digest(c.HERE / "GRID.json")},
        analysis_declaration_sha256=c.digest(c.HERE / "ANALYSIS_FROZEN.json"), selection_frozen_before_validation=True))
    print("P ACTUAL GRID FROZEN", declaration["settings"], "settings; cached", len(cached), "native", len(native), flush=True)


if __name__ == "__main__":
    main()
