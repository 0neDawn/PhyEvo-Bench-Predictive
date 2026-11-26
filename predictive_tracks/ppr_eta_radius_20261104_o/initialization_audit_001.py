"""One explicit no-op implementation diagnostic; no extra call in a forecast."""
import hashlib
import importlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import common as c
from . import native_worker as worker
from .hooks import factory
from predictive_tracks.controlled_tracks.registry import MODELS, OUTPUT
from predictive_tracks.controlled_tracks.response_full_run import describe
from predictive_tracks.ppr_20261028.run import rows_and_references
from sp_core.common import PairRecord, read_pair_rgb


def parameter_state(adapter):
    result = {}
    for owner in ("savi", "dynamics"):
        for name, parameter in getattr(adapter, owner).named_parameters():
            value = parameter.detach().cpu().contiguous()
            result[owner + "." + name] = dict(
                identity=id(parameter), version=parameter._version,
                shape=list(parameter.shape), dtype=str(parameter.dtype),
                requires_grad=parameter.requires_grad,
                value_sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest(),
            )
    return result


def main():
    model = "slotformer"
    destination = c.HERE / "SLOTFORMER_INITIALIZATION_AUDIT_001.json"
    if destination.exists():
        raise RuntimeError("Diagnostic is already retained; do not repeat it")
    c.require_resources(model)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.set_per_process_memory_fraction(.2, device=0)
    data = sorted(rows_and_references(model), key=lambda x: (x[0]["benchmark"] != "ie", x[0]["archive_index"]))
    row, reference_path, reference = worker.preselected(data)[0]
    job = next(j for j in c.read(c.HERE / "GRID.json")["native_new"] if j["model"] == model)
    job = dict(job, eta=0., alpha=1.)
    adapter = importlib.import_module(MODELS[model]).Adapter(device="cuda:0", memory_mode="safe", orientation="identity")
    traces = {json.dumps(r["neural_trace"], sort_keys=True) for r in c.read(OUTPUT / "deployment_audit_001.json")["details"][model]}
    shared = json.loads(next(iter(traces))) if len(traces) == 1 else None
    expected = reference.get("neural_call_trace", shared)
    if expected is None and row["split"] == "development":
        expected = c.read(OUTPUT / f"{model}_boundary_dev_001/pairs" / f'{row["key"]}.json')["neural_call_trace"]
    if expected is None:
        raise RuntimeError("Missing original call trace")
    count = expected["module_forward_invocations"]
    if count not in (1321, 2103):
        raise RuntimeError("Unknown original tokenizer schedule")
    if count == 1321:
        adapter.savi.clip_len = 14
    pair = PairRecord(row["pair_id"], row["category_directory"], int(row["t_ref"]), Path(row["possible_dir"]), Path(row["impossible_dir"]), Path(row["meta_path"]))
    possible, impossible = read_pair_rgb(pair)
    pixels = hashlib.sha256(possible.tobytes() + impossible.tobytes()).hexdigest()
    assert pixels == reference["source_decoded_pixels_sha256"]
    adapter.prediction_calls = row["archive_index"]
    before = parameter_state(adapter)
    calls = [0]
    trace = hashlib.sha256()

    def record_call(module, inputs):
        calls[0] += 1
        trace.update(json.dumps((module.__class__.__module__, module.__class__.__qualname__, describe(inputs)), sort_keys=True).encode())

    counter = torch.nn.modules.module.register_module_forward_pre_hook(record_call)
    try:
        hook = factory(adapter, job, c.HERE / "diagnostic_sources_001")
        with hook:
            result = adapter.score_pair(possible, impossible)
    finally:
        counter.remove()
    torch.cuda.synchronize()
    after = parameter_state(adapter)
    changed = {name: dict(before=before[name], after=after[name]) for name in before if before[name] != after[name]}
    values_unchanged = before.keys() == after.keys() and all(before[k]["value_sha256"] == after[k]["value_sha256"] for k in before)
    identity_unchanged = before.keys() == after.keys() and all(before[k]["identity"] == after[k]["identity"] for k in before)
    diff = max(float(np.max(np.abs(x - np.asarray(reference["matched_identity_" + branch + "_errors"])))) for x, branch in ((result.possible, "possible"), (result.impossible, "impossible")))
    observed = dict(module_forward_invocations=calls[0], ordered_class_and_input_shapes_sha256=trace.hexdigest())
    payload = dict(at=time.time(), purpose="Explicit diagnostic for the first planned eta0 audit failure; not a scientific parameter result", model=model,
                   case=row["key"], changed_parameters=changed, parameter_count=len(before), all_parameter_values_bitwise_unchanged=values_unchanged,
                   all_parameter_identities_unchanged=identity_unchanged, before=before, after=after,
                   no_op_max_abs_error=diff, observed_trace=observed, original_trace=expected,
                   original_trace_matches=observed == expected, original_prediction_count=adapter.prediction_calls == row["archive_index"] + 1,
                   frame_spans_match=result.frame_spans.tolist() == reference["frame_spans"], source_pixels_sha256=pixels,
                   reference_pair_sha256=c.digest(reference_path), extra_forwards_per_forecast=0)
    c.save(destination, payload)
    print(json.dumps({k: v for k, v in payload.items() if k not in ("before", "after")}, ensure_ascii=False), flush=True)
    assert values_unchanged and identity_unchanged and diff == 0 and observed == expected


if __name__ == "__main__":
    main()
