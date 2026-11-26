"""Offline parameter search; reuse the original parameter-independent prefix.

This is partial-graph search, not an extra pass in the proposed inference method.
Fresh full-pipeline checks and final full600 validation are stored separately.
"""
import argparse
import hashlib
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from . import common as c

sys.path.insert(0, str(c.ROOT / "predictive_tracks/sp/code"))
import numpy as np
import torch
from predictive_tracks.controlled_tracks.registry import MODELS, OUTPUT
from predictive_tracks.controlled_tracks.response_full_run import describe
from predictive_tracks.ppr_20261028.run import rows_and_references
from predictive_tracks.ppr_eta_20261030.cache import meta_path
from predictive_tracks.ppr_all_20261029.projection import reference, step
from predictive_tracks.ppr_eta_radius_20261104_n.hooks import TunedPreFinal
from predictive_tracks.ppr_state_insert_20261104_m.hooks import tensor_hash
from predictive_tracks.ppr_eta_radius_20261104_o.native_worker import versions
from predictive_tracks.state_tracks_20261023.all_models_20261023.run import load_data, evaluator_for, variant_record
from predictive_tracks.ppr_existing_20261102.metrics import make_summary
from sp_core.common import PairRecord, read_pair_rgb


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, tuple):
        return tuple(cpu(v) for v in value)
    if isinstance(value, list):
        return [cpu(v) for v in value]
    if isinstance(value, dict):
        return {key:cpu(v) for key,v in value.items()}
    assert value is None or isinstance(value, (int, float, bool, str))
    return value


def cuda(value):
    if isinstance(value, torch.Tensor):
        return value.to("cuda:0")
    if isinstance(value, tuple):
        return tuple(cuda(v) for v in value)
    if isinstance(value, list):
        return [cuda(v) for v in value]
    if isinstance(value, dict):
        return {key:cuda(v) for key,v in value.items()}
    return value


def save_tensor(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f".atomic.{os.getpid()}")
    with temp.open("xb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temp, path)
    temp.unlink()


def prepare_shift(values, alpha):
    assert values.ndim == 3 and values.shape[:2] == (1, 7*256)
    states = values.reshape(1, 7, 256, -1)
    history = states[:, :4].mean(2, keepdim=True)
    prediction = states[:, 4:].mean(2, keepdim=True)
    center, radius, valid = reference(history)
    state = (center, radius if alpha == 1 else alpha*radius, valid)
    hard, audit = [], []
    for t in range(3):
        value, item = step(prediction[:, t], state)
        hard.append(value)
        audit.append(item)
    return states, prediction, hard, dict(base_radius=radius.cpu().tolist(), alpha=alpha, steps=audit)


def shift(values, prepared, eta):
    states, prediction, hard, _ = prepared
    fixed = []
    for t in range(3):
        raw = prediction[:, t]
        after = raw if eta == 0 else hard[t] if eta == 1 else raw+eta*(hard[t]-raw)
        fixed.append(after)
    delta = torch.stack(fixed, 1)-prediction
    out = values if eta == 0 else torch.cat([states[:, :4], states[:, 4:]+delta], 1).reshape_as(values)
    return out


def tail(adapter, model, values, rest, kwargs):
    if model == "vjepa2_vitl":
        predictor = adapter.predictor
        assert not predictor.return_all_tokens and predictor.chop_last_n_tokens == 0
        masks = kwargs["mask"]
        identity = torch.arange(7*256, device=masks.device)[None]
        assert torch.equal(masks, identity) and rest == ()
        x = predictor.predictor_blocks[-1](values, *rest, **kwargs)
        x = predictor.predictor_norm(x)
        # Identical sorted-mask inverse and slicing from the original forward.
        argsort = torch.argsort(masks, dim=1)
        reverse_argsort = torch.argsort(argsort, dim=1)
        x = torch.stack([x[i, row, :] for i, row in enumerate(reverse_argsort)], dim=0)
        x = x[:, 4*256:]
        x = predictor.predictor_proj(x)
    else:
        x = adapter.model.decoder.blocks[-1](values, *rest, **kwargs)
        x = adapter.model.decoder.norm(x)
        x = x[:, -3*256:]
    return x.reshape(1, 3, 256, -1)


def native_prefix(adapter, model, possible):
    if model == "vjepa2_vitl":
        from sp_vjepa2_vith.adapter import official_video_tensor
        video = official_video_tensor(possible).to(adapter.device)
        container = torch.zeros_like(video)
        container[:, :, :8] = video[:, :, :8]
        return adapter._predict(container).reshape(1, 3, 256, -1)
    video = adapter._pad_to_official_16(possible)
    container = torch.zeros_like(video)
    container[:, :, :8] = video[:, :, :8]
    return adapter._prediction(container)


def collect_prefix(adapter, model, possible):
    block = adapter.predictor.predictor_blocks[-1] if model == "vjepa2_vitl" else adapter.model.decoder.blocks[-1]
    captured = []
    def before(module, args, kwargs):
        captured.append((cpu(args), cpu(kwargs)))
    handle = block.register_forward_pre_hook(before, with_kwargs=True)
    try:
        raw = native_prefix(adapter, model, possible)
    finally:
        handle.remove()
    assert len(captured) == 1
    return raw, captured[0]


def exact(actual, expected):
    difference = max(float(np.max(np.abs(np.asarray(x)-np.asarray(y)))) for x,y in zip(actual, expected))
    if difference != 0:
        raise RuntimeError(f"Original float32 CUDA errors differ: {difference}")
    return difference


def expected_trace(model, row, ref):
    if ref.get("neural_call_trace") is not None:
        return ref["neural_call_trace"]
    traces = c.read(OUTPUT / "deployment_audit_001.json")["details"][model]
    shared = {json.dumps(x["neural_trace"], sort_keys=True) for x in traces}
    if len(shared) == 1:
        return json.loads(next(iter(shared)))
    return c.read(OUTPUT / f"{model}_boundary_dev_001/pairs" / f'{row["key"]}.json')["neural_call_trace"]


def full_check(adapter, model, row, ref, possible, impossible, values, rest, kwargs, evaluator, eta, alpha):
    destination = c.HERE / "full_pipeline_preflight" / model / f'{row["key"]}_e{eta:.8g}_a{alpha:.8g}.json'
    if destination.exists():
        return
    if model == "vjepa2_vitl":
        adapter.predictor.to("cuda:0")
    else:
        adapter.model.decoder.to("cuda:0")
    with torch.inference_mode():
        prepared = prepare_shift(values, alpha)
        modified = shift(values, prepared, eta)
        hook = TunedPreFinal(adapter, model, eta, alpha, "pre_final_block")
        original_modified = hook.shift(values)
        assert torch.equal(modified, original_modified), "Optimized search write differs from original formula"
        replay = evaluator.errors(tail(adapter, model, modified, rest, kwargs))
    calls, trace = [0], hashlib.sha256()
    def count(module, inputs):
        calls[0] += 1
        trace.update(json.dumps((module.__class__.__module__, module.__class__.__qualname__, describe(inputs)), sort_keys=True).encode())
    adapter.prediction_calls = row["archive_index"]
    baseline_versions = versions(adapter, model)
    counter = torch.nn.modules.module.register_module_forward_pre_hook(count)
    try:
        h = TunedPreFinal(adapter, model, eta, alpha, "pre_final_block")
        with h:
            result = adapter.score_pair(possible, impossible)
    finally:
        counter.remove()
    actual = (result.possible, result.impossible)
    exact(replay, actual)
    observed = dict(module_forward_invocations=calls[0], ordered_class_and_input_shapes_sha256=trace.hexdigest())
    assert observed == expected_trace(model, row, ref)
    assert adapter.prediction_calls == row["archive_index"]+1
    assert versions(adapter, model) == baseline_versions
    assert h.activations == 1 and h.inactive_calls == (6 if model == "videomaev2" else 0)
    assert result.frame_spans.tolist() == ref["frame_spans"]
    c.save(destination, dict(at=time.time(), model=model, key=row["key"], eta=eta, alpha=alpha,
        full_pipeline_trace=observed, original_trace_exact=True, full_vs_tail_max_abs_error=0.,
        original_vs_optimized_shift_bitwise_equal=True, weights_identity_version_unchanged=True,
        additional_forwards_in_deployed_forecast=0))


def expand(payload, item):
    value = dict(payload["native"])
    value.update(payload["settings"][item["config_id"]])
    value.update(model=item["model"], eta=item["eta"], radius_multiplier=item["alpha"], mode=item["mode"],
                 partial_graph_search=True, extra_forwards_in_deployed_forecast=0)
    return value


def execute(model, stage2=False):
    c.verify()
    c.require_resources(model)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.set_per_process_memory_fraction(.45, device=0)
    grid = c.read(c.HERE / ("STAGE2.json" if stage2 else "GRID.json"))
    jobs = [j for j in grid["tail"] if j["model"] == model]
    phase = "tail_stage2" if stage2 else "tail"
    root = c.HERE / phase / model
    done = c.HERE / "models_complete" / f"{phase}_{model}.json"
    if done.exists():
        return
    if not jobs:
        c.save(done, dict(at=time.time(), settings=0))
        return
    data_rows = sorted(rows_and_references(model), key=lambda x:(x[0]["benchmark"] != "ie", x[0]["archive_index"]))
    adapter = importlib.import_module(MODELS[model]).Adapter(device="cuda:0", memory_mode="safe", orientation="identity")
    base = versions(adapter, model)
    analysis = c.read(c.HERE / "INTERNAL_ACTIVATION_ANALYSIS.json")[model]
    median = round(analysis["quantiles"]["0.5"], 8)
    known = {j["config_id"]:Path(j["confirmation_source"]) / "pairs" for j in jobs if j.get("confirmation_source")}
    source_audits = Path(analysis["source_directory"])
    preflight_keys = {f"{task}_{i:03d}" for task in ("sp", "ie") for i in (0,100,200)}
    with torch.inference_mode():
        for index, (row, rp, ref) in enumerate(data_rows, 1):
            dest = root / "pairs" / f'{row["key"]}.json'
            if dest.exists():
                payload = c.read(dest)
                assert set(payload["settings"]) == {j["config_id"] for j in jobs}
                continue
            c.require_resources(model)
            mp = meta_path(model, row)
            meta = c.read(mp)
            cached = load_data(meta)
            evaluator = evaluator_for(cached, model)
            evaluator.adapter.device = torch.device("cuda:0")
            evaluator.targets = [v.to(evaluator.adapter.device) for v in evaluator.targets]
            pair = PairRecord(row["pair_id"], row["category_directory"], int(row["t_ref"]), Path(row["possible_dir"]), Path(row["impossible_dir"]), Path(row["meta_path"]))
            possible, impossible = read_pair_rgb(pair)
            assert hashlib.sha256(possible.tobytes()+impossible.tobytes()).hexdigest() == ref["source_decoded_pixels_sha256"]
            capture_path = c.HERE / "prefix" / model / f'{row["key"]}.pt'
            capture_meta = capture_path.with_suffix(".json")
            if capture_path.exists() and not capture_meta.exists():
                # Keep the interrupted unpublished capture, then capture again.
                preserved = capture_path.with_name(capture_path.name+f".unpublished.{time.time_ns()}")
                capture_path.rename(preserved)
                c.save(c.HERE / "events" / f"capture_recovery_{time.time_ns()}.json",
                       dict(at=time.time(), preserved=str(preserved), original_file_not_deleted=True))
            if capture_path.exists():
                cm = c.read(capture_meta)
                assert c.digest(capture_path) == cm["sha256"]
                captured = torch.load(capture_path, map_location="cpu", weights_only=True)
                args, kwargs = cuda(captured["args"]), cuda(captured["kwargs"])
                raw = cached["prediction"].to("cuda:0")
            else:
                adapter.prediction_calls = row["archive_index"]
                raw, (args_cpu, kwargs_cpu) = collect_prefix(adapter, model, possible)
                assert torch.equal(raw.detach().cpu(), cached["prediction"]), "Native prediction differs from original cache"
                args, kwargs = cuda(args_cpu), cuda(kwargs_cpu)
                original_audit = c.read(source_audits / f'{row["key"]}.json')["projection_audits"][0]
                assert tensor_hash(args[0]) == original_audit["input_hash"], "Parameter-independent prefix differs"
                save_tensor(capture_path, dict(args=args_cpu, kwargs=kwargs_cpu))
                c.save(capture_meta, dict(at=time.time(), sha256=c.digest(capture_path),
                    input_hash=original_audit["input_hash"], native_prediction_bitwise_equal=True,
                    original_audit=str(source_audits / f'{row["key"]}.json'),
                    original_audit_sha256=c.digest(source_audits / f'{row["key"]}.json'),
                    source_meta=str(mp), source_meta_sha256=c.digest(mp), source_state_sha256=meta["sha256"],
                    capture_only_original_prediction_no_candidate_input=True))
            values, rest = args[0], args[1:]
            # A restart may leave the adapter on CPU while the cache is on GPU.
            if model == "vjepa2_vitl":
                adapter.predictor.to("cuda:0")
            else:
                adapter.model.decoder.to("cuda:0")
            native = evaluator.errors(raw)
            wanted = [meta["matched_identity_"+b+"_errors"] for b in ("possible", "impossible")]
            exact(native, wanted)
            raw_tail = tail(adapter, model, values, rest, kwargs)
            assert torch.equal(raw_tail, raw), "Original suffix does not reproduce original prediction bitwise"
            if row["key"] in preflight_keys:
                for eta in (0., .1):
                    full_check(adapter, model, row, ref, possible, impossible, values, rest, kwargs, evaluator, eta, median)
                if model == "vjepa2_vitl":
                    adapter.predictor.to("cuda:0")
                else:
                    adapter.model.decoder.to("cuda:0")
            settings, audits, confirmed = {}, {}, {}
            for alpha in sorted({j["alpha"] for j in jobs}):
                prepared = prepare_shift(values, alpha)
                audits[str(alpha)] = prepared[3]
                for item in (j for j in jobs if j["alpha"] == alpha):
                    modified = shift(values, prepared, item["eta"])
                    output = tail(adapter, model, modified, rest, kwargs)
                    errors = evaluator.errors(output)
                    key = item["config_id"]
                    if key in known:
                        old = c.read(known[key] / f'{row["key"]}.json')
                        exact(errors, [old[b+"_errors"] for b in ("possible", "impossible")])
                        confirmed[key] = c.digest(known[key] / f'{row["key"]}.json')
                    settings[key] = dict(possible_errors=np.asarray(errors[0]).tolist(), impossible_errors=np.asarray(errors[1]).tolist())
                del prepared, modified, output
            assert versions(adapter, model) == base
            c.save(dest, dict(native=variant_record(meta, native), settings=settings, projection_audits=audits,
                prefix_meta=str(capture_meta), prefix_meta_sha256=c.digest(capture_meta),
                source_meta=str(mp), source_meta_sha256=c.digest(mp), source_state_sha256=meta["sha256"],
                original_native_prediction_bitwise_equal=True, original_native_errors_exact=True,
                original_suffix_output_bitwise_equal=True, confirmed_full_pipeline_settings=confirmed,
                original_reference_pair=str(rp), original_reference_pair_sha256=c.digest(rp),
                search_kind="serial batch1 partial original graph; parameter-independent prefix cached",
                additional_forwards_in_deployed_forecast=0, original_dtype_device_expression=True))
            c.status(root / "progress.json", dict(at=time.time(), completed=index, total=600, settings=len(jobs), model=model))
            if index % 10 == 0:
                print("TAIL", model, phase, index, "/600", len(jobs), "settings", flush=True)
            del evaluator, cached, raw, values, rest, kwargs, args, settings, possible, impossible, raw_tail
    payloads = [c.read(path) for path in sorted((root / "pairs").glob("*.json"))]
    assert len(payloads) == 600
    for item in jobs:
        out = c.HERE / "search_settings" / item["model"] / item["config_id"]
        if (out / "summary.json").exists():
            continue
        summary = make_summary([expand(payload, item) for payload in payloads], model)
        summary.update(job=item, compact_case_source=str(root / "pairs"), original_errors_exact=True,
            partial_graph_parameter_search=True, additional_forwards_in_deployed_forecast=0,
            full600_native_validation_required_for_new_primary_row=True)
        c.save(out / "summary.json", summary)
    c.verify()
    c.save(done, dict(at=time.time(), model=model, settings=len(jobs), cases_each=600,
        original_prediction_and_suffix_exact=True, original_known_setting_errors_exact=True,
        original_parameters_unchanged=True, search_is_partial_original_graph=True))
    print("TAIL MODEL COMPLETE", model, phase, len(jobs), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=c.FOCUS, required=True)
    p.add_argument("--stage2", action="store_true")
    args = p.parse_args()
    try:
        execute(args.model, args.stage2)
    except BaseException as error:
        from predictive_tracks.ppr_eta_radius_20261104_o.common import ResourceBusy
        pause = isinstance(error, ResourceBusy)
        c.save(c.HERE / "events" / f'{"pause" if pause else "error"}_{time.time_ns()}.json',
               dict(at=time.time(), model=args.model, error=repr(error), traceback=traceback.format_exc()))
        if pause:
            raise SystemExit(75)
        raise


if __name__ == "__main__":
    main()
