"""Exploratory remaining-270 evaluation; same native graph, no extra forwards.

Uses the original archive data/splits and reusable interception adapters. Does
not mutate the frozen runner, its registry, or its confirmation experiment.
"""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

from .registry import CODE, ROOT, OUTPUT, MODELS
from .run import digest, frozen_files, verify_frozen, write_json, records_and_split
from .native_cache import NativeCache
from .response_calibration import response_calibration
from .orthogonal_boundary import align_orthogonal_boundary
from .response_replay import VARIANTS, RGB_MODELS, diagnostic
from .full_metrics_report import aggregate


def describe(value):
    if torch.is_tensor(value):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (tuple, list)):
        return [describe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): describe(v) for k, v in value.items()}
    return type(value).__name__


def execute(model, directory):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    sys.path.insert(0, str(CODE))
    from sp_core.common import PairRecord, read_pair_rgb
    if not torch.cuda.is_available():
        raise RuntimeError("Native GPU unavailable; no neural CPU fallback")
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(4)
    frozen = frozen_files()
    old_plan = Path(__file__).with_name("confirmation_plan.json")
    frozen.update(json.loads(old_plan.read_text())["locked_source_hashes"])
    frozen[str(old_plan)] = digest(old_plan)
    verify_frozen(frozen)
    plan_path = Path(__file__).with_name("response_full_plan.json")
    plan = json.loads(plan_path.read_text())
    if model not in plan["models"]:
        raise RuntimeError("Model outside frozen exploratory plan")
    verify_frozen(plan["method_source_hashes"])
    paths = [Path(__file__), Path(__file__).with_name("response_replay.py"),
             Path(__file__).with_name("response_calibration.py"), plan_path]
    paths.extend(CODE.rglob("track_response*.py"))
    snapshot = {str(p.resolve()): digest(p) for p in paths}
    write_json(directory / "run_config.json", {
        "model": model, "plan": plan, "frozen_hashes": frozen, "new_code_hashes": snapshot,
        "gpu": torch.cuda.get_device_name(), "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
        "scope": "270 remaining pairs/task; exploratory, not independent confirmation",
        "data_frames": {"history": list(range(12, 20)), "future": list(range(20, 26))},
        "additional_neural_forwards": 0, "no_native_state_cache_written": True,
    })
    for p in paths:
        destination = directory / "code_snapshot" / p.resolve().relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(p.read_bytes())
    module = MODELS[model].replace("track_orientation", "track_response")
    adapter = importlib.import_module(module).Adapter(device="cuda:0", memory_mode="safe", orientation="identity")
    write_json(directory / "model_provenance.json", adapter.provenance())
    (directory / "pairs").mkdir()
    all_rows = {v: [] for v in VARIANTS}
    count = 0
    for task in ("sp", "ie"):
        selected = [r for r in records_and_split(task) if r["split"] == "confirmation"]
        if len(selected) != 270:
            raise RuntimeError("Frozen 270-pair partition changed")
        for row in selected:
            index = row["archive_index"]
            reference_path = OUTPUT / f"{model}_parallel_confirmation_001" / "pairs" / f"{task}_{index:03d}.json"
            reference = json.loads(reference_path.read_text())
            record = PairRecord(row["pair_id"], row["category_directory"], int(row["t_ref"]),
                                Path(row["possible_dir"]), Path(row["impossible_dir"]), Path(row["meta_path"]))
            possible, impossible = read_pair_rgb(record)
            pixels_hash = hashlib.sha256(possible.tobytes() + impossible.tobytes()).hexdigest()
            if pixels_hash != reference["source_decoded_pixels_sha256"] or reference["pair_id"] != row["pair_id"]:
                raise RuntimeError("Original input or pair identity changed")
            adapter.prediction_calls = index
            trace, calls = hashlib.sha256(), [0]
            def trace_call(module, inputs):
                calls[0] += 1
                event = module.__class__.__module__, module.__class__.__qualname__, describe(inputs)
                trace.update(json.dumps(event, sort_keys=True).encode())
            hook = torch.nn.modules.module.register_module_forward_pre_hook(trace_call)
            cache = NativeCache(adapter, model)
            torch.cuda.synchronize()
            started = time.perf_counter()
            try:
                with cache:
                    result = adapter.score_pair(possible, impossible)
            finally:
                hook.remove()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            neural_trace = {"module_forward_invocations": calls[0],
                            "ordered_class_and_input_shapes_sha256": trace.hexdigest()}
            if neural_trace != reference["neural_call_trace"]:
                raise RuntimeError("Original neural call count/order/input shapes changed")
            if adapter.prediction_calls != index + 1:
                raise RuntimeError("Forecast count changed")
            if hashlib.sha256(possible.tobytes() + impossible.tobytes()).hexdigest() != pixels_hash:
                raise RuntimeError("Source pixels were mutated")
            if result.frame_spans.tolist() != reference["frame_spans"]:
                raise RuntimeError("Original time grid changed")
            # All controls run after the native graph. Forbid any accidental
            # encoder/decoder call while evaluating tensor-only alternatives.
            def forbid(module, inputs):
                raise RuntimeError("Extra neural call in analytic controls")
            guard = torch.nn.modules.module.register_module_forward_pre_hook(forbid)
            try:
                cache.finish_targets(possible, impossible)
                controls = cache.audit_and_controls(result)
                baseline = [np.asarray(controls[f"native_prediction_{b}_errors"]) for b in ("possible", "impossible")]
                for value, branch in zip(baseline, ("possible", "impossible")):
                    if not np.allclose(value, reference[f"matched_identity_{branch}_errors"], rtol=1e-5, atol=1e-7):
                        raise RuntimeError("Native forecast no longer matches previous same-graph baseline")
                payload = {"source_decoded_pixels_sha256": pixels_hash, "reference_pair_sha256": digest(reference_path),
                           "neural_call_trace": neural_trace, "neural_trace_matches_previous": True,
                           "native_graph_with_correction_seconds": seconds, "variants": {}}
                p, h = cache.data["prediction"], cache.data["history"]
                old_errors = None
                with torch.inference_mode():
                    for variant in VARIANTS:
                        analytic_start = time.perf_counter()
                        if variant == "frozen_parallel":
                            corrected, audit = align_orthogonal_boundary(p, h, rgb=model in RGB_MODELS, component="parallel")
                        elif variant == "response":
                            corrected, audit = cache.data["corrected_prediction"], adapter._projection_audit
                        else:
                            corrected, audit = response_calibration(p, h, rgb=model in RGB_MODELS, mode=variant)
                        errors = cache.errors(corrected)
                        if variant == "frozen_parallel":
                            old_errors = errors
                            for value, branch in zip(errors, ("possible", "impossible")):
                                if not np.allclose(value, reference[f"{branch}_errors"], rtol=1e-5, atol=1e-7):
                                    raise RuntimeError("Old parallel control changed")
                        if variant == "response":
                            for value, actual in zip(errors, (result.possible, result.impossible)):
                                if not np.allclose(value, actual, rtol=1e-5, atol=1e-7):
                                    raise RuntimeError("Deployed result differs from same-state replay")
                        entry = {"benchmark": task, "pair_id": row["pair_id"], "category": row["category_directory"],
                                 "split": "remaining270_exploratory", "t_ref": int(row["t_ref"]),
                                 "frame_spans": result.frame_spans.tolist(),
                                 "possible_errors": errors[0].tolist(), "impossible_errors": errors[1].tolist(),
                                 "matched_identity_possible_errors": baseline[0].tolist(),
                                 "matched_identity_impossible_errors": baseline[1].tolist(),
                                 "frozen_parallel_possible_errors": old_errors[0].tolist(),
                                 "frozen_parallel_impossible_errors": old_errors[1].tolist(),
                                 "analytic_seconds": time.perf_counter() - analytic_start,
                                 "audit": audit, "diagnostic": diagnostic(p, corrected, h, cache.data["possible_target"])}
                        payload["variants"][variant] = entry
                        all_rows[variant].append(entry)
            finally:
                guard.remove()
            write_json(directory / "pairs" / f"{task}_{index:03d}.json", payload)
            count += 1
            if count % 90 == 0:
                print(model, "completed", count, "/540; neural graph verified", flush=True)
        # No interim scores are used for selecting or changing the method.
    verify_frozen(frozen)
    verify_frozen(snapshot)
    report = {v: {task: aggregate([r for r in rows if r["benchmark"] == task], model)
                  for task in ("sp", "ie")} for v, rows in all_rows.items()}
    write_json(directory / "summary.json", report)
    print("COMPLETE", model, directory, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError("Simple exclusive run name required")
    directory = OUTPUT / args.name
    directory.mkdir(parents=True, exist_ok=False)
    try:
        execute(args.model, directory)
    except BaseException as exc:
        write_json(directory / "FAILED.json", {"exception": repr(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
