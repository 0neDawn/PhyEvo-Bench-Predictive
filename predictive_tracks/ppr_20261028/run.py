"""Full SP300 + IE300 PPR, using frozen native adapters and metric functions."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
import numpy as np
import torch

from .control import HERE, ROOT, await_gate, write_status
from .hooks import DDLPHooks, GSWMHooks
from predictive_tracks.controlled_tracks.registry import CODE, OUTPUT, MODELS
from predictive_tracks.controlled_tracks.run import records_and_split, digest, frozen_files, verify_frozen
from predictive_tracks.controlled_tracks.response_full_run import describe
from predictive_tracks.controlled_tracks.full_metrics_report import aggregate


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def rows_and_references(model):
    rows = []
    for benchmark in ("sp", "ie"):
        selected = records_and_split(benchmark)
        if len(selected) != 300 or len({x["pair_id"] for x in selected}) != 300:
            raise RuntimeError("Expected 300 distinct pairs per benchmark")
        for row in selected:
            key = f"{benchmark}_{row['archive_index']:03d}"
            split = "dev" if row["split"] == "development" else "confirmation"
            path = OUTPUT / f"{model}_parallel_{split}_001/pairs/{key}.json"
            ref = json.loads(path.read_text())
            if ref["pair_id"] != row["pair_id"]:
                raise RuntimeError("Native reference identity mismatch")
            rows.append((dict(row, benchmark=benchmark, key=key), path, ref))
    return rows


def execute(model, mode, run_dir):
    await_gate(consecutive=2)
    sys.path.insert(0, str(CODE))
    from sp_core.common import PairRecord, read_pair_rgb
    if not torch.cuda.is_available():
        raise RuntimeError("Native CUDA required; no CPU fallback")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    frozen = frozen_files()
    old = json.loads((ROOT / "predictive_tracks/state_tracks_20261023/all_models_20261023/frozen.json").read_text())
    frozen.update(old["old_sources"])
    frozen.update(old["new_sources"])
    own = {str(p): digest(p) for p in HERE.glob("*.py")}
    verify_frozen(frozen)
    data = rows_and_references(model)
    if mode == "audit":
        seen, selected = set(), []
        for row in data:
            group = row[0]["benchmark"], row[0]["category_directory"]
            if group not in seen:
                seen.add(group)
                selected.append(row)
        data = selected
    adapter = importlib.import_module(MODELS[model]).Adapter(
        device="cuda:0", memory_mode="safe", orientation="identity")
    save(run_dir / "config.json", {"model": model, "mode": mode, "cases": len(data),
         "data_scope": "SP300 and IE300, previously exposed; exploratory, not heldout",
         "method": "history mean/max-radius raw persistent-state projection",
         "reliability": {"presence_min": .5, "native_pixel_mass_min": 4., "relative_mass_min": .5, "min_history": 2},
         "model_provenance": adapter.provenance(), "torch": torch.__version__, "cuda": torch.version.cuda,
         "frozen_hashes": frozen, "new_source_hashes": own,
         "validation_overhead": "audit is a separate six-case no-op validation; no extra forecasts inside any track",
         "GSWM_budget": "original eight MC rollouts per case, unchanged; six future steps each",
         "failure_policy": "fail the track on provenance/budget/numeric mismatch; retain partial outputs"})
    run_dir.joinpath("pairs").mkdir()
    reference_audit = json.loads((OUTPUT / "deployment_audit_001.json").read_text())["details"][model]
    expected_traces = {json.dumps(r["neural_trace"], sort_keys=True) for r in reference_audit}
    if len(expected_traces) != 1:
        raise RuntimeError("Ambiguous original native graph reference")
    common_trace = json.loads(next(iter(expected_traces)))
    hooks_type = DDLPHooks if model == "ddlp" else GSWMHooks
    results = []
    for row, ref_path, reference in data:
        await_gate(own_pid=os.getpid())
        record = PairRecord(row["pair_id"], row["category_directory"], int(row["t_ref"]),
                            Path(row["possible_dir"]), Path(row["impossible_dir"]), Path(row["meta_path"]))
        possible, impossible = read_pair_rgb(record)
        pixels_hash = hashlib.sha256(possible.tobytes() + impossible.tobytes()).hexdigest()
        if pixels_hash != reference["source_decoded_pixels_sha256"]:
            raise RuntimeError("Decoded input pixels changed")
        adapter.prediction_calls = row["archive_index"]
        trace, calls = hashlib.sha256(), [0]
        def count(module, inputs):
            calls[0] += 1
            event = module.__class__.__module__, module.__class__.__qualname__, describe(inputs)
            trace.update(json.dumps(event, sort_keys=True).encode())
        counter = torch.nn.modules.module.register_module_forward_pre_hook(count)
        started = time.perf_counter()
        try:
            with hooks_type(adapter, mode, run_dir / "instrumented_sources") as hooks:
                result = adapter.score_pair(possible, impossible)
        finally:
            counter.remove()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        observed = {"module_forward_invocations": calls[0],
                    "ordered_class_and_input_shapes_sha256": trace.hexdigest()}
        if observed != reference.get("neural_call_trace", common_trace):
            raise RuntimeError("Native forward count/order/shapes changed")
        if adapter.prediction_calls != row["archive_index"] + 1:
            raise RuntimeError("Original forecast budget changed")
        if result.frame_spans.tolist() != reference["frame_spans"]:
            raise RuntimeError("Original time grid changed")
        if hashlib.sha256(possible.tobytes() + impossible.tobytes()).hexdigest() != pixels_hash:
            raise RuntimeError("Source pixels were mutated")
        for branch, actual in (("possible", result.possible), ("impossible", result.impossible)):
            if not np.isfinite(actual).all():
                raise RuntimeError("Nonfinite score")
            if mode == "audit" and not np.allclose(actual, reference[f"matched_identity_{branch}_errors"], atol=1e-7, rtol=1e-5):
                raise RuntimeError("No-op instrumentation changed native scores")
        expected_rollouts = 1 if model == "ddlp" else 8
        if len(hooks.audits) != expected_rollouts:
            raise RuntimeError("Original native sample count changed")
        if mode != "audit" and any(len(a["steps"]) != 6 for a in hooks.audits):
            raise RuntimeError("Projection must occur at exactly six native future steps")
        payload = {"benchmark": row["benchmark"], "archive_index": row["archive_index"],
                   "pair_id": row["pair_id"], "category": row["category_directory"], "split": row["split"],
                   "t_ref": int(row["t_ref"]), "frame_spans": result.frame_spans.tolist(),
                   "possible_errors": result.possible.tolist(), "impossible_errors": result.impossible.tolist(),
                   "matched_identity_possible_errors": reference["matched_identity_possible_errors"],
                   "matched_identity_impossible_errors": reference["matched_identity_impossible_errors"],
                   "source_decoded_pixels_sha256": pixels_hash, "reference_pair_sha256": digest(ref_path),
                   "neural_call_trace": observed, "trace_matches_original": True,
                   "wall_seconds": seconds, "projection_audits": hooks.audits,
                   "instrumented_function_hashes": hooks.functions}
        save(run_dir / "pairs" / f"{row['key']}.json", payload)
        results.append(payload)
        write_status(run_dir / "progress.json", {"completed": len(results), "total": len(data), "last": row["key"], "at": time.time()})
        if len(results) % 10 == 0 or mode == "audit":
            print(model, mode, len(results), "/", len(data), row["key"], flush=True)
    verify_frozen(frozen)
    verify_frozen(own)
    report = {"model": model, "mode": mode, "cases": len(results), "original_sources_unchanged": True,
              "all_forward_traces_matched": True, "additional_forwards_per_forecast": 0}
    if mode != "audit":
        report["metrics"] = {task: aggregate([r for r in results if r["benchmark"] == task], model) for task in ("sp", "ie")}
    save(run_dir / "summary.json", report)
    print("COMPLETE", model, mode, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("ddlp", "gswm"), required=True)
    parser.add_argument("--mode", choices=("audit", "feedback", "output_only"), required=True)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    if Path(args.run).name != args.run:
        raise ValueError("Simple new run name required")
    path = HERE / "runs" / args.run / args.model / args.mode
    path.mkdir(parents=True, exist_ok=False)
    try:
        execute(args.model, args.mode, path)
    except BaseException as exc:
        save(path / "FAILED.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
