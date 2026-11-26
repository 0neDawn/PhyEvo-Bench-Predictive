"""Replay frozen adapters or run a separately labelled budget-matched track."""

from __future__ import annotations

import argparse
import csv
import contextlib
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from .orientation import MODES
from .registry import CODE, MODELS, HISTORY_MODELS, MOMENT_MODELS, CONTINUITY_MODELS, BOUNDARY_MODELS, ORTHOGONAL_MODELS, PARALLEL_MODELS, NATIVE_TOLERANCE, OUTPUT, ROOT, archive, qualification
from .history_context import HISTORY_SCHEDULES, read_history_view

PLAN = Path(__file__).with_name("experiment.json")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def records_and_split(track):
    with (archive(track) / "evaluated_pairs.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    plan = json.loads(PLAN.read_text())
    grouped = {}
    for index, row in enumerate(rows):
        row["archive_index"] = index
        grouped.setdefault(row["category_directory"], []).append(row)
    if len(rows) != 300 or len(grouped) != 3 or any(len(v) != 100 for v in grouped.values()):
        raise RuntimeError("The frozen 300-pair dataset changed")
    for group in grouped.values():
        ordered = sorted(group, key=lambda r: hashlib.sha256(
            (plan["split_salt"] + r["pair_id"]).encode()).hexdigest())
        for rank, row in enumerate(ordered):
            row["split"] = ("development" if rank < plan["development_pairs_per_category"]
                            else "confirmation")
    return rows


def frozen_files():
    paths = {PLAN.resolve()}
    expected = {}
    for track in ("sp", "ie"):
        base = archive(track)
        paths.update((base / name).resolve() for name in
                     ("evaluated_pairs.csv", "CODE_PROVENANCE.json", "RESULTS.json", "SHA256SUMS.txt"))
        for entry in json.loads((base / "CODE_PROVENANCE.json").read_text())["files"]:
            path = Path(entry["source_path"])
            expected[path] = entry["sha256"]
            paths.add(path)
    for path, wanted in expected.items():
        if digest(path) != wanted:
            raise RuntimeError(f"Frozen mainline source differs from archive: {path}")
    return {str(p): digest(p) for p in sorted(paths)}


def verify_frozen(before):
    for path, wanted in before.items():
        if digest(path) != wanted:
            raise RuntimeError(f"Frozen file changed during experiment: {path}")


def scalar_metrics(summary):
    return {k: v for k, v in summary.items()
            if k not in ("margin", "relative_error", "t_star", "delta_t")}


def execute(args, run_dir):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    sys.path.insert(0, str(CODE))
    import torch
    from sp_core.common import PairRecord, read_pair_rgb
    from sp_benchmark.protocol_metrics import summarize

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("GPU unavailable; no silent slow CPU fallback")
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(4)
    before = frozen_files()
    if args.split == "confirmation":
        confirmation_path = PLAN.with_name("confirmation_plan.json")
        confirmation = json.loads(confirmation_path.read_text())
        if (args.correction != confirmation["selected_correction"] or
            args.model not in confirmation["models"] or not args.paired_native_control or
            args.orientation != "identity" or args.history_schedule != "recent8" or
            args.limit_per_category or args.cache_native_states or
            sorted(args.benchmarks) != ["ie", "sp"] or not args.trace_neural_calls):
            raise RuntimeError("Arguments differ from the frozen confirmation plan")
        for path, wanted in confirmation["locked_source_hashes"].items():
            if digest(path) != wanted:
                raise RuntimeError(f"Confirmation source changed after freezing: {path}")
            before[path] = wanted
        before[str(confirmation_path)] = digest(confirmation_path)
        with (run_dir / confirmation_path.name).open("x") as handle:
            handle.write(confirmation_path.read_text())
    with (run_dir / "experiment.json").open("x") as f:
        f.write(PLAN.read_text())
    source_snapshot = {str(p): digest(p) for p in
                       Path(__file__).parent.glob("*.py")}
    modules = MODELS if args.history_schedule == "recent8" else HISTORY_MODELS
    if args.correction != "none":
        modules = MOMENT_MODELS
    if args.correction in ("history_rate_ball", "history_persistence"):
        modules = CONTINUITY_MODELS
    if args.correction == "boundary_alignment":
        modules = BOUNDARY_MODELS
    if args.correction == "orthogonal_boundary":
        modules = ORTHOGONAL_MODELS
    if args.correction == "parallel_boundary":
        modules = PARALLEL_MODELS
    entry_path = CODE / (modules[args.model].replace(".", "/") + ".py")
    source_snapshot[str(entry_path)] = digest(entry_path)
    # Include additive wrapper dependencies as well as the selected entry point.
    source_snapshot.update({str(p): digest(p) for p in CODE.rglob("track_*.py")})
    for source_name in source_snapshot:
        source = Path(source_name)
        target = run_dir / "code_snapshot" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(source.read_bytes())
    active_plan = PLAN if args.history_schedule == "recent8" else PLAN.with_name("context_experiment.json")
    if args.correction != "none":
        active_plan = PLAN.with_name("moment_experiment.json")
    if args.correction in ("history_rate_ball", "history_persistence"):
        active_plan = PLAN.with_name("continuity_experiment.json")
    if args.correction == "boundary_alignment":
        active_plan = PLAN.with_name("boundary_experiment.json")
    if args.correction == "orthogonal_boundary":
        active_plan = PLAN.with_name("orthogonal_experiment.json")
    if args.correction == "parallel_boundary":
        active_plan = PLAN.with_name("parallel_experiment.json")
    before[str(active_plan.resolve())] = digest(active_plan)
    if active_plan != PLAN:
        with (run_dir / active_plan.name).open("x") as handle:
            handle.write(active_plan.read_text())
    write_json(run_dir / "run_config.json", {
        **vars(args), "run_dir": str(run_dir), "frozen_hashes": before,
        "new_code_hashes": source_snapshot,
        "qualification": qualification(args.model, args.correction),
        "forecast_budget": "Original per-pair forward graph, sample count and horizon; no added model calls",
        "split_plan_sha256": digest(PLAN), "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(args.device),
        "cudnn_version": torch.backends.cudnn.version(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "active_plan_sha256": digest(active_plan),
        "history_frames": list(HISTORY_SCHEDULES[args.history_schedule]),
    })
    adapter = importlib.import_module(modules[args.model]).Adapter(
        device=args.device, memory_mode="safe", orientation=args.orientation)
    if args.correction == "history_persistence":
        adapter.persistence = True
    write_json(run_dir / "model_provenance.json", adapter.provenance())
    (run_dir / "pairs").mkdir()
    if args.cache_native_states:
        (run_dir / "native_states").mkdir()
    final = {}
    matched_config = None
    if args.reference_run:
        matched_root = (OUTPUT / args.reference_run).resolve()
        if not matched_root.is_relative_to(OUTPUT.resolve()):
            raise ValueError("Matched reference must be an additive run")
        matched_config = json.loads((matched_root / "run_config.json").read_text())
        if (matched_config["model"] != args.model or matched_config["orientation"] != "identity"
            or matched_config.get("history_schedule", "recent8") != "recent8"
            or matched_config.get("correction", "none") != "none"):
            raise ValueError("Matched reference must use the same model and identity condition")
        if not (matched_root / "summary.json").is_file():
            raise ValueError("Matched identity reference did not complete")
    for track in args.benchmarks:
        rows = records_and_split(track)
        chosen = [r for r in rows if args.split == "all" or r["split"] == args.split]
        if args.limit_per_category:
            counts = {}
            selected = []
            for row in chosen:
                category = row["category_directory"]
                if counts.get(category, 0) < args.limit_per_category:
                    selected.append(row)
                    counts[category] = counts.get(category, 0) + 1
            chosen = selected
        reference_path = archive(track) / "models" / args.model / "raw/error_arrays.npz"
        with np.load(reference_path, allow_pickle=False) as raw:
            reference = {key: raw[key].copy() for key in raw.files}
        results = []
        for row in chosen:
            index = row["archive_index"]
            if reference["pair_ids"][index] != row["pair_id"]:
                raise RuntimeError("Archive pair ordering mismatch")
            record = PairRecord(row["pair_id"], row["category_directory"], int(row["t_ref"]),
                                Path(row["possible_dir"]), Path(row["impossible_dir"]),
                                Path(row["meta_path"]))
            possible, impossible = read_pair_rgb(record)
            original_reference_hash = hashlib.sha256(possible.tobytes() + impossible.tobytes()).hexdigest()
            view_started = time.perf_counter()
            if args.history_schedule != "recent8":
                possible, impossible = read_history_view(record, args.history_schedule)
            view_seconds = time.perf_counter() - view_started
            source_hash = hashlib.sha256(possible.tobytes() + impossible.tobytes()).hexdigest()
            # Preserve original row-index-conditioned SAVi seeds despite subset selection.
            adapter.prediction_calls = index
            torch.cuda.synchronize()
            started = time.perf_counter()
            neural_trace = hashlib.sha256()
            neural_call_count = [0]
            def describe_input(value):
                if torch.is_tensor(value):
                    return {"shape": list(value.shape), "dtype": str(value.dtype)}
                if isinstance(value, (tuple, list)):
                    return [describe_input(v) for v in value]
                if isinstance(value, dict):
                    return {str(k): describe_input(v) for k, v in value.items()}
                return type(value).__name__
            def trace_call(module, inputs):
                neural_call_count[0] += 1
                event = (module.__class__.__module__, module.__class__.__qualname__, describe_input(inputs))
                neural_trace.update(json.dumps(event, sort_keys=True).encode())
            trace_handle = (torch.nn.modules.module.register_module_forward_pre_hook(trace_call)
                            if args.trace_neural_calls else None)
            cache = None
            if args.cache_native_states or args.paired_native_control:
                from .native_cache import NativeCache
                cache = NativeCache(adapter,args.model)
            try:
                with cache if cache is not None else contextlib.nullcontext():
                    errors = adapter.score_pair(possible, impossible)
            finally:
                if trace_handle is not None:
                    trace_handle.remove()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if hashlib.sha256(possible.tobytes() + impossible.tobytes()).hexdigest() != source_hash:
                raise RuntimeError("Adapter modified source arrays")
            if not np.array_equal(errors.frame_spans, reference["frame_spans"]):
                raise RuntimeError("Native time grid changed")
            entry = {"pair_id": row["pair_id"], "benchmark": track,
                     "category": row["category_directory"], "archive_index": index,
                     "split": row["split"], "t_ref": int(row["t_ref"]),
                     "possible_errors": errors.possible.tolist(),
                     "impossible_errors": errors.impossible.tolist(),
                     "frame_spans": errors.frame_spans.tolist(),
                     "reference_possible_errors": reference["possible_errors"][index].tolist(),
                     "reference_impossible_errors": reference["impossible_errors"][index].tolist(),
                     "source_decoded_pixels_sha256": source_hash,
                     "original_reference_pixels_sha256": original_reference_hash,
                     "history_frames": list(HISTORY_SCHEDULES[args.history_schedule]),
                     "view_io_seconds": view_seconds,
                     "seconds": elapsed, "track_audit": adapter.last_track_audit}
            if args.trace_neural_calls:
                entry["neural_call_trace"] = {"module_forward_invocations": neural_call_count[0],
                                               "ordered_class_and_input_shapes_sha256": neural_trace.hexdigest()}
            if cache is not None:
                cache_started = time.perf_counter()
                cache.finish_targets(possible,impossible)
                entry["cached_controls"] = cache.audit_and_controls(errors)
                if args.cache_native_states:
                    cache_path = run_dir / "native_states" / f"{track}_{index:03d}.pt"
                    with cache_path.open("xb") as handle:
                        torch.save(cache.data, handle)
                    entry["native_state_cache_sha256"] = digest(cache_path)
                entry["cache_controls_and_storage_seconds"] = time.perf_counter()-cache_started
                if args.paired_native_control:
                    entry["matched_identity_source"] = "same original forward before forecast correction; original native scoring"
                    for branch in ("possible", "impossible"):
                        entry[f"matched_identity_{branch}_errors"] = entry["cached_controls"][f"native_prediction_{branch}_errors"]
            entry["baseline_replay_max_abs_delta"] = float(max(
                np.max(np.abs(errors.possible - reference["possible_errors"][index])),
                np.max(np.abs(errors.impossible - reference["impossible_errors"][index]))))
            if matched_config is not None:
                matched_path = matched_root / "pairs" / f"{track}_{index:03d}.json"
                matched = json.loads(matched_path.read_text())
                if (matched["pair_id"] != row["pair_id"] or
                    matched["source_decoded_pixels_sha256"] != original_reference_hash):
                    raise RuntimeError("Matched identity baseline input differs")
                if not args.paired_native_control:
                    entry["matched_identity_possible_errors"] = matched["possible_errors"]
                    entry["matched_identity_impossible_errors"] = matched["impossible_errors"]
                entry["matched_identity_sha256"] = digest(matched_path)
                if cache is not None:
                    for branch in ("possible", "impossible"):
                        raw = entry["cached_controls"][f"native_prediction_{branch}_errors"]
                        if not np.allclose(raw,matched[f"{branch}_errors"],rtol=1e-5,atol=1e-7):
                            raise RuntimeError("Native pre-correction prediction differs from matched control")
                if args.trace_neural_calls and "neural_call_trace" in matched:
                    if entry["neural_call_trace"] != matched["neural_call_trace"]:
                        raise RuntimeError("Native neural call count/order/input shapes changed")
                    entry["neural_trace_matches_identity"] = True
            write_json(run_dir / "pairs" / f"{track}_{index:03d}.json", entry)
            results.append(entry)
            if not args.quiet_pairs:
                print(json.dumps({"model": args.model, "orientation": args.orientation,
                                  "pair": row["pair_id"], "seconds": round(elapsed, 3),
                                  "flip": entry["track_audit"]["flip"],
                                  "correct": bool(errors.impossible.mean() > errors.possible.mean())}), flush=True)
        ep = np.asarray([r["possible_errors"] for r in results])
        ei = np.asarray([r["impossible_errors"] for r in results])
        bp = np.asarray([r["reference_possible_errors"] for r in results])
        bi = np.asarray([r["reference_impossible_errors"] for r in results])
        spans = reference["frame_spans"]
        tref = np.asarray([r["t_ref"] for r in results])
        kwargs = dict(frame_spans=spans, violation_frames=tref, epsilon=1e-12,
                      threshold=0.0, persistence_frames=2,
                      joint_tolerances=(NATIVE_TOLERANCE[args.model],))
        current = scalar_metrics(summarize(ep, ei, **kwargs))
        baseline = scalar_metrics(summarize(bp, bi, **kwargs))
        final[track] = {"current": current, "archived_baseline_same_pairs": baseline,
                        "paired_accuracy_delta": current["paired_accuracy"] - baseline["paired_accuracy"],
                        "max_baseline_replay_delta": max(r["baseline_replay_max_abs_delta"] for r in results),
                        "flip_count": sum(r["track_audit"]["flip"] for r in results),
                        "seconds": sum(r["seconds"] for r in results)}
        if matched_config is not None or args.paired_native_control:
            mp = np.asarray([r["matched_identity_possible_errors"] for r in results])
            mi = np.asarray([r["matched_identity_impossible_errors"] for r in results])
            matched_summary = scalar_metrics(summarize(mp, mi, **kwargs))
            final[track]["matched_identity_same_pairs"] = matched_summary
            final[track]["matched_paired_accuracy_delta"] = current["paired_accuracy"] - matched_summary["paired_accuracy"]
            final[track]["matched_max_abs_delta"] = float(max(np.abs(ep-mp).max(), np.abs(ei-mi).max()))
        with (run_dir / f"{track}_error_arrays.npz").open("xb") as handle:
            np.savez_compressed(handle, possible_errors=ep, impossible_errors=ei,
                                frame_spans=spans, violation_frames=tref,
                                pair_ids=np.asarray([r["pair_id"] for r in results]),
                                categories=np.asarray([r["category"] for r in results]))
    verify_frozen(before)
    write_json(run_dir / "summary.json", final)
    print(json.dumps({"completed": str(run_dir), "summary": final}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--orientation", choices=MODES, required=True)
    parser.add_argument("--benchmarks", nargs="+", choices=("sp", "ie"), default=["sp", "ie"])
    parser.add_argument("--split", choices=("development", "confirmation", "all"), default="development")
    parser.add_argument("--limit-per-category", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--name", required=True)
    parser.add_argument("--reference-run", default=None,
                        help="Completed, same-environment identity control (never select the better baseline)")
    parser.add_argument("--history-schedule", choices=HISTORY_SCHEDULES, default="recent8")
    parser.add_argument("--correction", choices=("none", "history_mean_envelope", "history_rate_ball", "history_persistence", "boundary_alignment", "orthogonal_boundary", "parallel_boundary"), default="none")
    parser.add_argument("--trace-neural-calls", action="store_true",
                        help="Audit ordered module calls and input shapes; no neural computation added")
    parser.add_argument("--cache-native-states", action="store_true",
                        help="Development-only existing-state capture and pure-tensor persistence control; no new model calls")
    parser.add_argument("--paired-native-control", action="store_true",
                        help="Score the unmodified native forecast from the same original forward as the primary control; no extra model call")
    parser.add_argument("--quiet-pairs", action="store_true",help="Do not print intermediate pair outcomes")
    args = parser.parse_args()
    if args.limit_per_category < 0 or not args.name or Path(args.name).name != args.name:
        raise ValueError("Invalid run name/limit")
    if args.history_schedule != "recent8" and args.orientation != "identity":
        raise ValueError("Run context and orientation hypotheses separately")
    if args.correction != "none" and (args.history_schedule != "recent8" or args.orientation != "identity"):
        raise ValueError("Run prediction correction separately from input changes")
    if args.cache_native_states and (args.correction == "none" or args.split != "development"):
        raise ValueError("Native state caching is restricted to development-only correction runs")
    if args.paired_native_control and args.correction == "none":
        raise ValueError("Same-forward native control requires an output-correction track")
    run_dir = OUTPUT / args.name
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        execute(args, run_dir)
    except BaseException as exc:
        write_json(run_dir / "FAILED.json", {"exception": repr(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
