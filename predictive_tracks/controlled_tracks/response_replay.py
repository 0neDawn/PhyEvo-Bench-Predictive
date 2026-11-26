"""Locked exploratory analytic replay of existing development caches only."""

import argparse
import json
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

from .registry import CODE, OUTPUT, NATIVE_TOLERANCE, ROOT
from .run import digest, frozen_files, verify_frozen, write_json, records_and_split
from .native_cache import NativeCache
from .orthogonal_boundary import align_orthogonal_boundary
from .response_calibration import response_calibration
from .full_metrics_report import aggregate

sys.path.insert(0, str(CODE))
from sp_benchmark.protocol_metrics import summarize


VARIANTS = ("frozen_parallel", "response", "hard_response", "random_norm")
RGB_MODELS = ("ocvp_seq", "savi_transformer", "ddlp", "gswm")


def comparison(rows, model, reference="native", repetitions=10000):
    tolerance = str(NATIVE_TOLERANCE[model])
    flags = []
    for prefix in ("matched_identity_" if reference == "native" else reference + "_", ""):
        ep = np.asarray([r[prefix + "possible_errors"] for r in rows])
        ei = np.asarray([r[prefix + "impossible_errors"] for r in rows])
        metrics = summarize(ep, ei, frame_spans=np.asarray(rows[0]["frame_spans"]),
                            violation_frames=np.asarray([r["t_ref"] for r in rows]),
                            epsilon=1e-12, threshold=0., persistence_frames=2,
                            joint_tolerances=(NATIVE_TOLERANCE[model],))
        correct = ei.mean(1) > ep.mean(1)
        delta = np.asarray(metrics["delta_t"], dtype=np.float64)
        hit = correct & np.isfinite(delta) & (np.abs(delta) <= int(tolerance))
        flags.append((correct, hit))
    result = {}
    rng = np.random.default_rng(20261022)
    categories = sorted({r["category"] for r in rows})
    groups = [np.flatnonzero([r["category"] == c for r in rows]) for c in categories]
    indices = np.concatenate([rng.choice(g, (repetitions, len(g)), replace=True) for g in groups], axis=1)
    for index, endpoint in enumerate(("paired_accuracy", "joint_hit")):
        before, after = flags[0][index], flags[1][index]
        difference = after.astype(float) - before.astype(float)
        gain, loss = int((after & ~before).sum()), int((before & ~after).sum())
        import math
        discordant = gain + loss
        exact_p = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(gain, loss) + 1)) / 2**discordant) if discordant else 1.
        result[endpoint] = {"reference": reference, "gained": gain, "lost": loss,
                            "delta": float(difference.mean()),
                            "stratified_bootstrap_95": np.quantile(difference[indices].mean(1), [.025, .975]).tolist(),
                            "exact_mcnemar_p": exact_p, "exploratory_not_independent_confirmation": True}
    return result


def diagnostic(prediction, corrected, history, possible_target):
    # Targets enter ONLY after the intervention. This measures representation
    # curvature, not true 3D object motion or a contact oracle.
    p, c, h, y = (x.float() for x in (prediction, corrected, history, possible_target))
    curvature = h[:, -1:] - 2 * y[:, :1] + y[:, 1:2]
    correction = c[:, :1] - p[:, :1]
    rms = lambda x: float(x.square().mean().sqrt())
    return {
        "possible_representation_rmse_before": rms(p - y),
        "possible_representation_rmse_after": rms(c - y),
        "target_boundary_curvature_rms": rms(curvature),
        "correction_curvature_dot_per_element": float((correction * curvature).mean()),
        "first_step_possible_rmse_before": rms(p[:, :1] - y[:, :1]),
        "first_step_possible_rmse_after": rms(c[:, :1] - y[:, :1]),
        "not_a_physical_state_measurement": True,
    }


def execute(directory):
    torch.set_num_threads(4)
    plan_path = Path(__file__).with_name("response_experiment.json")
    plan = json.loads(plan_path.read_text())
    frozen = frozen_files()
    old_plan = Path(__file__).with_name("confirmation_plan.json")
    frozen.update(json.loads(old_plan.read_text())["locked_source_hashes"])
    frozen[str(old_plan)] = digest(old_plan)
    verify_frozen(frozen)
    new_paths = [plan_path, Path(__file__), Path(__file__).with_name("response_calibration.py"),
                 Path(__file__).with_name("test_response_calibration.py")]
    new_paths.extend(CODE.rglob("track_response*.py"))
    code_hashes = {str(p.resolve()): digest(p) for p in new_paths}
    write_json(directory / "run_config.json", {
        "plan": plan, "new_code_hashes": code_hashes, "frozen_hashes": frozen,
        "scope": "development_only_30_SP_30_IE_per_model", "torch_version": torch.__version__,
        "device": "cpu", "neural_forwards": 0, "extra_neural_forwards_per_deployment": 0,
        "timing_scope": "analytic replay; original neural compute excluded, not end-to-end latency",
    })
    (directory / "code_snapshot").mkdir()
    for p in new_paths:
        destination = directory / "code_snapshot" / p.resolve().relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(p.read_bytes())
    def forbid(module, inputs):
        raise RuntimeError("Neural forward forbidden in cached replay")
    guard = torch.nn.modules.module.register_module_forward_pre_hook(forbid)
    report = {"plan": plan, "scope": "30 development pairs per task per model; NOT 300",
              "models": {}, "neural_forwards": 0}
    all_comparisons = []
    try:
        for model in plan["models"]:
            source = OUTPUT / f"{model}_boundary_dev_001"
            config = json.loads((source / "run_config.json").read_text())
            if config["split"] != "development" or config["limit_per_category"] or not config["cache_native_states"]:
                raise RuntimeError("Only full development native caches are permitted")
            expected = {(task, r["pair_id"]) for task in ("sp", "ie")
                        for r in records_and_split(task) if r["split"] == "development"}
            model_dir = directory / model
            (model_dir / "pairs").mkdir(parents=True)
            rows = {v: [] for v in VARIANTS}
            seen = set()
            for pair_path in sorted((source / "pairs").glob("*.json")):
                original = json.loads(pair_path.read_text())
                identifier = original["benchmark"], original["pair_id"]
                if identifier not in expected or identifier in seen:
                    raise RuntimeError("Unexpected or duplicate pair")
                seen.add(identifier)
                cache_path = source / "native_states" / pair_path.with_suffix(".pt").name
                if digest(cache_path) != original["native_state_cache_sha256"]:
                    raise RuntimeError("Native cache checksum changed")
                data = torch.load(cache_path, map_location="cpu", weights_only=True)
                evaluator = NativeCache(SimpleNamespace(device=torch.device("cpu")), model)
                evaluator.data = data
                evaluator.targets = [data["possible_target"], data["impossible_target"]]
                with torch.inference_mode():
                    raw = evaluator.errors(data["prediction"])
                    baseline = [np.asarray(original[f"matched_identity_{b}_errors"]) for b in ("possible", "impossible")]
                    for actual, saved in zip(raw, baseline):
                        if not np.allclose(actual, saved, rtol=1e-5, atol=1e-7):
                            raise RuntimeError("Native baseline replay mismatch")
                    if bool(raw[1].mean() > raw[0].mean()) != bool(baseline[1].mean() > baseline[0].mean()):
                        raise RuntimeError("Native baseline paired decision changed")
                    payload = {"source_pair": str(pair_path), "source_pair_sha256": digest(pair_path),
                               "source_cache_sha256": digest(cache_path), "variants": {}}
                    old_errors = None
                    for variant in VARIANTS:
                        started = time.perf_counter()
                        if variant == "frozen_parallel":
                            corrected, audit = align_orthogonal_boundary(data["prediction"], data["history"],
                                                                         rgb=model in RGB_MODELS, component="parallel")
                        else:
                            corrected, audit = response_calibration(data["prediction"], data["history"],
                                                                   rgb=model in RGB_MODELS, mode=variant)
                        elapsed = time.perf_counter() - started
                        errors = evaluator.errors(corrected)
                        if variant == "frozen_parallel":
                            old_errors = errors
                            old_row = json.loads((OUTPUT / f"{model}_parallel_dev_001" / "pairs" / pair_path.name).read_text())
                            for value, name in zip(errors, ("possible_errors", "impossible_errors")):
                                if not np.allclose(value, old_row[name], rtol=1e-5, atol=1e-7):
                                    raise RuntimeError("Frozen parallel replay mismatch")
                        row = {k: original[k] for k in ("benchmark", "pair_id", "category", "split", "frame_spans", "t_ref")}
                        row.update(possible_errors=errors[0].tolist(), impossible_errors=errors[1].tolist(),
                                   matched_identity_possible_errors=baseline[0].tolist(),
                                   matched_identity_impossible_errors=baseline[1].tolist(),
                                   frozen_parallel_possible_errors=old_errors[0].tolist(),
                                   frozen_parallel_impossible_errors=old_errors[1].tolist(),
                                   analytic_seconds=elapsed, audit=audit,
                                   diagnostic=diagnostic(data["prediction"], corrected, data["history"], data["possible_target"]))
                        rows[variant].append(row)
                        payload["variants"][variant] = row
                write_json(model_dir / "pairs" / pair_path.name, payload)
            if seen != expected or len(seen) != 60:
                raise RuntimeError("Incomplete development dataset")
            model_report = {"variants": {}}
            for variant in VARIANTS:
                entry = {}
                for task in ("sp", "ie"):
                    selected = [r for r in rows[variant] if r["benchmark"] == task]
                    metrics = aggregate(selected, model)
                    metrics["primary_vs_native"] = comparison(selected, model)
                    metrics["primary_vs_frozen_parallel"] = comparison(selected, model, reference="frozen_parallel")
                    metrics["diagnostics_by_type"] = {}
                    for category in sorted({r["category"] for r in selected}):
                        values = [r for r in selected if r["category"] == category]
                        dkeys = [k for k in values[0]["diagnostic"] if k != "not_a_physical_state_measurement"]
                        metrics["diagnostics_by_type"][category] = {
                            key: float(np.mean([r["diagnostic"][key] for r in values])) for key in dkeys}
                        if variant != "frozen_parallel":
                            metrics["diagnostics_by_type"][category]["retained_correction_ratio_mean"] = float(np.mean([
                                r["audit"]["batch"][0]["retained_correction_ratio"] for r in values]))
                    entry[task] = metrics
                    if variant == "response":
                        for ref in ("primary_vs_native", "primary_vs_frozen_parallel"):
                            for endpoint, value in metrics[ref].items():
                                value.update(model=model, benchmark=task, endpoint=endpoint)
                                all_comparisons.append(value)
                model_report["variants"][variant] = entry
            tolerance = str(NATIVE_TOLERANCE[model])
            primary = model_report["variants"]["response"]
            model_report["passes_development_gate"] = all(
                primary[t]["corrected"]["overall"]["paired_accuracy"] > primary[t]["native_baseline"]["overall"]["paired_accuracy"]
                and primary[t]["corrected"]["overall"]["joint_hit"][tolerance] >= primary[t]["native_baseline"]["overall"]["joint_hit"][tolerance]
                for t in ("sp", "ie"))
            model_report["analytic_seconds_median"] = float(np.median([r["analytic_seconds"] for r in rows["response"]]))
            report["models"][model] = model_report
            write_json(model_dir / "summary.json", model_report)
            print(model, {t: {"baseline_PA": primary[t]["native_baseline"]["overall"]["paired_accuracy"],
                             "response_PA": primary[t]["corrected"]["overall"]["paired_accuracy"]} for t in ("sp", "ie")}, flush=True)
        # Two references x two endpoints x two tasks x eight models = 64 tests.
        previous = 0.
        for rank, value in enumerate(sorted(all_comparisons, key=lambda x: x["exact_mcnemar_p"])):
            previous = max(previous, min(1., (len(all_comparisons) - rank) * value["exact_mcnemar_p"]))
            value["holm_p_64_exploratory_primary_tests"] = previous
        report["models_passing_development_gate"] = sum(m["passes_development_gate"] for m in report["models"].values())
        report["development_gate_passed"] = report["models_passing_development_gate"] >= 5
        report["primary_comparisons"] = all_comparisons
        verify_frozen(frozen)
        verify_frozen(code_hashes)
        report["provenance_verified"] = True
        write_json(directory / "summary.json", report)
        print("COMPLETE", directory, "development gate:", report["models_passing_development_gate"], "/ 9", flush=True)
    finally:
        guard.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError("Simple exclusive run name required")
    directory = OUTPUT / args.name
    directory.mkdir(exist_ok=False, parents=True)
    try:
        execute(directory)
    except BaseException as exc:
        write_json(directory / "FAILED.json", {"exception": repr(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
