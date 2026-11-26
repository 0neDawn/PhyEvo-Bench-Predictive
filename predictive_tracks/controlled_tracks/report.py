"""Report every completed treatment, using fixed matched identity controls."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

from .registry import CODE, MODELS, NATIVE_TOLERANCE, OUTPUT, qualification
from .run import write_json

sys.path.insert(0, str(CODE))
from sp_benchmark.protocol_metrics import pairwise_results, summarize


def indicators(possible, impossible, spans, tref, tolerance):
    correct = pairwise_results(possible, impossible).correct
    summary = summarize(possible, impossible, frame_spans=spans,
                        violation_frames=tref, joint_tolerances=(tolerance,),
                        epsilon=1e-12, threshold=0.0, persistence_frames=2)
    hits = np.asarray([bool(ok and t is not None and abs(t-ref) <= tolerance)
                       for ok, t, ref in zip(correct, summary["t_star"], tref)])
    assert np.isclose(hits.mean(), summary["joint_hit"][str(tolerance)])
    return correct, hits


def paired_stats(current, baseline, categories, seed=20261022, repetitions=10000):
    current, baseline = np.asarray(current, bool), np.asarray(baseline, bool)
    categories = np.asarray(categories)
    if current.shape != baseline.shape or current.shape != categories.shape or not current.size:
        raise ValueError("Paired observations/category alignment is required")
    difference = current.astype(float) - baseline.astype(float)
    rng = np.random.default_rng(seed)
    sampled = np.concatenate([rng.choice(np.flatnonzero(categories == category),
                                         size=(repetitions, int((categories == category).sum())),
                                         replace=True)
                              for category in sorted(set(categories.tolist()))], axis=1)
    boot = difference[sampled].mean(axis=1)
    gained = int((current & ~baseline).sum())
    lost = int((~current & baseline).sum())
    discordant = gained + lost
    p = (min(1.0, 2 * sum(math.comb(discordant, k)
                         for k in range(min(gained, lost) + 1)) / (2 ** discordant))
         if discordant else 1.0)
    return {"baseline": float(baseline.mean()), "current": float(current.mean()),
            "delta": float(difference.mean()),
            "paired_stratified_bootstrap_95_ci": np.quantile(boot, [0.025, 0.975]).tolist(),
            "bootstrap_repetitions": repetitions, "gained_pairs": gained,
            "lost_pairs": lost, "exact_mcnemar_p": p}


def collect(root):
    comparisons, jobs, failures, ablations = [], [], [], []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir() or not (run_dir / "run_config.json").exists():
            continue
        config = json.loads((run_dir / "run_config.json").read_text())
        if (run_dir / "FAILED.json").exists():
            failures.append({"run": run_dir.name, **json.loads((run_dir / "FAILED.json").read_text())})
        if not (run_dir / "summary.json").exists():
            jobs.append({"run": run_dir.name, "status": "failed" if (run_dir / "FAILED.json").exists() else "incomplete"})
            continue
        jobs.append({"run": run_dir.name, "status": "complete",
                     "model": config["model"], "orientation": config["orientation"],
                     "split": config["split"], "smoke_subset": bool(config["limit_per_category"])})
        condition = (config.get("history_schedule") if config.get("history_schedule", "recent8") != "recent8"
                     else config["orientation"])
        if config.get("correction", "none") != "none":
            condition = config["correction"]
        if condition == "identity" or config["limit_per_category"]:
            continue
        if not config.get("reference_run") and not config.get("paired_native_control"):
            raise RuntimeError(f"Treatment has no matched control: {run_dir}")
        all_pairs = [json.loads(p.read_text()) for p in sorted((run_dir / "pairs").glob("*.json"))]
        for track in config["benchmarks"]:
            pairs = [p for p in all_pairs if p["benchmark"] == track]
            if not pairs:
                continue
            tolerance = NATIVE_TOLERANCE[config["model"]]
            args = (np.asarray(pairs[0]["frame_spans"]), np.asarray([p["t_ref"] for p in pairs]), tolerance)
            current = indicators(np.asarray([p["possible_errors"] for p in pairs]),
                                 np.asarray([p["impossible_errors"] for p in pairs]), *args)
            baseline = indicators(np.asarray([p["matched_identity_possible_errors"] for p in pairs]),
                                  np.asarray([p["matched_identity_impossible_errors"] for p in pairs]), *args)
            categories = [p["category"] for p in pairs]
            if all("cached_controls" in p for p in pairs):
                held = indicators(np.asarray([p["cached_controls"]["persistence_possible_errors"] for p in pairs]),
                                  np.asarray([p["cached_controls"]["persistence_impossible_errors"] for p in pairs]), *args)
                for endpoint, now, original, hold in zip(("paired_accuracy", "native_joint_hit"), current, baseline, held):
                    for name, left, right in (("persistence_vs_native",hold,original),
                                               ("method_vs_persistence",now,hold)):
                        ablations.append({"run":run_dir.name,"model":config["model"],
                                          "condition":condition,"comparison":name,"benchmark":track,
                                          "split":config["split"],"num_pairs":len(pairs),"endpoint":endpoint,
                                          "additional_neural_forwards":0,
                                          **paired_stats(left,right,categories)})
            for endpoint, now, before in zip(("paired_accuracy", "native_joint_hit"), current, baseline):
                comparisons.append({"run": run_dir.name, "model": config["model"],
                                    "condition": condition, "benchmark": track,
                                    "split": config["split"], "num_pairs": len(pairs),
                                    "endpoint": endpoint, "qualification": qualification(config["model"], config.get("correction", "none")),
                                    "median_pair_seconds": float(np.median([p["seconds"] for p in pairs])),
                                    "timing_scope": config.get("timing_scope", "adapter wall time, including optional tracing/capture overhead"),
                                    "mean_preprocessing_seconds": float(np.mean([p["track_audit"]["preprocessing_seconds"] for p in pairs])),
                                    **paired_stats(now, before, categories)})
    # Include all reported conditions/endpoints/models, not only favourable ones.
    ordered = sorted(comparisons + ablations, key=lambda x: x["exact_mcnemar_p"])
    previous = 0.0
    for rank, row in enumerate(ordered):
        previous = max(previous, min(1.0, (len(ordered)-rank)*row["exact_mcnemar_p"]))
        row["holm_p_all_reported_comparisons"] = previous
    return {"interpretation": "Exploratory development results unless explicitly marked confirmation; no best-of-condition selection.",
            "original_model_denominator": len(MODELS), "comparisons": comparisons,
            "same_forward_cached_ablations": ablations,
            "jobs": jobs, "failed_jobs": failures,
            "uncertainty_note": "Bootstrap intervals are marginal, not simultaneous; Holm correction covers all p-values in this report."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()
    if Path(args.output_name).name != args.output_name:
        raise ValueError("Use a simple output name")
    payload = collect(OUTPUT)
    output = OUTPUT / args.output_name
    write_json(output, payload)
    for row in payload["comparisons"]:
        if row["endpoint"] == "paired_accuracy":
            print(f"{row['model']:18s} {row['condition']:16s} {row['benchmark']:2s} "
                  f"N={row['num_pairs']:3d} PA {100*row['baseline']:.2f} -> {100*row['current']:.2f} "
                  f"delta={100*row['delta']:+.2f} pp CI={row['paired_stratified_bootstrap_95_ci']}")
    print(f"Saved all comparisons and failures: {output}")


if __name__ == "__main__":
    main()
