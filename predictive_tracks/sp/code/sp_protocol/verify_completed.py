#!/usr/bin/env python3
"""Recompute completed corrected runs and create an additive audit report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from sp_core.common import PAIRS_CSV, sha256, write_json_new
from sp_protocol.run import metric_view


EXPECTED = ("vjepa2_corrected", "videomaev2_corrected", "gswm_corrected")
OLD = {
    "vjepa2_corrected": Path(
        "predictive_tracks/sp/runs/sp_all5_formal_001/vjepa2_retry_001/summary.json"
    ),
    "videomaev2_corrected": Path(
        "predictive_tracks/sp/runs/sp_all5_formal_001/videomaev2/summary.json"
    ),
    "gswm_corrected": Path(
        "predictive_tracks/sp/runs/sp_all5_formal_001/gswm_obj3d/summary.json"
    ),
}
SCALARS = (
    "num_pairs",
    "paired_accuracy",
    "tie_rate",
    "auc",
    "detection_coverage",
    "localized_and_correct_fraction",
    "localization_mae",
    "localization_mae_denominator",
)


def same(left, right) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    return left == right


def selected_metrics(summary: dict) -> dict:
    overall = summary["overall"]
    return {
        "paired_accuracy": overall["paired_accuracy"],
        "auc": overall["auc"],
        "localization_mae": overall["localization_mae"],
        "localization_mae_denominator": overall["localization_mae_denominator"],
        "localized_and_correct_fraction": overall["localized_and_correct_fraction"],
        "joint_hit_at_2": overall["joint_hit"]["2"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    report_path = run_root / "verification.json"
    comparison_path = run_root / "comparison_to_frozen_baseline.json"
    for path in (report_path, comparison_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    manifest_hash = sha256(PAIRS_CSV)
    reports = {}
    comparisons = {}
    for model in EXPECTED:
        directory = run_root / model
        config = json.loads((directory / "run_config.json").read_text())
        stored = json.loads((directory / "summary.json").read_text())
        arrays = np.load(directory / "error_arrays.npz")
        possible = arrays["possible_errors"]
        impossible = arrays["impossible_errors"]
        spans = arrays["frame_spans"]
        violation = arrays["violation_frames"]
        categories = arrays["categories"]
        pair_ids = arrays["pair_ids"]
        recomputed = metric_view(possible, impossible, spans, violation)

        checks = {
            "model_name_matches": config["model"] == model == stored["model"],
            "manifest_hash_matches": config["frozen_pair_manifest_sha256"] == manifest_hash,
            "labels_unavailable_to_predictor": config["labels_available_to_predictor"] is False,
            "three_hundred_unique_pairs": len(pair_ids) == len(set(pair_ids.tolist())) == 300,
            "three_hundred_pair_json_files": len(list((directory / "pairs").glob("*.json"))) == 300,
            "error_rows_are_300": possible.shape[0] == impossible.shape[0] == 300,
            "errors_are_finite": bool(np.isfinite(possible).all() and np.isfinite(impossible).all()),
            "strict_checkpoint_load": bool(
                config["provenance"].get("strict_state", {}).get("strict_match", False)
                or all(
                    value.get("strict_match", False)
                    for value in config["provenance"].get("strict_states", {}).values()
                )
            ),
            "no_sp_label_selection_or_calibration": (
                config["provenance"].get("selection_or_calibration_on_sp_labels") is False
            ),
        }
        for key in SCALARS:
            checks[f"recomputed_{key}"] = same(stored["overall"][key], recomputed[key])
        checks["recomputed_joint_hit_at_2"] = same(
            stored["overall"]["joint_hit"]["2"], recomputed["joint_hit"]["2"]
        )
        checks["recomputed_t_star"] = stored["overall"]["t_star"] == recomputed["t_star"]
        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise AssertionError(f"{model} verification failed: {failed}")

        by_category = {}
        for category in sorted(set(categories.tolist())):
            selected = categories == category
            current = metric_view(
                possible[selected], impossible[selected], spans, violation[selected]
            )
            saved = stored["by_category"][category]
            category_checks = {
                key: same(saved[key], current[key]) for key in SCALARS
            }
            category_checks["joint_hit_at_2"] = same(
                saved["joint_hit"]["2"], current["joint_hit"]["2"]
            )
            if not all(category_checks.values()):
                raise AssertionError(f"{model}/{category} metric mismatch")
            by_category[category] = "passed"

        reports[model] = {
            "status": "passed",
            "checks": checks,
            "category_recomputation": by_category,
        }
        old_summary = json.loads(OLD[model].read_text())
        old_metrics = selected_metrics(old_summary)
        new_metrics = selected_metrics(stored)
        comparisons[model] = {
            "frozen_baseline": old_metrics,
            "corrected": new_metrics,
            "delta": {
                key: (
                    None
                    if old_metrics[key] is None or new_metrics[key] is None
                    else new_metrics[key] - old_metrics[key]
                )
                for key in old_metrics
                if isinstance(old_metrics[key], (int, float))
            },
        }

    write_json_new(
        report_path,
        {
            "status": "passed",
            "run_root": str(run_root),
            "manifest": str(PAIRS_CSV),
            "manifest_sha256": manifest_hash,
            "models": reports,
        },
    )
    write_json_new(
        comparison_path,
        {
            "interpretation": (
                "post-hoc protocol-correction sensitivity analysis; frozen baseline "
                "must remain reported"
            ),
            "models": comparisons,
        },
    )
    print(json.dumps({"verification": str(report_path), "status": "passed"}))


if __name__ == "__main__":
    main()
