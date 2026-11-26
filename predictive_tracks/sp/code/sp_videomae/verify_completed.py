#!/usr/bin/env python3
"""Independently audit a completed VideoMAE-V2 decoder-native run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from sp_core.common import PAIRS_CSV, sha256, write_json_new
from sp_protocol.run import metric_view


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    report = directory / "verification.json"
    if report.exists():
        raise FileExistsError(f"Refusing to overwrite {report}")

    config = json.loads((directory / "run_config.json").read_text())
    stored = json.loads((directory / "summary.json").read_text())
    with np.load(directory / "error_arrays.npz") as arrays:
        possible = arrays["possible_errors"]
        impossible = arrays["impossible_errors"]
        spans = arrays["frame_spans"]
        violation = arrays["violation_frames"]
        categories = arrays["categories"]
        pair_ids = arrays["pair_ids"]
    recomputed = metric_view(possible, impossible, spans, violation)
    pair_payloads = [
        json.loads(path.read_text())
        for path in sorted((directory / "pairs").glob("*.json"))
    ]
    provenance = config["provenance"]
    checkpoint = Path(provenance["checkpoint"])
    checks = {
        "model_name_matches": (
            config["model"] == stored["model"] == "videomaev2_decoder_native"
        ),
        "manifest_hash_matches": (
            config["frozen_pair_manifest_sha256"] == sha256(PAIRS_CSV)
        ),
        "checkpoint_hash_matches": (
            sha256(checkpoint) == provenance["checkpoint_sha256"]
        ),
        "strict_checkpoint_load": provenance["strict_state"]["strict_match"] is True,
        "labels_unavailable_to_predictor": config["labels_available_to_predictor"] is False,
        "candidate_unavailable_to_prediction": (
            provenance["candidate_available_to_prediction"] is False
        ),
        "no_additional_prediction_head": (
            provenance["additional_prediction_head"] is False
        ),
        "no_sp_label_selection_or_calibration": (
            provenance["selection_or_calibration_on_sp_labels"] is False
        ),
        "post_hoc_status_disclosed": (
            provenance["status"] == "post-hoc native-space sensitivity analysis"
        ),
        "three_hundred_unique_pairs": len(pair_ids) == len(set(pair_ids.tolist())) == 300,
        "three_hundred_pair_files": len(pair_payloads) == 300,
        "one_prediction_per_pair": all(
            item["prediction_calls_for_pair"] == 1 for item in pair_payloads
        ),
        "one_hundred_per_category": all(
            int(np.sum(categories == category)) == 100
            for category in (
                "identity_change",
                "occluded_wrong_reappearance",
                "sudden_disappearance",
            )
        ),
        "three_decoder_steps": possible.shape == impossible.shape == (300, 3),
        "finite_errors": bool(np.isfinite(possible).all() and np.isfinite(impossible).all()),
    }
    for key in SCALARS:
        checks[f"recomputed_{key}"] = same(stored["overall"][key], recomputed[key])
    checks["recomputed_joint_hit_at_2"] = same(
        stored["overall"]["joint_hit"]["2"], recomputed["joint_hit"]["2"]
    )
    checks["recomputed_t_star"] = stored["overall"]["t_star"] == recomputed["t_star"]
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"Verification failed: {failed}")
    write_json_new(
        report,
        {
            "status": "passed",
            "run_dir": str(directory),
            "checks": checks,
        },
    )
    print(json.dumps({"verification": str(report), "status": "passed"}))


if __name__ == "__main__":
    main()
