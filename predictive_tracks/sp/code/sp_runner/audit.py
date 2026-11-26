"""Independent integrity and metric audit shared by snapshots and new suites."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from sp_protocol.run import metric_view

from .registry import ModelSpec, PAIR_MANIFEST, PAIR_MANIFEST_SHA256, ROOT


LEGACY_PAIR_MANIFEST = ROOT / "predictive_benchmark_v2/manifests/pairs_8to6.csv"


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def same(left, right) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    return left == right


def strict_load_is_complete(provenance: dict) -> bool:
    states = []
    if isinstance(provenance.get("strict_state"), dict):
        states.append(provenance["strict_state"])
    if isinstance(provenance.get("strict_states"), dict):
        states.extend(provenance["strict_states"].values())
    for key, value in provenance.items():
        if key.endswith("_strict_state") and isinstance(value, dict):
            states.append(value)
    return bool(states) and all(state.get("strict_match") is True for state in states)


def checkpoint_report(spec: ModelSpec) -> dict:
    report = {}
    for checkpoint in spec.checkpoints:
        if not checkpoint.path.is_file():
            raise FileNotFoundError(f"Missing {checkpoint.role}: {checkpoint.path}")
        observed = sha256(checkpoint.path)
        if observed != checkpoint.sha256:
            raise AssertionError(
                f"Checkpoint hash mismatch for {checkpoint.role}: {observed}"
            )
        report[checkpoint.role] = {
            "path": str(checkpoint.path),
            "bytes": checkpoint.path.stat().st_size,
            "sha256": observed,
            "verified": True,
        }
    return report


def metric_checks(stored: dict, recomputed: dict, prefix: str = "") -> dict:
    checks = {
        f"{prefix}recomputed_{key}": same(stored[key], recomputed[key])
        for key in SCALARS
    }
    checks[f"{prefix}recomputed_joint_hit_at_2"] = same(
        stored["joint_hit"]["2"], recomputed["joint_hit"]["2"]
    )
    checks[f"{prefix}recomputed_t_star"] = stored["t_star"] == recomputed["t_star"]
    return checks


def audit_run(
    spec: ModelSpec,
    run_dir: Path,
    *,
    expected_pairs_per_category: int = 100,
    hash_checkpoints: bool = True,
) -> dict:
    run_dir = run_dir.resolve()
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    stored = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    with np.load(run_dir / "error_arrays.npz", allow_pickle=False) as arrays:
        possible = arrays["possible_errors"]
        impossible = arrays["impossible_errors"]
        spans = arrays["frame_spans"]
        violation = arrays["violation_frames"]
        categories = arrays["categories"]
        pair_ids = arrays["pair_ids"]

    expected_pairs = 3 * expected_pairs_per_category
    recomputed = metric_view(possible, impossible, spans, violation)
    pair_files = sorted((run_dir / "pairs").glob("*.json"))
    pair_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in pair_files]
    provenance = config["provenance"]
    explicit_label_isolation = config.get("labels_available_to_predictor") is False
    legacy_candidate_isolation = provenance.get("candidate_use") in {
        "post-prediction scoring only",
        "post-prediction recurrent SAVi encoding only",
    }

    pair_arrays_match = False
    if len(pair_payloads) == expected_pairs:
        pair_possible = np.asarray([item["possible_errors"] for item in pair_payloads])
        pair_impossible = np.asarray([item["impossible_errors"] for item in pair_payloads])
        pair_arrays_match = bool(
            np.array_equal(pair_possible, possible)
            and np.array_equal(pair_impossible, impossible)
        )

    checks = {
        "model_name_matches": config["model"] == stored["model"] == spec.model_arg,
        "manifest_path_matches": Path(config["frozen_pair_manifest"])
        in {PAIR_MANIFEST, LEGACY_PAIR_MANIFEST},
        "manifest_hash_matches_frozen_value": (
            config["frozen_pair_manifest_sha256"] == PAIR_MANIFEST_SHA256
        ),
        "manifest_current_file_hash_matches": sha256(PAIR_MANIFEST) == PAIR_MANIFEST_SHA256,
        "strict_checkpoint_load_recorded": strict_load_is_complete(provenance),
        "predictor_label_isolation_recorded": (
            explicit_label_isolation or legacy_candidate_isolation
        ),
        "expected_error_shape": possible.shape
        == impossible.shape
        == (expected_pairs, spec.expected_steps),
        "expected_frame_spans": np.array_equal(
            spans, np.asarray(spec.expected_frame_spans, dtype=np.int64)
        ),
        "finite_nonnegative_errors": bool(
            np.isfinite(possible).all()
            and np.isfinite(impossible).all()
            and (possible >= 0).all()
            and (impossible >= 0).all()
        ),
        "expected_unique_pairs": (
            len(pair_ids) == len(set(pair_ids.tolist())) == expected_pairs
        ),
        "expected_pair_files": len(pair_payloads) == expected_pairs,
        "pair_files_match_saved_arrays": pair_arrays_match,
        "one_prediction_per_pair": all(
            item.get("prediction_calls_for_pair") == 1 for item in pair_payloads
        ),
        "expected_category_counts": all(
            int(np.sum(categories == category)) == expected_pairs_per_category
            for category in (
                "identity_change",
                "occluded_wrong_reappearance",
                "sudden_disappearance",
            )
        ),
    }
    checks.update(metric_checks(stored["overall"], recomputed))
    for category in sorted(set(categories.tolist())):
        selected = categories == category
        current = metric_view(
            possible[selected], impossible[selected], spans, violation[selected]
        )
        checks.update(
            metric_checks(
                stored["by_category"][category],
                current,
                prefix=f"{category}_",
            )
        )

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"Audit failed for {spec.key}: {failed}")

    code = {}
    for path in spec.code_files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing selected adapter source: {path}")
        code[str(path)] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    return {
        "status": "passed",
        "key": spec.key,
        "display_name": spec.display_name,
        "run_dir": str(run_dir),
        "model_arg": spec.model_arg,
        "reporting_status": spec.reporting_status,
        "checks": checks,
        "checkpoints": checkpoint_report(spec) if hash_checkpoints else "checked in suite preflight",
        "selected_adapter_code": code,
        "metrics": selected_metrics(stored),
    }


def selected_metrics(summary: dict) -> dict:
    overall = summary["overall"]
    return {
        "num_pairs": overall["num_pairs"],
        "paired_accuracy": overall["paired_accuracy"],
        "auc": overall["auc"],
        "detection_coverage": overall["detection_coverage"],
        "localized_and_correct_fraction": overall["localized_and_correct_fraction"],
        "localization_mae": overall["localization_mae"],
        "localization_mae_denominator": overall["localization_mae_denominator"],
        "joint_hit_at_2": overall["joint_hit"]["2"],
    }


def write_json_new(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
