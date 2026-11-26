#!/usr/bin/env python3
"""Resume one interrupted SP model run from its verified pair-file prefix.

Existing pair files are treated as immutable.  This module validates that they
form the exact contiguous prefix of the frozen manifest, evaluates only the
remaining pairs, and creates aggregate files exclusively after all 300 pair
files are available.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path

import numpy as np
import torch

from sp_benchmark.adapter_contract import CandidateErrors
from sp_benchmark.protocol_metrics import pairwise_results
from sp_core.common import (
    EXPECTED_CATEGORIES,
    PAIRS_CSV,
    load_pair_records,
    read_pair_rgb,
    sha256,
    write_json_new,
)
from sp_protocol.run import metric_view

from .registry import BY_KEY
from .run_one import base as canonical_run


def _expected_pair_name(index: int, pair_id: str) -> str:
    return f'{index:04d}__{pair_id.replace("/", "__")}.json'


def validated_prefix(spec, run_dir: Path, records) -> list[dict]:
    config_path = run_dir / "run_config.json"
    pair_dir = run_dir / "pairs"
    if not config_path.is_file() or not pair_dir.is_dir():
        raise RuntimeError(f"Interrupted run lacks run_config.json or pairs/: {run_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model") != spec.model_arg:
        raise RuntimeError(
            f"Model mismatch in {config_path}: {config.get('model')} != {spec.model_arg}"
        )
    if config.get("frozen_pair_manifest_sha256") != sha256(PAIRS_CSV):
        raise RuntimeError("Interrupted run uses a different pair manifest")
    if config.get("max_pairs", 0) != 0 or config.get("max_pairs_per_category", 0) != 0:
        raise RuntimeError("This resume entry only accepts a full 300-pair SP run")
    if config.get("categories") != list(EXPECTED_CATEGORIES):
        raise RuntimeError("Interrupted run has different category selection or order")

    paths = sorted(pair_dir.glob("*.json"))
    if len(paths) > len(records):
        raise RuntimeError("More pair files than frozen manifest records")
    payloads = []
    expected_spans = np.asarray(spec.expected_frame_spans, dtype=np.int64)
    for index, path in enumerate(paths):
        record = records[index]
        if path.name != _expected_pair_name(index, record.pair_id):
            raise RuntimeError(
                f"Pair files are not a contiguous frozen-manifest prefix at index {index}: "
                f"observed {path.name}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("index") != index
            or payload.get("pair_id") != record.pair_id
            or payload.get("category") != record.category
            or payload.get("t_ref") != record.t_ref
            or payload.get("prediction_calls_for_pair") != 1
        ):
            raise RuntimeError(f"Invalid existing pair payload: {path}")
        errors = CandidateErrors(
            payload["possible_errors"],
            payload["impossible_errors"],
            payload["frame_spans"],
        )
        if not np.array_equal(errors.frame_spans, expected_spans):
            raise RuntimeError(f"Unexpected frame spans in {path}")
        payloads.append(payload)
    return payloads


def _arrays(payloads: list[dict]):
    possible = np.asarray([item["possible_errors"] for item in payloads])
    impossible = np.asarray([item["impossible_errors"] for item in payloads])
    violation = np.asarray([item["t_ref"] for item in payloads], dtype=np.int64)
    pair_ids = np.asarray([item["pair_id"] for item in payloads])
    categories = np.asarray([item["category"] for item in payloads])
    spans = np.asarray(payloads[0]["frame_spans"], dtype=np.int64)
    return possible, impossible, spans, violation, pair_ids, categories


def _write_or_validate_npz(path: Path, arrays: tuple[np.ndarray, ...]) -> None:
    names = (
        "possible_errors",
        "impossible_errors",
        "frame_spans",
        "violation_frames",
        "pair_ids",
        "categories",
    )
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            if any(not np.array_equal(stored[name], value) for name, value in zip(names, arrays)):
                raise RuntimeError(f"Existing aggregate differs; refusing overwrite: {path}")
        return
    with path.open("xb") as handle:
        np.savez_compressed(handle, **dict(zip(names, arrays)))


def _write_or_validate_csv(
    path: Path, payloads: list[dict], possible: np.ndarray, impossible: np.ndarray, overall: dict
) -> None:
    decisions = pairwise_results(possible, impossible)
    fields = (
        "pair_id",
        "category",
        "t_ref",
        "mean_possible",
        "mean_impossible",
        "margin",
        "correct",
        "t_star",
        "delta_t",
    )
    rows = []
    for index, payload in enumerate(payloads):
        rows.append(
            {
                "pair_id": payload["pair_id"],
                "category": payload["category"],
                "t_ref": payload["t_ref"],
                "mean_possible": decisions.mean_possible[index],
                "mean_impossible": decisions.mean_impossible[index],
                "margin": decisions.margin[index],
                "correct": bool(decisions.correct[index]),
                "t_star": overall["t_star"][index],
                "delta_t": overall["delta_t"][index],
            }
        )
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            if len(list(csv.DictReader(handle))) != len(rows):
                raise RuntimeError(f"Incomplete existing aggregate; refusing overwrite: {path}")
        return
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finalize(spec, run_dir: Path, payloads: list[dict], resumed_from: int) -> None:
    if len(payloads) != 300:
        raise RuntimeError(f"Cannot finalize {spec.key}: observed {len(payloads)}/300 pairs")
    possible, impossible, spans, violation, pair_ids, categories = _arrays(payloads)
    overall = metric_view(possible, impossible, spans, violation)
    by_category = {}
    for category in EXPECTED_CATEGORIES:
        selected = categories == category
        by_category[category] = metric_view(
            possible[selected], impossible[selected], spans, violation[selected]
        )
    summary = {
        "model": spec.model_arg,
        "num_pairs": len(payloads),
        "overall": overall,
        "by_category": by_category,
    }
    _write_or_validate_npz(
        run_dir / "error_arrays.npz",
        (possible, impossible, spans, violation, pair_ids, categories),
    )
    _write_or_validate_csv(
        run_dir / "pair_details.csv", payloads, possible, impossible, overall
    )
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        if json.loads(summary_path.read_text(encoding="utf-8")) != summary:
            raise RuntimeError(f"Existing summary differs; refusing overwrite: {summary_path}")
    else:
        write_json_new(summary_path, summary)
    log_path = run_dir / "resume_record.json"
    if not log_path.exists():
        write_json_new(
            log_path,
            {
                "resume_mode": "verified contiguous pair prefix",
                "existing_pairs_reused": resumed_from,
                "new_pairs_evaluated": 300 - resumed_from,
                "existing_files_overwritten": False,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True, choices=tuple(BY_KEY))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    spec = BY_KEY[args.key]
    run_dir = args.run_dir.resolve()
    records = load_pair_records()
    payloads = validated_prefix(spec, run_dir, records)
    resumed_from = len(payloads)
    print(
        json.dumps(
            {
                "model": spec.key,
                "run_dir": str(run_dir),
                "verified_existing_pairs": resumed_from,
                "remaining_pairs": len(records) - resumed_from,
                "next_pair": (
                    records[resumed_from].pair_id if resumed_from < len(records) else None
                ),
                "existing_files_will_be_overwritten": False,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        return

    aggregate_names = ("summary.json", "error_arrays.npz", "pair_details.csv")
    if resumed_from < len(records) and any((run_dir / name).exists() for name in aggregate_names):
        raise RuntimeError("Partial pair prefix already has aggregate files; refusing ambiguous resume")

    if resumed_from < len(records):
        config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        torch.manual_seed(0)
        np.random.seed(0)
        torch.set_num_threads(4)
        module_name = canonical_run.ADAPTERS[spec.model_arg]
        adapter = importlib.import_module(module_name).Adapter(
            device=config["device"], memory_mode=config.get("memory_mode", "safe")
        )
        expected_spans = np.asarray(spec.expected_frame_spans, dtype=np.int64)
        for index in range(resumed_from, len(records)):
            record = records[index]
            possible_rgb, impossible_rgb = read_pair_rgb(record)
            calls_before = adapter.prediction_calls
            errors = adapter.score_pair(possible_rgb, impossible_rgb)
            if adapter.prediction_calls != calls_before + 1:
                raise RuntimeError(f"Expected one prediction for {record.pair_id}")
            if not np.array_equal(errors.frame_spans, expected_spans):
                raise RuntimeError(f"Unexpected frame spans for {record.pair_id}")
            payload = {
                "index": index,
                "pair_id": record.pair_id,
                "category": record.category,
                "t_ref": record.t_ref,
                "possible_errors": errors.possible.tolist(),
                "impossible_errors": errors.impossible.tolist(),
                "frame_spans": errors.frame_spans.tolist(),
                "prediction_calls_for_pair": 1,
            }
            write_json_new(
                run_dir / "pairs" / _expected_pair_name(index, record.pair_id), payload
            )
            payloads.append(payload)
            print(
                json.dumps(
                    {
                        "completed": index + 1,
                        "total": len(records),
                        "pair_id": record.pair_id,
                        "resumed": True,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finalize(spec, run_dir, payloads, resumed_from)


if __name__ == "__main__":
    main()
