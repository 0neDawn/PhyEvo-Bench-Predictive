#!/usr/bin/env python3
"""Run one frozen model on Interaction Evolution pairs."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from sp_benchmark.protocol_metrics import pairwise_results
from sp_core.common import read_pair_rgb, sha256, write_json_new
from sp_protocol.run import metric_view
from sp_runner.run_one import base as canonical_sp_runner

from .common import (
    DATA_ROOT,
    EXPECTED_CATEGORIES,
    PAIR_MANIFEST,
    load_pair_records,
)


ADAPTERS = {
    **canonical_sp_runner.ADAPTERS,
    "vjepa2_vith_official": "sp_vjepa2_vith.adapter",
    "vjepa2_vitl_official": "sp_vjepa2_vitl.adapter",
}
HEAVY_MODELS = {
    *canonical_sp_runner.HEAVY_MODELS,
    "vjepa2_vith_official",
    "vjepa2_vitl_official",
}


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(ADAPTERS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--memory-mode", choices=("safe", "all_gpu"), default="safe")
    parser.add_argument("--max-pairs", type=int, default=0, help="0 means all 300")
    parser.add_argument(
        "--max-pairs-per-category",
        type=int,
        default=0,
        help="0 means unlimited; 1 selects one pair from each IE category",
    )
    parser.add_argument("--categories", nargs="+", choices=EXPECTED_CATEGORIES)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--allow-heavy-cpu", action="store_true")
    args = parser.parse_args()

    if args.max_pairs < 0 or args.max_pairs_per_category < 0:
        raise ValueError("pair limits must be non-negative")
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"Refusing to reuse or overwrite run directory: {run_dir}")
    device = resolve_device(args.device)
    if (
        args.model in HEAVY_MODELS
        and not device.startswith("cuda")
        and not args.allow_heavy_cpu
    ):
        raise RuntimeError(
            f"{args.model} is blocked on CPU by default; use CUDA or explicitly "
            "pass --allow-heavy-cpu"
        )

    categories = args.categories or list(EXPECTED_CATEGORIES)
    records = load_pair_records(
        categories=categories,
        max_pairs=args.max_pairs if args.max_pairs > 0 else None,
        max_pairs_per_category=(
            args.max_pairs_per_category if args.max_pairs_per_category > 0 else None
        ),
    )
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(4)
    adapter_module = importlib.import_module(ADAPTERS[args.model])
    adapter = adapter_module.Adapter(device=device, memory_mode=args.memory_mode)

    run_dir.mkdir(parents=True)
    pair_dir = run_dir / "pairs"
    pair_dir.mkdir()
    write_json_new(
        run_dir / "run_config.json",
        {
            "track": "IE",
            "dataset_name": "Interaction Evolution",
            "source_directory_name": DATA_ROOT.name,
            "source_data_root": str(DATA_ROOT),
            "source_data_access": "read-only",
            "protocol": "IE fixed frames 12-19 -> 20-25, single prediction",
            "protocol_status": (
                "same frozen model adapters and outer metric implementation as SP"
            ),
            "model": args.model,
            "device": device,
            "memory_mode": args.memory_mode,
            "max_pairs": args.max_pairs,
            "max_pairs_per_category": args.max_pairs_per_category,
            "categories": categories,
            "frozen_pair_manifest": str(PAIR_MANIFEST),
            "frozen_pair_manifest_sha256": sha256(PAIR_MANIFEST),
            "epsilon": 1e-12,
            "localization_threshold": 0.0,
            "persistence_original_frames": 2,
            "joint_hit_tolerance": 2,
            "tie_policy": "margin == 0 is incorrect",
            "labels_available_to_predictor": False,
            "candidate_available_to_prediction": False,
            "selection_or_calibration_on_ie_labels": False,
            "adapter_selection_note": (
                "adapter frozen from the preceding SP protocol work before IE scoring"
            ),
            "sp_adapter_module": ADAPTERS[args.model],
            "provenance": adapter.provenance(),
        },
    )

    payloads = []
    canonical_spans = None
    for index, record in enumerate(records):
        possible_rgb, impossible_rgb = read_pair_rgb(record)
        calls_before = adapter.prediction_calls
        errors = adapter.score_pair(possible_rgb, impossible_rgb)
        if adapter.prediction_calls != calls_before + 1:
            raise RuntimeError(
                f"Adapter must emit exactly one prediction per pair: {record.pair_id}"
            )
        if canonical_spans is None:
            canonical_spans = errors.frame_spans
        elif not np.array_equal(canonical_spans, errors.frame_spans):
            raise RuntimeError("Frame spans changed between pairs")
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
        payloads.append(payload)
        safe_id = record.pair_id.replace("/", "__")
        write_json_new(pair_dir / f"{index:04d}__{safe_id}.json", payload)
        print(
            json.dumps(
                {"completed": index + 1, "total": len(records), "pair_id": record.pair_id},
                ensure_ascii=False,
            ),
            flush=True,
        )

    possible = np.asarray([item["possible_errors"] for item in payloads])
    impossible = np.asarray([item["impossible_errors"] for item in payloads])
    violation = np.asarray([item["t_ref"] for item in payloads], dtype=np.int64)
    pair_ids = np.asarray([item["pair_id"] for item in payloads])
    category_values = np.asarray([item["category"] for item in payloads])
    assert canonical_spans is not None

    overall = metric_view(possible, impossible, canonical_spans, violation)
    by_category = {}
    for category in categories:
        selected = category_values == category
        if selected.any():
            by_category[category] = metric_view(
                possible[selected], impossible[selected], canonical_spans, violation[selected]
            )
    write_json_new(
        run_dir / "summary.json",
        {
            "track": "IE",
            "model": args.model,
            "num_pairs": len(records),
            "overall": overall,
            "by_category": by_category,
        },
    )

    with (run_dir / "error_arrays.npz").open("xb") as handle:
        np.savez_compressed(
            handle,
            possible_errors=possible,
            impossible_errors=impossible,
            frame_spans=canonical_spans,
            violation_frames=violation,
            pair_ids=pair_ids,
            categories=category_values,
        )

    decisions = pairwise_results(possible, impossible)
    with (run_dir / "pair_details.csv").open("x", newline="", encoding="utf-8") as handle:
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
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, payload in enumerate(payloads):
            writer.writerow(
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
    print(
        json.dumps({"completed_run": str(run_dir), "overall": overall}, ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
