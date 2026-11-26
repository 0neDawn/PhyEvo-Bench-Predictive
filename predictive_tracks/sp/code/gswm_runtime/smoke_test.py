#!/usr/bin/env python3
"""Strict load, deterministic forecast, and paired-scoring smoke test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from runtime import (
    CHECKPOINT_SHA256,
    FUTURE_STEPS,
    GSWMObj3D,
    HISTORY_STEPS,
    NATIVE_SIZE,
    OFFICIAL_REPO,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[4]
PAIR = ROOT / "data/anomaly_3d_state2/identity_change/pair_000000"


def load_frames(directory: Path, first: int, count: int) -> np.ndarray:
    lookup = {int(path.stem): path for path in directory.glob("*.png")}
    frames: list[np.ndarray] = []
    for index in range(first, first + count):
        with Image.open(lookup[index]) as image:
            image = image.convert("RGB").resize(
                (NATIVE_SIZE, NATIVE_SIZE),
                resample=Image.Resampling.NEAREST,
            )
            frames.append(np.asarray(image, dtype=np.uint8).copy())
    return np.stack(frames)


def main() -> None:
    possible_dir = PAIR / "possible"
    impossible_dir = PAIR / "impossible"
    history = load_frames(possible_dir, 12, HISTORY_STEPS)
    possible = load_frames(possible_dir, 20, FUTURE_STEPS)
    impossible = load_frames(impossible_dir, 20, FUTURE_STEPS)

    runtime = GSWMObj3D(device="auto")
    first = runtime.predict(history)
    second = runtime.predict(history)
    deterministic = all(
        np.array_equal(getattr(first, name), getattr(second, name))
        for name in ("rgb", "z_pres", "z_depth", "z_where", "z_what", "z_dyna", "ids")
    )
    if not deterministic:
        raise AssertionError("Repeated prediction from the same history was not deterministic")

    possible_error = runtime.pixel_mse(first, possible)
    impossible_error = runtime.pixel_mse(first, impossible)
    if possible_error.shape != (FUTURE_STEPS,) or impossible_error.shape != (FUTURE_STEPS,):
        raise AssertionError("Paired error arrays do not have six steps")

    source_head = (
        __import__("subprocess")
        .check_output(["git", "-C", str(OFFICIAL_REPO), "rev-parse", "HEAD"], text=True)
        .strip()
    )
    result = {
        "status": "pass",
        "model": runtime.model_name,
        "official_source_commit": source_head,
        "checkpoint_sha256": sha256_file(OFFICIAL_REPO / "pretrained/obj3d.pth"),
        "checkpoint_hash_matches_frozen_value": sha256_file(
            OFFICIAL_REPO / "pretrained/obj3d.pth"
        )
        == CHECKPOINT_SHA256,
        "strict_state_load": True,
        "checkpoint_metadata": runtime.checkpoint_metadata,
        "protocol": "8 history -> one candidate-independent prediction -> 6 future steps",
        "prediction_shape_tchw": list(first.rgb.shape),
        "repeated_prediction_bitwise_equal": deterministic,
        "paired_possible_error_shape": list(possible_error.shape),
        "paired_impossible_error_shape": list(impossible_error.shape),
        "paired_scoring_used_same_prediction": True,
        "candidate_future_was_input_to_predictor": False,
        "future_prior_sampling": False,
        "history_posterior_randomness_control": "SHA-256-derived per-history seed",
        "device_tested": str(runtime.device),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
