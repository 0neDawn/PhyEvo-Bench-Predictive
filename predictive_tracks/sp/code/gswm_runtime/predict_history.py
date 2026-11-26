#!/usr/bin/env python3
"""Direct command-line inference for G-SWM OBJ3D."""

from __future__ import annotations

import argparse
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
)


def indexed_pngs(directory: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in directory.glob("*.png"):
        try:
            index = int(path.stem)
        except ValueError:
            continue
        if index in result:
            raise ValueError(f"Duplicate numeric frame index {index} in {directory}")
        result[index] = path
    return result


def load_history(directory: Path, start_frame: int) -> np.ndarray:
    paths = indexed_pngs(directory)
    frames: list[np.ndarray] = []
    for index in range(start_frame, start_frame + HISTORY_STEPS):
        if index not in paths:
            raise FileNotFoundError(f"Frame {index} is missing from {directory}")
        with Image.open(paths[index]) as image:
            rgb = image.convert("RGB").resize(
                (NATIVE_SIZE, NATIVE_SIZE),
                resample=Image.Resampling.NEAREST,
            )
            frames.append(np.asarray(rgb, dtype=np.uint8).copy())
    return np.stack(frames, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one six-frame G-SWM forecast from eight shared history frames."
    )
    parser.add_argument("--history-dir", required=True, type=Path)
    parser.add_argument("--start-frame", type=int, default=12)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    history = load_history(args.history_dir.resolve(), args.start_frame)
    runtime = GSWMObj3D(device=args.device)
    prediction = runtime.predict(history, seed=args.seed)

    metadata = {
        "model": runtime.model_name,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_epoch": runtime.checkpoint_metadata["epoch"],
        "checkpoint_global_step": runtime.checkpoint_metadata["global_step"],
        "history_dir": str(args.history_dir.resolve()),
        "history_frames": list(range(args.start_frame, args.start_frame + HISTORY_STEPS)),
        "predicted_frames": list(
            range(
                args.start_frame + HISTORY_STEPS,
                args.start_frame + HISTORY_STEPS + FUTURE_STEPS,
            )
        ),
        "native_resolution": [NATIVE_SIZE, NATIVE_SIZE],
        "seed": prediction.seed,
        "foreground_future_sampling": False,
        "background_future_sampling": False,
        "candidate_future_was_input": False,
        "device": str(runtime.device),
    }

    with output.open("xb") as handle:
        np.savez_compressed(
            handle,
            predicted_rgb=np.transpose(prediction.rgb, (0, 2, 3, 1)),
            z_pres=prediction.z_pres,
            z_depth=prediction.z_depth,
            z_where=prediction.z_where,
            z_what=prediction.z_what,
            z_dyna=prediction.z_dyna,
            ids=prediction.ids,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    print(json.dumps({"output": str(output), **metadata}, ensure_ascii=False))


if __name__ == "__main__":
    main()
