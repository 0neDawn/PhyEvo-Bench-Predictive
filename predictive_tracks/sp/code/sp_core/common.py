"""Shared data and checkpoint utilities for the additive SP benchmark."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


CODE_ROOT = Path(__file__).resolve().parents[1]
SP_ROOT = CODE_ROOT.parent
ROOT = SP_ROOT.parents[1]
PAIRS_CSV = CODE_ROOT / "sp_benchmark/manifests/pairs_8to6.csv"
HISTORY_START = 12
HISTORY_END = 19
FUTURE_START = 20
FUTURE_END = 25
EXPECTED_CATEGORIES = (
    "identity_change",
    "occluded_wrong_reappearance",
    "sudden_disappearance",
)


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    category: str
    t_ref: int
    possible_path: Path
    impossible_path: Path
    meta_path: Path


def load_pair_records(
    categories: Iterable[str] | None = None,
    max_pairs: int | None = None,
    max_pairs_per_category: int | None = None,
) -> list[PairRecord]:
    selected = set(categories or EXPECTED_CATEGORIES)
    unknown = selected - set(EXPECTED_CATEGORIES)
    if unknown:
        raise ValueError(f"Unknown SP categories: {sorted(unknown)}")
    records: list[PairRecord] = []
    category_counts = {category: 0 for category in selected}
    with PAIRS_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("track") != "SP":
                continue
            category = row["category_directory"]
            if category not in selected:
                continue
            if (
                max_pairs_per_category is not None
                and max_pairs_per_category > 0
                and category_counts[category] >= max_pairs_per_category
            ):
                continue
            record = PairRecord(
                pair_id=row["pair_id"],
                category=category,
                t_ref=int(row["t_ref"]),
                possible_path=Path(row["possible_dir"]),
                impossible_path=Path(row["impossible_dir"]),
                meta_path=Path(row["meta_path"]),
            )
            records.append(record)
            category_counts[category] += 1
            if max_pairs is not None and max_pairs > 0 and len(records) >= max_pairs:
                break
    if not records:
        raise ValueError("No pairs selected")
    return records


def _indexed_pngs(directory: Path) -> dict[int, Path]:
    indexed = {int(path.stem): path for path in directory.glob("*.png")}
    needed = set(range(HISTORY_START, FUTURE_END + 1))
    missing = sorted(needed - indexed.keys())
    if missing:
        raise ValueError(f"Missing frames {missing} in {directory}")
    return indexed


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).copy()


def read_pair_rgb(record: PairRecord) -> tuple[np.ndarray, np.ndarray]:
    videos = []
    for directory in (record.possible_path, record.impossible_path):
        indexed = _indexed_pngs(directory)
        frames = [_read_rgb(indexed[index]) for index in range(HISTORY_START, FUTURE_END + 1)]
        videos.append(np.stack(frames))
    possible, impossible = videos
    if not np.array_equal(possible[:8], impossible[:8]):
        raise AssertionError(f"Non-shared history: {record.pair_id}")
    if np.array_equal(possible[8:], impossible[8:]):
        raise AssertionError(f"Candidate futures do not differ: {record.pair_id}")
    return possible, impossible


def video_tensor(
    frames: np.ndarray,
    resolution: int,
    *,
    imagenet_normalize: bool = False,
    minus_one_to_one: bool = False,
) -> torch.Tensor:
    """Return [1,T,C,H,W] without exposing candidate data to a predictor."""
    value = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float() / 255.0
    value = F.interpolate(
        value, size=(resolution, resolution), mode="bilinear", align_corners=False
    )
    if imagenet_normalize:
        mean = value.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = value.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        value = (value - mean) / std
    if minus_one_to_one:
        value = (value - 0.5) / 0.5
    return value.unsqueeze(0)


def video_tensor_bcthw(
    frames: np.ndarray, resolution: int, *, imagenet_normalize: bool = True
) -> torch.Tensor:
    value = video_tensor(
        frames, resolution, imagenet_normalize=imagenet_normalize
    )
    return value.permute(0, 2, 1, 3, 4).contiguous()


def clean_state(state: dict, *, backbone: bool = False) -> dict:
    result = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        if backbone and key.startswith("backbone."):
            key = key[len("backbone.") :]
        if key in result:
            raise ValueError(f"Duplicate checkpoint key after cleanup: {key}")
        result[key] = value
    return result


def strict_state_report(model: torch.nn.Module, state: dict) -> dict:
    current = model.state_dict()
    common = current.keys() & state.keys()
    mismatch = {
        key: {"model": list(current[key].shape), "checkpoint": list(state[key].shape)}
        for key in common
        if current[key].shape != state[key].shape
    }
    missing = sorted(current.keys() - state.keys())
    unexpected = sorted(state.keys() - current.keys())
    return {
        "model_keys": len(current),
        "checkpoint_keys": len(state),
        "matched_keys": len(common - mismatch.keys()),
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatch": mismatch,
        "strict_match": not missing and not unexpected and not mismatch,
    }


def load_strict(model: torch.nn.Module, state: dict, *, assign: bool = False) -> dict:
    report = strict_state_report(model, state)
    if not report["strict_match"]:
        raise RuntimeError(json.dumps(report))
    loaded = model.load_state_dict(state, strict=True, assign=assign)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(f"Unexpected strict-load result: {loaded}")
    if any(value.is_meta for value in model.state_dict().values()):
        raise RuntimeError("Strict load left meta tensors in model")
    return report


@contextlib.contextmanager
def empty_construction():
    """Construct billion-parameter models on meta without allocating random weights."""
    original = torch.linspace

    def cpu_linspace(*args, **kwargs):
        kwargs.setdefault("device", "cpu")
        return original(*args, **kwargs)

    torch.linspace = cpu_linspace
    try:
        with torch.device("meta"):
            yield
    finally:
        torch.linspace = original


def sha256(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
