"""Read-only IE data selection from the shared frozen benchmark manifest."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Iterable

from sp_core.common import PairRecord
from sp_runner.registry import (
    PAIR_MANIFEST as SP_PAIR_MANIFEST,
    PAIR_MANIFEST_SHA256 as SP_PAIR_MANIFEST_SHA256,
)


IE_CODE_ROOT = Path(__file__).resolve().parents[1]
IE_ROOT = IE_CODE_ROOT.parent
PROJECT_ROOT = IE_ROOT.parents[1]
SP_CODE_ROOT = PROJECT_ROOT / "predictive_tracks/sp/code"
PAIR_MANIFEST = SP_PAIR_MANIFEST
PAIR_MANIFEST_SHA256 = SP_PAIR_MANIFEST_SHA256
DATA_ROOT = PROJECT_ROOT / "data/process_obj3d"
EXPECTED_CATEGORIES = ("contact_response", "no_contact", "penetration")


def manifest_protocol_report() -> dict:
    """Verify every frozen IE row against the paper's temporal protocol."""

    with PAIR_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("track") == "IE"]
    checks = {
        "exactly_300_ie_pairs": len(rows) == 300,
        "unique_ie_pair_ids": len({row["pair_id"] for row in rows}) == 300,
        "three_balanced_categories": Counter(
            row["category_directory"] for row in rows
        )
        == Counter({category: 100 for category in EXPECTED_CATEGORIES}),
        "history_is_12_to_19": all(
            (row["history_start"], row["history_end"]) == ("12", "19")
            for row in rows
        ),
        "future_is_20_to_25": all(
            (row["future_start"], row["future_end"]) == ("20", "25")
            for row in rows
        ),
        "event_and_reference_are_20": all(
            row["event_frame"] == row["t_ref"] == "20" for row in rows
        ),
        "first_decoded_difference_is_20": all(
            row["first_decoded_pixel_difference"] == "20" for row in rows
        ),
        "reference_source_is_explicit": all(
            row["t_ref_source"] == "derived_first_decoded_pixel_difference"
            for row in rows
        ),
        "fixed_window_exists": all(
            int(row["possible_frame_count"]) > 25
            and int(row["impossible_frame_count"]) > 25
            for row in rows
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"Frozen IE manifest violates protocol: {failed}")
    return {
        "status": "passed",
        "checks": checks,
        "t_ref_source": "derived_first_decoded_pixel_difference",
        "metadata_corroboration": "event_frame=20 for all 300 IE pairs",
        "source_metadata_modified": False,
    }


def load_pair_records(
    categories: Iterable[str] | None = None,
    max_pairs: int | None = None,
    max_pairs_per_category: int | None = None,
) -> list[PairRecord]:
    """Select IE only; no source-data file is created or changed."""

    selected = set(categories or EXPECTED_CATEGORIES)
    unknown = selected - set(EXPECTED_CATEGORIES)
    if unknown:
        raise ValueError(f"Unknown IE categories: {sorted(unknown)}")
    records: list[PairRecord] = []
    counts = {category: 0 for category in selected}
    with PAIR_MANIFEST.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("track") != "IE":
                continue
            category = row["category_directory"]
            if category not in selected:
                continue
            if (
                max_pairs_per_category is not None
                and max_pairs_per_category > 0
                and counts[category] >= max_pairs_per_category
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
            for path in (record.possible_path, record.impossible_path, record.meta_path):
                if not path.resolve().is_relative_to(DATA_ROOT.resolve()):
                    raise ValueError(f"IE manifest path escapes its data root: {path}")
            records.append(record)
            counts[category] += 1
            if max_pairs is not None and max_pairs > 0 and len(records) >= max_pairs:
                break
    if not records:
        raise ValueError("No IE pairs selected")
    return records
