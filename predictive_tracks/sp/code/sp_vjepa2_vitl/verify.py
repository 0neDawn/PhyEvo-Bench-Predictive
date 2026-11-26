#!/usr/bin/env python3
"""Independently recompute and audit a completed ViT-L SP run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sp_runner.audit import audit_run, write_json_new

from .definition import VITL_SPEC


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pairs-per-category", type=int, default=100)
    args = parser.parse_args()
    if args.pairs_per_category <= 0:
        raise ValueError("--pairs-per-category must be positive")
    run_dir = args.run_dir.resolve()
    report = audit_run(
        VITL_SPEC,
        run_dir,
        expected_pairs_per_category=args.pairs_per_category,
        hash_checkpoints=True,
    )
    output = run_dir / "verification.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    write_json_new(output, report)
    print(json.dumps({"status": "passed", "report": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
