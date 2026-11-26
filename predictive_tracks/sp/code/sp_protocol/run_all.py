#!/usr/bin/env python3
"""Run the three corrected SP adapters sequentially without overwriting output."""

from __future__ import annotations

from release_paths import runtime_python as _release_python, project_path as _release_path

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
JEPA_PYTHON = Path(_release_python('other'))
DDLP_PYTHON = Path(_release_python('ddlp'))
DEFAULT_MODELS = ("vjepa2_corrected", "videomaev2_corrected", "gswm_corrected")
ALL_MODELS = DEFAULT_MODELS + ("ddlp_obj3d", "slotformer_obj3d")


def python_for(model: str) -> Path:
    return DDLP_PYTHON if model == "ddlp_obj3d" else JEPA_PYTHON


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=list(DEFAULT_MODELS))
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--max-pairs-per-category", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    if run_root.exists():
        raise FileExistsError(f"Refusing to reuse or overwrite run root: {run_root}")

    commands = []
    for model in args.models:
        command = [
            str(python_for(model)), "-B", "-m",
            "sp_protocol.run",
            "--model", model,
            "--device", args.device,
            "--run-dir", str(run_root / model),
        ]
        if model == "vjepa2_corrected":
            # Verified on the local 16 GB 4060 Ti.  Avoids moving three giant
            # modules between host and device once per pair.
            command.extend(["--memory-mode", "all_gpu"])
        if args.max_pairs > 0:
            command.extend(["--max-pairs", str(args.max_pairs)])
        if args.max_pairs_per_category > 0:
            command.extend(
                ["--max-pairs-per-category", str(args.max_pairs_per_category)]
            )
        commands.append(command)

    print(
        json.dumps(
            {
                "run_root": str(run_root),
                "models": args.models,
                "data_changed": False,
                "commands": commands,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if args.dry_run:
        return

    run_root.mkdir(parents=True)
    child_environment = os.environ.copy()
    child_environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for command in commands:
        subprocess.run(command, cwd=ROOT, env=child_environment, check=True)


if __name__ == "__main__":
    main()
