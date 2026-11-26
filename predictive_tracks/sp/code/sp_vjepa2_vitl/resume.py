#!/usr/bin/env python3
"""Resume an interrupted full ViT-L run without changing existing pair files."""

from sp_runner import resume_one
from sp_runner.registry import BY_KEY
from sp_runner.run_one import base as canonical_run

from .definition import VITL_SPEC


BY_KEY[VITL_SPEC.key] = VITL_SPEC
canonical_run.ADAPTERS[VITL_SPEC.model_arg] = "sp_vjepa2_vitl.adapter"
canonical_run.HEAVY_MODELS.add(VITL_SPEC.model_arg)


def main() -> None:
    resume_one.main()


if __name__ == "__main__":
    main()
