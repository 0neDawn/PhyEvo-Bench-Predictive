#!/usr/bin/env python3
"""Dedicated SP runner for the additive V-JEPA 2 ViT-L evaluation."""

from sp_protocol import run as base


base.ADAPTERS = {
    "vjepa2_vitl_official": "sp_vjepa2_vitl.adapter",
}
base.HEAVY_MODELS = {"vjepa2_vitl_official"}


def main() -> None:
    base.main()


if __name__ == "__main__":
    main()
