#!/usr/bin/env python3
"""Run the VideoMAE-V2 official decoder-native state evaluation."""

from sp_protocol import run as base


ADAPTERS = {
    "videomaev2_decoder_native": (
        "sp_videomae.adapter"
    )
}
base.ADAPTERS = ADAPTERS
base.HEAVY_MODELS = set(ADAPTERS)


if __name__ == "__main__":
    base.main()
