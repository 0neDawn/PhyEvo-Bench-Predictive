#!/usr/bin/env python3
"""Canonical per-model runner exposing only the five selected SP adapters."""

from sp_protocol import run as base


base.ADAPTERS = {
    "ddlp_obj3d": "sp_core.adapters.ddlp_obj3d",
    "slotformer_obj3d": "sp_core.adapters.slotformer_obj3d",
    "vjepa2_corrected": (
        "sp_protocol.adapters."
        "vjepa2_official_preprocess"
    ),
    "gswm_foreground_robust": (
        "sp_gswm.adapters."
        "gswm_foreground_predictive_mean_robust"
    ),
    "videomaev2_decoder_native": (
        "sp_videomae.adapter"
    ),
}
base.HEAVY_MODELS = {"vjepa2_corrected", "videomaev2_decoder_native"}


if __name__ == "__main__":
    base.main()

