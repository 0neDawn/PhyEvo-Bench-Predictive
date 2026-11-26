#!/usr/bin/env python3
"""Run the additive G-SWM foreground variant with empty-alpha fallback."""

from sp_protocol import run as base


ADAPTERS = {
    "gswm_foreground_robust": (
        "sp_gswm.adapters."
        "gswm_foreground_predictive_mean_robust"
    )
}
base.ADAPTERS = ADAPTERS
base.HEAVY_MODELS = set()


if __name__ == "__main__":
    base.main()
