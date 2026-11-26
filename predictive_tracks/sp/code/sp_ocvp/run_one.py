#!/usr/bin/env python3
"""Run either additive OCVP model with the frozen SP protocol."""

from sp_protocol import run as base


base.ADAPTERS = {
    "ocvp_seq_obj3d": "sp_ocvp.ocvp_seq",
    "savi_lstm_obj3d": "sp_ocvp.savi_lstm",
}
base.HEAVY_MODELS = set()


if __name__ == "__main__":
    base.main()

