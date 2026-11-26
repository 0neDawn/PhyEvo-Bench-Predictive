from __future__ import annotations

import unittest

import torch

from sp_protocol.adapters.videomaev2_official16 import (
    CONTEXT_TUBELETS,
    SCORED_FUTURE_TUBELETS,
    TOKENS_PER_TUBELET,
    TOTAL_TUBELETS,
)
from sp_videomae.adapter import Adapter


class DecoderNativeMaskTests(unittest.TestCase):
    def adapter_without_model(self) -> Adapter:
        adapter = object.__new__(Adapter)
        adapter.device = torch.device("cpu")
        return adapter

    def test_history_encoder_mask_is_prefix_only(self) -> None:
        adapter = self.adapter_without_model()
        mask = adapter._prefix_encoder_mask(CONTEXT_TUBELETS)
        stop = CONTEXT_TUBELETS * TOKENS_PER_TUBELET
        self.assertFalse(mask[:, :stop].any())
        self.assertTrue(mask[:, stop:].all())

    def test_only_real_future_positions_are_predicted(self) -> None:
        adapter = self.adapter_without_model()
        positions = adapter._future_decode_positions()
        start = CONTEXT_TUBELETS * TOKENS_PER_TUBELET
        stop = (CONTEXT_TUBELETS + SCORED_FUTURE_TUBELETS) * TOKENS_PER_TUBELET
        self.assertFalse(positions[:, :start].any())
        self.assertTrue(positions[:, start:stop].all())
        self.assertFalse(positions[:, stop:].any())
        self.assertEqual(
            int(positions.sum()), SCORED_FUTURE_TUBELETS * TOKENS_PER_TUBELET
        )
        self.assertEqual(tuple(positions.shape), (1, TOTAL_TUBELETS * TOKENS_PER_TUBELET))


if __name__ == "__main__":
    unittest.main()
