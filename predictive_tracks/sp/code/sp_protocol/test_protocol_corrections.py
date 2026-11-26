from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

from sp_protocol.adapters.gswm_predictive_mean import (
    Adapter as GSWMAdapter,
    MC_SAMPLES,
)
from sp_protocol.adapters.videomaev2_official16 import (
    Adapter as VideoMAEAdapter,
)
from sp_protocol.adapters.vjepa2_official_preprocess import (
    official_video_tensor,
)


class _Prediction:
    def __init__(self, value: float) -> None:
        self.rgb = np.full((6, 3, 64, 64), value, dtype=np.float32)


class ProtocolCorrectionTests(unittest.TestCase):
    def test_vjepa2_official_preprocess_shape_and_finiteness(self) -> None:
        frames = np.zeros((14, 128, 128, 3), dtype=np.uint8)
        value = official_video_tensor(frames)
        self.assertEqual(tuple(value.shape), (1, 3, 14, 256, 256))
        self.assertTrue(torch.isfinite(value).all())

    def test_gswm_predictive_mean_is_fixed_arithmetic_mean(self) -> None:
        adapter = object.__new__(GSWMAdapter)
        runtime = mock.Mock()
        runtime.predict.side_effect = [_Prediction(float(i)) for i in range(MC_SAMPLES)]
        adapter.runtime = runtime
        history = np.zeros((8, 128, 128, 3), dtype=np.uint8)
        prediction = adapter._predictive_mean(history)
        self.assertEqual(runtime.predict.call_count, MC_SAMPLES)
        self.assertTrue(np.allclose(prediction.rgb, (MC_SAMPLES - 1) / 2))

    def test_videomae_padding_is_masked_and_keeps_14_input_frames(self) -> None:
        adapter = object.__new__(VideoMAEAdapter)
        adapter.device = torch.device("cpu")
        frames = np.zeros((14, 128, 128, 3), dtype=np.uint8)
        value = adapter._pad_to_official_16(frames)
        self.assertEqual(tuple(value.shape), (1, 3, 16, 224, 224))
        self.assertTrue(torch.equal(value[:, :, 14:], torch.zeros_like(value[:, :, 14:])))

    def test_runner_keeps_ddlp_and_slotformer_on_frozen_modules(self) -> None:
        from sp_protocol.run import ADAPTERS

        self.assertEqual(
            ADAPTERS["ddlp_obj3d"],
            "sp_core.adapters.ddlp_obj3d",
        )
        self.assertEqual(
            ADAPTERS["slotformer_obj3d"],
            "sp_core.adapters.slotformer_obj3d",
        )


if __name__ == "__main__":
    unittest.main()
