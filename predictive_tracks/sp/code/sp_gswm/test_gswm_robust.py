from __future__ import annotations

import unittest

import numpy as np

from sp_gswm.adapters.gswm_foreground_predictive_mean import (
    ForegroundPrediction,
)
from sp_gswm.adapters.gswm_foreground_predictive_mean_robust import (
    Adapter,
)


class GSWMRobustTests(unittest.TestCase):
    def test_empty_alpha_falls_back_to_full_frame_mse(self) -> None:
        prediction = ForegroundPrediction(
            rgb=np.zeros((6, 3, 64, 64), dtype=np.float32),
            alpha=np.zeros((6, 1, 64, 64), dtype=np.float32),
        )
        candidate_tchw = np.ones((6, 3, 64, 64), dtype=np.float32)
        result = Adapter._foreground_mse(
            prediction, candidate_tchw.transpose(0, 2, 3, 1)
        )
        self.assertTrue(np.allclose(result, 1.0))

    def test_nonempty_alpha_keeps_foreground_weighting(self) -> None:
        alpha = np.zeros((6, 1, 64, 64), dtype=np.float32)
        alpha[:, :, :32] = 1.0
        prediction = ForegroundPrediction(
            rgb=np.zeros((6, 3, 64, 64), dtype=np.float32), alpha=alpha
        )
        candidate_tchw = np.zeros((6, 3, 64, 64), dtype=np.float32)
        candidate_tchw[:, :, :32] = 1.0
        result = Adapter._foreground_mse(
            prediction, candidate_tchw.transpose(0, 2, 3, 1)
        )
        self.assertTrue(np.allclose(result, 1.0))


if __name__ == "__main__":
    unittest.main()
