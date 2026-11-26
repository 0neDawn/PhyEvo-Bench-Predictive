"""G-SWM foreground score with a defined empty-alpha fallback.

The original exploratory foreground formula is undefined when G-SWM predicts
no foreground pixels.  For only those frames, this additive variant falls back
to the already established full-frame RGB MSE.  The branch depends solely on
the model prediction and never on either candidate or its label.
"""

from __future__ import annotations

import numpy as np

from gswm_runtime.runtime import FUTURE_STEPS, _as_tchw
from sp_gswm.adapters.gswm_foreground_predictive_mean import (
    Adapter as ForegroundAdapter,
    ForegroundPrediction,
)


class Adapter(ForegroundAdapter):
    model_name = "gswm_obj3d_foreground_predictive_mean_robust"

    @staticmethod
    def _foreground_mse(
        prediction: ForegroundPrediction, candidate: np.ndarray
    ) -> np.ndarray:
        candidate_tchw = _as_tchw(candidate, FUTURE_STEPS).numpy()
        pixel_error = np.mean((prediction.rgb - candidate_tchw) ** 2, axis=1)
        weights = prediction.alpha[:, 0]
        denominator = weights.sum(axis=(1, 2))

        error = np.empty(FUTURE_STEPS, dtype=np.float64)
        has_foreground = denominator > 1e-12
        error[has_foreground] = (
            (pixel_error[has_foreground] * weights[has_foreground]).sum(axis=(1, 2))
            / denominator[has_foreground]
        )
        error[~has_foreground] = pixel_error[~has_foreground].mean(axis=(1, 2))
        return error

    def provenance(self) -> dict:
        value = super().provenance()
        value["evaluation_variant"] = (
            "exploratory model-predicted foreground distance with empty-alpha fallback"
        )
        value["distance"] = (
            "candidate-independent predicted-alpha-weighted RGB MSE; when and only "
            "when predicted alpha mass is zero, full-frame RGB MSE"
        )
        value["empty_alpha_fallback"] = "full-frame RGB MSE"
        value["fallback_condition_uses_candidate_or_label"] = False
        return value
