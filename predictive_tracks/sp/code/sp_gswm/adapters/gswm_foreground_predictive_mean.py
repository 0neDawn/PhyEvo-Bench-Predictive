"""Object-focused G-SWM distance using its predicted foreground alpha.

The foreground alpha is generated from history before candidates are scored.
It is therefore neither a ground-truth object mask nor an anomaly label.
"""

from __future__ import annotations

from dataclasses import dataclass
from release_paths import verified_source_revision

import numpy as np
import torch

from gswm_runtime.runtime import (
    CHECKPOINT_SHA256,
    FUTURE_STEPS,
    GSWMObj3D,
    HISTORY_STEPS,
    NATIVE_SIZE,
    OBJ3D_CHECKPOINT,
    OBJ3D_CONFIG,
    OFFICIAL_REPO,
    _as_tchw,
    _configure_reproducibility,
    seed_from_history,
)
from sp_benchmark.adapter_contract import CandidateErrors
from sp_core.adapters.gswm_obj3d import (
    OFFICIAL_CHECKPOINT_FILE_ID,
    OFFICIAL_SOURCE_REVISION,
)
from sp_protocol.adapters.gswm_predictive_mean import (
    MC_SAMPLES,
    SEED_STRIDE,
)


@dataclass(frozen=True)
class ForegroundPrediction:
    rgb: np.ndarray
    alpha: np.ndarray


class Adapter:
    model_name = "gswm_obj3d_foreground_predictive_mean"
    frame_spans = np.asarray([[frame, frame] for frame in range(20, 26)])

    def __init__(self, device: str, memory_mode: str = "safe") -> None:
        del memory_mode
        self.runtime = GSWMObj3D(device=device)
        self.prediction_calls = 0
        source_revision = verified_source_revision("third_party/G-SWM")
        if source_revision != OFFICIAL_SOURCE_REVISION:
            raise RuntimeError("Unexpected G-SWM source revision")
        self.source_revision = source_revision

    def _one_rollout(self, history: np.ndarray, seed: int) -> ForegroundPrediction:
        history_tensor = _as_tchw(history, HISTORY_STEPS)
        _configure_reproducibility(seed)
        sequence = torch.zeros(
            1,
            HISTORY_STEPS + FUTURE_STEPS,
            3,
            NATIVE_SIZE,
            NATIVE_SIZE,
            dtype=torch.float32,
            device=self.runtime.device,
        )
        sequence[:, :HISTORY_STEPS] = history_tensor.unsqueeze(0).to(
            self.runtime.device
        )
        with torch.inference_mode():
            log = self.runtime.model.generate(
                sequence,
                cond_steps=HISTORY_STEPS,
                fg_sample=False,
                bg_sample=False,
            )
        rgb = log["recon"][0, HISTORY_STEPS:].detach().cpu().numpy().copy()
        alpha = log["alpha_map"][0, HISTORY_STEPS:].detach().cpu().numpy().copy()
        return ForegroundPrediction(rgb=rgb, alpha=alpha)

    def _predict(self, history: np.ndarray) -> ForegroundPrediction:
        base_seed = seed_from_history(history)
        rollouts = [
            self._one_rollout(
                history, (base_seed + index * SEED_STRIDE) % (2**31)
            )
            for index in range(MC_SAMPLES)
        ]
        rgb = np.mean(np.stack([item.rgb for item in rollouts]), axis=0)
        alpha = np.mean(np.stack([item.alpha for item in rollouts]), axis=0)
        alpha = np.clip(alpha, 0.0, 1.0)
        if rgb.shape != (6, 3, 64, 64) or alpha.shape != (6, 1, 64, 64):
            raise RuntimeError(f"Unexpected G-SWM output: rgb={rgb.shape}, alpha={alpha.shape}")
        if not np.isfinite(rgb).all() or not np.isfinite(alpha).all():
            raise RuntimeError("Non-finite G-SWM prediction")
        return ForegroundPrediction(rgb=rgb, alpha=alpha)

    @staticmethod
    def _foreground_mse(
        prediction: ForegroundPrediction, candidate: np.ndarray
    ) -> np.ndarray:
        candidate_tchw = _as_tchw(candidate, FUTURE_STEPS).numpy()
        pixel_error = np.mean((prediction.rgb - candidate_tchw) ** 2, axis=1)
        weights = prediction.alpha[:, 0]
        denominator = weights.sum(axis=(1, 2))
        if np.any(denominator <= 1e-12):
            raise RuntimeError("G-SWM predicted an empty foreground alpha map")
        error = (pixel_error * weights).sum(axis=(1, 2)) / denominator
        return error.astype(np.float64)

    def score_pair(
        self, possible_rgb: np.ndarray, impossible_rgb: np.ndarray
    ) -> CandidateErrors:
        prediction = self._predict(possible_rgb[:8])
        self.prediction_calls += 1
        possible_error = self._foreground_mse(prediction, possible_rgb[8:])
        impossible_error = self._foreground_mse(prediction, impossible_rgb[8:])
        return CandidateErrors(possible_error, impossible_error, self.frame_spans)

    def provenance(self) -> dict:
        state_count = int(self.runtime.checkpoint_metadata["state_tensor_count"])
        return {
            "model": "G-SWM official OBJ3D",
            "evaluation_variant": "exploratory model-predicted foreground distance",
            "official_repository": "https://github.com/zhixuan-lin/G-SWM",
            "official_source_revision": self.source_revision,
            "config": str(OBJ3D_CONFIG),
            "checkpoint": str(OBJ3D_CHECKPOINT),
            "checkpoint_official_google_drive_file_id": OFFICIAL_CHECKPOINT_FILE_ID,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "strict_state": {
                "model_keys": len(self.runtime.model.state_dict()),
                "checkpoint_keys": state_count,
                "matched_keys": state_count,
                "missing": [],
                "unexpected": [],
                "shape_mismatch": {},
                "strict_match": True,
            },
            "posterior_predictive_samples": MC_SAMPLES,
            "prediction_aggregation": "RGB mean and alpha mean before candidates",
            "distance": (
                "RGB MSE spatially averaged with G-SWM's candidate-independent "
                "predicted foreground alpha and normalized by alpha mass"
            ),
            "ground_truth_masks_used": False,
            "candidate_difference_masks_used": False,
            "selection_or_calibration_on_sp_labels": False,
            "status": "post-hoc exploratory sensitivity analysis",
        }

