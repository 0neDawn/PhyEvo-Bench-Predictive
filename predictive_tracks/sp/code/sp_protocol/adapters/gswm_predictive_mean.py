"""G-SWM scored from one candidate-independent posterior-predictive mean.

The official G-SWM history encoder samples posterior state even when future
sampling is disabled.  A single arbitrary draw adds avoidable evaluation
variance.  We therefore average eight fixed, history-derived draws *before*
either candidate is scored.  This is one predictive-mean output, not best-of-N.
"""

from __future__ import annotations

from dataclasses import dataclass
import subprocess

import numpy as np

from gswm_runtime.runtime import (
    CHECKPOINT_SHA256,
    FUTURE_STEPS,
    GSWMObj3D,
    OBJ3D_CHECKPOINT,
    OBJ3D_CONFIG,
    OFFICIAL_REPO,
    seed_from_history,
)
from sp_benchmark.adapter_contract import CandidateErrors
from sp_core.adapters.gswm_obj3d import (
    OFFICIAL_CHECKPOINT_FILE_ID,
    OFFICIAL_SOURCE_REVISION,
)


MC_SAMPLES = 8
SEED_STRIDE = 104729


@dataclass(frozen=True)
class PredictiveMean:
    rgb: np.ndarray


class Adapter:
    model_name = "gswm_obj3d_predictive_mean"
    frame_spans = np.asarray([[frame, frame] for frame in range(20, 26)])

    def __init__(self, device: str, memory_mode: str = "safe") -> None:
        del memory_mode
        self.runtime = GSWMObj3D(device=device)
        self.prediction_calls = 0
        source_revision = subprocess.check_output(
            ["git", "-C", str(OFFICIAL_REPO), "rev-parse", "HEAD"], text=True
        ).strip()
        if source_revision != OFFICIAL_SOURCE_REVISION:
            raise RuntimeError(
                "Unexpected G-SWM source revision: "
                f"expected {OFFICIAL_SOURCE_REVISION}, observed {source_revision}"
            )
        self.source_revision = source_revision

    def _predictive_mean(self, history: np.ndarray) -> PredictiveMean:
        base_seed = seed_from_history(history)
        samples = []
        for index in range(MC_SAMPLES):
            seed = (base_seed + index * SEED_STRIDE) % (2**31)
            samples.append(self.runtime.predict(history, seed=seed).rgb)
        rgb = np.mean(np.stack(samples, axis=0), axis=0, dtype=np.float64)
        if rgb.shape != (FUTURE_STEPS, 3, 64, 64):
            raise RuntimeError(f"Unexpected G-SWM predictive mean shape: {rgb.shape}")
        if not np.isfinite(rgb).all():
            raise RuntimeError("G-SWM predictive mean contains NaN or infinity")
        return PredictiveMean(rgb=rgb.astype(np.float32))

    def score_pair(
        self, possible_rgb: np.ndarray, impossible_rgb: np.ndarray
    ) -> CandidateErrors:
        history = possible_rgb[:8]
        prediction = self._predictive_mean(history)
        self.prediction_calls += 1
        possible_error = self.runtime.pixel_mse(prediction, possible_rgb[8:])
        impossible_error = self.runtime.pixel_mse(prediction, impossible_rgb[8:])
        return CandidateErrors(possible_error, impossible_error, self.frame_spans)

    def provenance(self) -> dict:
        state_count = int(self.runtime.checkpoint_metadata["state_tensor_count"])
        return {
            "model": "G-SWM official OBJ3D",
            "evaluation_variant": "fixed posterior-predictive RGB mean",
            "official_repository": "https://github.com/zhixuan-lin/G-SWM",
            "official_source_revision": self.source_revision,
            "pretraining_dataset": "OBJ3D",
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
            "history_frames": [12, 19],
            "future_frames": [20, 25],
            "posterior_predictive_samples": MC_SAMPLES,
            "sample_seeds": (
                "SHA-256 history seed + index*104729 modulo 2^31; candidate-independent"
            ),
            "aggregation": "arithmetic RGB mean before candidate scoring",
            "prediction_contract": (
                "one predictive-mean future per history; no best-of-N and no candidate "
                "is available during prediction"
            ),
            "distance": "mean squared error in native 64x64 RGB [0,1], per frame",
            "score_adjustment": "none",
            "selection_or_calibration_on_sp_labels": False,
        }

