"""G-SWM adapter using the official OBJ3D checkpoint and frozen SP protocol."""

from __future__ import annotations

import subprocess

import numpy as np

from gswm_runtime.runtime import (
    CHECKPOINT_SHA256,
    FUTURE_STEPS,
    GSWMObj3D,
    OBJ3D_CHECKPOINT,
    OBJ3D_CONFIG,
    OFFICIAL_REPO,
)
from sp_benchmark.adapter_contract import CandidateErrors


OFFICIAL_SOURCE_REVISION = "1423553c30e4340cbe2278639316941cfd16acb6"
OFFICIAL_CHECKPOINT_FILE_ID = "1Fo7BDbuslX7t0Ak8alkK__R9DJwK_QWj"


class Adapter:
    """Make one history-only G-SWM forecast, then score both candidates."""

    model_name = "gswm_obj3d"
    frame_spans = np.asarray([[frame, frame] for frame in range(20, 26)])

    def __init__(self, device: str, memory_mode: str = "safe") -> None:
        del memory_mode
        self.runtime = GSWMObj3D(device=device)
        self.prediction_calls = 0
        source_revision = subprocess.check_output(
            ["git", "-C", str(OFFICIAL_REPO), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        if source_revision != OFFICIAL_SOURCE_REVISION:
            raise RuntimeError(
                "Unexpected G-SWM source revision: "
                f"expected {OFFICIAL_SOURCE_REVISION}, observed {source_revision}"
            )
        self.source_revision = source_revision

    def score_pair(
        self,
        possible_rgb: np.ndarray,
        impossible_rgb: np.ndarray,
    ) -> CandidateErrors:
        # Only the shared history is available when the model is called.
        history = possible_rgb[:8]
        prediction = self.runtime.predict(history)
        self.prediction_calls += 1

        # Candidate preparation and scoring occur only after prediction exists.
        possible_error = self.runtime.pixel_mse(prediction, possible_rgb[8:])
        impossible_error = self.runtime.pixel_mse(prediction, impossible_rgb[8:])
        if possible_error.shape != (FUTURE_STEPS,) or impossible_error.shape != (
            FUTURE_STEPS,
        ):
            raise RuntimeError("G-SWM must return exactly six per-frame errors")
        return CandidateErrors(possible_error, impossible_error, self.frame_spans)

    def provenance(self) -> dict:
        state_count = int(self.runtime.checkpoint_metadata["state_tensor_count"])
        return {
            "model": "G-SWM (Generative Structured World Model)",
            "official_repository": "https://github.com/zhixuan-lin/G-SWM",
            "official_source_revision": self.source_revision,
            "pretraining_dataset": "OBJ3D",
            "config": str(OBJ3D_CONFIG),
            "checkpoint": str(OBJ3D_CHECKPOINT),
            "checkpoint_official_google_drive_file_id": OFFICIAL_CHECKPOINT_FILE_ID,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "checkpoint_epoch": int(self.runtime.checkpoint_metadata["epoch"]),
            "checkpoint_global_step": int(
                self.runtime.checkpoint_metadata["global_step"]
            ),
            "strict_state": {
                "model_keys": len(self.runtime.model.state_dict()),
                "checkpoint_keys": state_count,
                "matched_keys": state_count,
                "missing": [],
                "unexpected": [],
                "shape_mismatch": {},
                "strict_match": True,
            },
            "parameter_count": int(
                self.runtime.checkpoint_metadata["parameter_count"]
            ),
            "history_frames": [12, 19],
            "future_frames": [20, 25],
            "native_resolution": [64, 64],
            "resize": (
                "nearest neighbor, explicitly frozen to reproduce Pillow 6.2.1 "
                "Image.resize default used by the official OBJ3D loader"
            ),
            "future_sampling": {
                "foreground": False,
                "background": False,
                "history_posterior_seed": "SHA-256 derived from normalized history",
            },
            "distance": "mean squared error in native 64x64 RGB [0,1], per frame",
            "candidate_use": "post-prediction scoring only",
            "prediction_contract": "one prediction per shared history; no best-of-N",
        }
