"""V-JEPA 2 with the preprocessing shipped with its official checkpoint.

This changes neither the checkpoint nor the prediction/distance definition.
It only replaces the earlier direct 128->256 resize by the checkpoint's
documented resize-short-side-to-292 then center-crop-to-256 transform.
"""

from __future__ import annotations

from release_paths import project_path as _release_path

from pathlib import Path

import numpy as np
from PIL import Image
import torch

from sp_benchmark.adapter_contract import CandidateErrors
from sp_core.adapters.vjepa2 import (
    Adapter as FrozenVJEPA2Adapter,
    TOKENS_PER_TUBELET,
)
from sp_core.common import sha256


PROCESSOR_CONFIG = Path(
    _release_path('third_party/vjepa2/vjepa2-vitg-fpc64-256/video_preprocessor_config.json')
)
RESIZE_SHORTEST_EDGE = 292
CROP_SIZE = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def official_video_tensor(frames: np.ndarray) -> torch.Tensor:
    """Return B,C,T,H,W using the checkpoint's deterministic eval transform."""

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected THWC RGB frames, observed {frames.shape}")
    transformed = []
    for frame in frames:
        image = Image.fromarray(frame.astype(np.uint8, copy=False), mode="RGB")
        width, height = image.size
        scale = RESIZE_SHORTEST_EDGE / min(width, height)
        resized = image.resize(
            (round(width * scale), round(height * scale)),
            resample=Image.Resampling.BILINEAR,
        )
        left = (resized.width - CROP_SIZE) // 2
        top = (resized.height - CROP_SIZE) // 2
        image = resized.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))
        transformed.append(np.asarray(image, dtype=np.uint8).copy())

    value = torch.from_numpy(np.stack(transformed)).permute(0, 3, 1, 2).float()
    value = value / 255.0
    mean = value.new_tensor(IMAGENET_MEAN)[None, :, None, None]
    std = value.new_tensor(IMAGENET_STD)[None, :, None, None]
    value = (value - mean) / std
    return value.permute(1, 0, 2, 3).unsqueeze(0).contiguous()


class Adapter(FrozenVJEPA2Adapter):
    """The frozen V-JEPA2 adapter with official spatial preprocessing."""

    model_name = "vjepa2_official_preprocess"

    def score_pair(
        self, possible_rgb: np.ndarray, impossible_rgb: np.ndarray
    ) -> CandidateErrors:
        possible = official_video_tensor(possible_rgb).to(self.device)
        impossible = official_video_tensor(impossible_rgb).to(self.device)
        model_input = torch.zeros_like(possible)
        model_input[:, :, :8] = possible[:, :, :8]
        prediction = self._predict(model_input).reshape(
            1, 3, TOKENS_PER_TUBELET, -1
        )
        possible_target, impossible_target = self._causal_targets(
            possible, impossible
        )
        errors = [
            (prediction - target).abs().mean(dim=(0, 2, 3))
            for target in (possible_target, impossible_target)
        ]
        if not all(torch.isfinite(item).all() for item in (prediction, *errors)):
            raise RuntimeError("V-JEPA 2 produced non-finite values")
        result = CandidateErrors(
            errors[0].cpu().numpy(), errors[1].cpu().numpy(), self.frame_spans
        )
        if self.memory_mode == "safe" and self.device.type == "cuda":
            self.target_encoder.cpu()
            self._clear_cuda()
        return result

    def provenance(self) -> dict:
        value = super().provenance()
        value.update(
            {
                "evaluation_variant": "official checkpoint preprocessing correction",
                "processor_config": str(PROCESSOR_CONFIG),
                "processor_config_sha256": sha256(PROCESSOR_CONFIG),
                "spatial_preprocessing": {
                    "resize_shortest_edge": RESIZE_SHORTEST_EDGE,
                    "center_crop": [CROP_SIZE, CROP_SIZE],
                    "resampling": "Pillow bilinear",
                    "normalization_mean": list(IMAGENET_MEAN),
                    "normalization_std": list(IMAGENET_STD),
                },
                "score_adjustment": "none",
                "selection_or_calibration_on_sp_labels": False,
            }
        )
        return value

