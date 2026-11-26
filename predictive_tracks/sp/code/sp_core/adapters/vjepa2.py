"""V-JEPA 2 adapter with causal-prefix target encoding for localization."""

from __future__ import annotations

from release_paths import project_path as _release_path

import copy
import gc
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

from sp_benchmark.adapter_contract import CandidateErrors
from sp_core.common import (
    clean_state,
    empty_construction,
    load_strict,
    video_tensor_bcthw,
)


VJEPA_ROOT = Path(_release_path('third_party/vjepa2'))
CHECKPOINT = VJEPA_ROOT / "vjepa2-vitg-fpc64-256/original/model.pth"
CHECKPOINT_SHA256 = "67129f011434e605d894e69f2c8e13d9db118deabe59d54bf6e0fa62c2c5cb8e"
TOKENS_PER_TUBELET = 16 * 16


class Adapter:
    model_name = "vjepa2"
    frame_spans = np.asarray([[20, 21], [22, 23], [24, 25]])

    def __init__(self, device: str, memory_mode: str = "safe") -> None:
        if memory_mode not in ("safe", "all_gpu"):
            raise ValueError("memory_mode must be safe or all_gpu")
        self.device = torch.device(device)
        self.memory_mode = memory_mode
        self.prediction_calls = 0
        sys.path.insert(0, str(VJEPA_ROOT))
        from src.hub.backbones import _make_vjepa2_model
        import src.models.predictor  # noqa: F401
        import src.models.vision_transformer  # noqa: F401

        with empty_construction():
            self.encoder, self.predictor = _make_vjepa2_model(
                model_name="vit_giant", img_size=256, num_frames=14, pretrained=False
            )
            self.target_encoder = copy.deepcopy(self.encoder)
        checkpoint = torch.load(
            CHECKPOINT, map_location="cpu", weights_only=True, mmap=True
        )
        self.state_reports = {
            "encoder": load_strict(
                self.encoder,
                clean_state(checkpoint["encoder"], backbone=True),
                assign=True,
            ),
            "target_encoder": load_strict(
                self.target_encoder,
                clean_state(checkpoint["target_encoder"], backbone=True),
                assign=True,
            ),
            "predictor": load_strict(
                self.predictor,
                clean_state(checkpoint["predictor"], backbone=True),
                assign=True,
            ),
        }
        del checkpoint
        for model in (self.encoder, self.target_encoder, self.predictor):
            model.eval().requires_grad_(False)
        if self.memory_mode == "all_gpu" or self.device.type == "cpu":
            self.encoder.to(self.device)
            self.target_encoder.to(self.device)
            self.predictor.to(self.device)

    def _clear_cuda(self) -> None:
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _predict(self, model_input: torch.Tensor) -> torch.Tensor:
        if self.memory_mode == "safe" and self.device.type == "cuda":
            self.encoder.to(self.device)
            self.predictor.to(self.device)
        context_mask = torch.arange(
            4 * TOKENS_PER_TUBELET, device=self.device
        )[None]
        future_mask = torch.arange(
            4 * TOKENS_PER_TUBELET,
            7 * TOKENS_PER_TUBELET,
            device=self.device,
        )[None]
        with torch.inference_mode():
            context = self.encoder(model_input, [context_mask])
            prediction = self.predictor(
                context, [context_mask], [future_mask], mask_index=0
            )
        self.prediction_calls += 1
        return prediction

    def _causal_targets(
        self, possible: torch.Tensor, impossible: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.memory_mode == "safe" and self.device.type == "cuda":
            self.encoder.cpu()
            self.predictor.cpu()
            self._clear_cuda()
            self.target_encoder.to(self.device)
        candidates = torch.cat([possible, impossible], dim=0)
        target_steps = []
        # Encode each candidate only through the current tubelet. This prevents
        # frames 22-25 from influencing the target at frames 20-21, etc.
        with torch.inference_mode():
            for step in range(3):
                prefix_end = 8 + 2 * (step + 1)
                encoded = self.target_encoder(candidates[:, :, :prefix_end])
                encoded = F.layer_norm(encoded, (encoded.shape[-1],))
                target_steps.append(encoded[:, -TOKENS_PER_TUBELET:])
        targets = torch.stack(target_steps, dim=1)
        return targets[0:1], targets[1:2]

    def score_pair(self, possible_rgb: np.ndarray, impossible_rgb: np.ndarray) -> CandidateErrors:
        possible = video_tensor_bcthw(possible_rgb, 256).to(self.device)
        impossible = video_tensor_bcthw(impossible_rgb, 256).to(self.device)
        model_input = torch.zeros_like(possible)
        model_input[:, :, :8] = possible[:, :, :8]
        prediction = self._predict(model_input).reshape(
            1, 3, TOKENS_PER_TUBELET, -1
        )
        possible_target, impossible_target = self._causal_targets(possible, impossible)
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
        return {
            "model": "V-JEPA 2 ViT-g/16, 256px, FPC64",
            "not_vjepa_1": True,
            "not_vjepa_2_ac": True,
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "strict_states": self.state_reports,
            "distance": "L1 in layer-normalized target-encoder latent space, per two-frame tubelet",
            "target_causality": "candidate prefix ends at the scored tubelet; no later candidate frame is encoded",
            "candidate_use": "post-prediction target encoding only",
            "memory_mode": self.memory_mode,
        }

