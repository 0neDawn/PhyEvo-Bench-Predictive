"""VideoMAE-V2 with the 16-frame temporal geometry used by its checkpoint."""

from __future__ import annotations

import sys

import numpy as np
import torch

from sp_benchmark.adapter_contract import CandidateErrors
from sp_core.common import (
    ROOT,
    clean_state,
    empty_construction,
    load_strict,
    video_tensor_bcthw,
)


VIDEOMAE = ROOT / "VideoMAEv2"
CHECKPOINT = VIDEOMAE / "checkpoints/vit_g_hybrid_pt_1200e.pth"
CHECKPOINT_SHA256 = "5773c881d303426f2f0e906df9842c4ca95fba6a5901849a3e8d9d1ed8f79b5f"
TOKENS_PER_TUBELET = 16 * 16
TOTAL_TUBELETS = 8
CONTEXT_TUBELETS = 4
SCORED_FUTURE_TUBELETS = 3


class Adapter:
    model_name = "videomaev2_official16"
    frame_spans = np.asarray([[20, 21], [22, 23], [24, 25]])

    def __init__(self, device: str, memory_mode: str = "safe") -> None:
        del memory_mode
        self.device = torch.device(device)
        self.prediction_calls = 0
        sys.path.insert(0, str(VIDEOMAE))
        import models
        from models.modeling_pretrain import get_sinusoid_encoding_table

        with empty_construction():
            self.model = models.pretrain_videomae_giant_patch14_224(
                pretrained=False,
                all_frames=16,
                tubelet_size=2,
                decoder_depth=4,
            )
        checkpoint = torch.load(
            CHECKPOINT, map_location="cpu", weights_only=False, mmap=True
        )
        raw_state = checkpoint.get("model", checkpoint.get("module", checkpoint))
        state = clean_state(raw_state)
        self.state_report = load_strict(self.model, state, assign=True)
        del checkpoint, raw_state, state

        total_tokens = TOTAL_TUBELETS * TOKENS_PER_TUBELET
        self.model.pos_embed = get_sinusoid_encoding_table(
            total_tokens, self.model.decoder.head.in_features
        )
        self.model.encoder.pos_embed = get_sinusoid_encoding_table(
            total_tokens, self.model.encoder.embed_dim
        )
        self.model.to(self.device).eval().requires_grad_(False)

        context_tokens = CONTEXT_TUBELETS * TOKENS_PER_TUBELET
        self.mask = torch.zeros(1, total_tokens, dtype=torch.bool, device=self.device)
        self.mask[:, context_tokens:] = True
        self.decode_mask = ~self.mask

    @staticmethod
    def _patch_targets(x: torch.Tensor) -> torch.Tensor:
        mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None, None]
        std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None, None]
        value = x * std + mean
        b, c, t, h, w = value.shape
        value = value.reshape(b, c, t // 2, 2, h // 14, 14, w // 14, 14)
        value = value.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        value = value.reshape(b, -1, 2 * 14 * 14, c)
        value = (value - value.mean(dim=-2, keepdim=True)) / (
            value.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6
        )
        return value.reshape(b, value.shape[1], -1)

    def _pad_to_official_16(self, frames: np.ndarray) -> torch.Tensor:
        value = video_tensor_bcthw(frames, 224).to(self.device)
        if value.shape[2] != 14:
            raise ValueError(f"Expected 14 SP frames, observed {value.shape[2]}")
        padding = torch.zeros(
            value.shape[0], value.shape[1], 2, value.shape[3], value.shape[4],
            dtype=value.dtype, device=value.device,
        )
        return torch.cat([value, padding], dim=2)

    def score_pair(
        self, possible_rgb: np.ndarray, impossible_rgb: np.ndarray
    ) -> CandidateErrors:
        possible = self._pad_to_official_16(possible_rgb)
        impossible = self._pad_to_official_16(impossible_rgb)
        model_input = torch.zeros_like(possible)
        model_input[:, :, :8] = possible[:, :, :8]
        with torch.inference_mode():
            prediction = self.model(model_input, self.mask, self.decode_mask)
        self.prediction_calls += 1

        scored_tokens = SCORED_FUTURE_TUBELETS * TOKENS_PER_TUBELET
        prediction = prediction[:, :scored_tokens]
        errors = []
        with torch.inference_mode():
            for candidate in (possible, impossible):
                all_targets = self._patch_targets(candidate)
                future_targets = all_targets[self.mask].reshape(
                    1, (TOTAL_TUBELETS - CONTEXT_TUBELETS) * TOKENS_PER_TUBELET, -1
                )
                target = future_targets[:, :scored_tokens]
                error = (prediction - target).square().reshape(
                    1, SCORED_FUTURE_TUBELETS, TOKENS_PER_TUBELET, -1
                ).mean(dim=(0, 2, 3))
                errors.append(error)
        if not all(torch.isfinite(item).all() for item in (prediction, *errors)):
            raise RuntimeError("VideoMAE-V2 produced non-finite values")
        return CandidateErrors(
            errors[0].cpu().numpy(), errors[1].cpu().numpy(), self.frame_spans
        )

    def provenance(self) -> dict:
        return {
            "model": "VideoMAE-V2 ViT-g/14 hybrid pretraining 1200e",
            "evaluation_variant": "checkpoint-native 16-frame temporal geometry",
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "checkpoint_bytes": CHECKPOINT.stat().st_size,
            "strict_state": self.state_report,
            "checkpoint_native_frames": 16,
            "observed_history_frames": [12, 19],
            "masked_future_slots": [20, 27],
            "scored_future_frames": [20, 25],
            "unscored_padding_slots": [26, 27],
            "padding_value": "zero in normalized model-input space; always masked",
            "distance": (
                "official normalized masked-patch mean squared error, per two-frame "
                "tubelet"
            ),
            "candidate_use": "targets formed only after one future-masked prediction",
            "prediction_contract": "one prediction per shared history",
            "score_adjustment": "none",
            "selection_or_calibration_on_sp_labels": False,
        }
