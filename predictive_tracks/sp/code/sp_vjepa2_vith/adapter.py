"""V-JEPA 2 ViT-H adapter for the frozen SP predictive protocol.

This module is intentionally separate from the existing ViT-g adapter.  The
predictor sees the shared history once; possible/impossible futures are used
only after prediction to construct candidate target-encoder states.
"""

from __future__ import annotations

from release_paths import project_path as _release_path

import copy
import gc
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from sp_benchmark.adapter_contract import CandidateErrors
from sp_core.common import clean_state, empty_construction, load_strict, sha256


VJEPA_SOURCE_ROOT = Path(_release_path('third_party/vjepa2'))
SP_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = SP_ROOT / "checkpoints/vjepa2-vith-fpc64-256"
CHECKPOINT = MODEL_ROOT / "original/model.pth"
CONFIG = MODEL_ROOT / "config.json"
PROCESSOR_CONFIG = MODEL_ROOT / "video_preprocessor_config.json"

OFFICIAL_REPOSITORY = "facebook/vjepa2-vith-fpc64-256"
OFFICIAL_REPOSITORY_REVISION = "b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5"
CHECKPOINT_BYTES = 10_374_445_514
CHECKPOINT_SHA256 = "2f18bab855c310d75fac12bfdb64f2dc2e2e048979f3fb9312c1d97a713b33cd"
MODEL_NAME = "vit_huge"
TOKENS_PER_TUBELET = 16 * 16

RESIZE_SHORTEST_EDGE = 292
CROP_SIZE = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def verify_checkpoint(path: Path = CHECKPOINT, *, verify_hash: bool = True) -> dict:
    """Reject partial, substituted, or otherwise unverifiable model files."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing V-JEPA 2 ViT-H checkpoint: {path}")
    observed_bytes = path.stat().st_size
    if observed_bytes != CHECKPOINT_BYTES:
        raise RuntimeError(
            "Incomplete or unexpected V-JEPA 2 ViT-H checkpoint: "
            f"observed {observed_bytes} bytes, expected {CHECKPOINT_BYTES}"
        )
    observed_hash = sha256(path) if verify_hash else None
    if verify_hash and observed_hash != CHECKPOINT_SHA256:
        raise RuntimeError(
            "V-JEPA 2 ViT-H checkpoint SHA-256 mismatch: "
            f"observed {observed_hash}, expected {CHECKPOINT_SHA256}"
        )
    return {
        "path": str(path),
        "bytes": observed_bytes,
        "sha256": observed_hash,
        "official_sha256": CHECKPOINT_SHA256,
        "verified": bool(verify_hash),
    }


def verify_official_configs() -> dict:
    """Check the local metadata that determines architecture and preprocessing."""

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    processor = json.loads(PROCESSOR_CONFIG.read_text(encoding="utf-8"))
    expected_model = {
        "hidden_size": 1280,
        "num_hidden_layers": 32,
        "num_attention_heads": 16,
        "patch_size": 16,
        "tubelet_size": 2,
        "image_size": 256,
        "pred_hidden_size": 384,
        "pred_num_hidden_layers": 12,
        "pred_num_attention_heads": 12,
        "pred_num_mask_tokens": 10,
    }
    mismatches = {
        key: {"observed": config.get(key), "expected": value}
        for key, value in expected_model.items()
        if config.get(key) != value
    }
    expected_processor = {
        "crop_size": {"height": CROP_SIZE, "width": CROP_SIZE},
        "size": {"shortest_edge": RESIZE_SHORTEST_EDGE},
        "image_mean": list(IMAGENET_MEAN),
        "image_std": list(IMAGENET_STD),
        "resample": 2,
    }
    processor_mismatches = {
        key: {"observed": processor.get(key), "expected": value}
        for key, value in expected_processor.items()
        if processor.get(key) != value
    }
    if mismatches or processor_mismatches:
        raise RuntimeError(
            json.dumps(
                {"model_config": mismatches, "processor_config": processor_mismatches},
                ensure_ascii=False,
            )
        )
    return {
        "model_config": str(CONFIG),
        "model_config_sha256": sha256(CONFIG),
        "processor_config": str(PROCESSOR_CONFIG),
        "processor_config_sha256": sha256(PROCESSOR_CONFIG),
        "architecture_checks": expected_model,
        "preprocessing_checks": expected_processor,
    }


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


class Adapter:
    model_name = "vjepa2_vith_official"
    frame_spans = np.asarray([[20, 21], [22, 23], [24, 25]], dtype=np.int64)

    def __init__(self, device: str, memory_mode: str = "safe") -> None:
        if memory_mode not in ("safe", "all_gpu"):
            raise ValueError("memory_mode must be safe or all_gpu")
        self.device = torch.device(device)
        self.memory_mode = memory_mode
        self.prediction_calls = 0
        self.checkpoint_identity = verify_checkpoint()
        self.config_identity = verify_official_configs()

        if str(VJEPA_SOURCE_ROOT) not in sys.path:
            sys.path.insert(0, str(VJEPA_SOURCE_ROOT))
        from src.hub.backbones import _make_vjepa2_model
        import src.models.predictor  # noqa: F401
        import src.models.vision_transformer  # noqa: F401

        with empty_construction():
            self.encoder, self.predictor = _make_vjepa2_model(
                model_name=MODEL_NAME,
                img_size=CROP_SIZE,
                num_frames=14,
                pretrained=False,
            )
            self.target_encoder = copy.deepcopy(self.encoder)

        checkpoint = torch.load(
            CHECKPOINT, map_location="cpu", weights_only=True, mmap=True
        )
        required = {"encoder", "target_encoder", "predictor"}
        missing_sections = sorted(required - checkpoint.keys())
        if missing_sections:
            raise RuntimeError(f"Checkpoint lacks required states: {missing_sections}")
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
        self.parameter_counts = {
            "encoder": sum(parameter.numel() for parameter in self.encoder.parameters()),
            "target_encoder": sum(
                parameter.numel() for parameter in self.target_encoder.parameters()
            ),
            "predictor": sum(parameter.numel() for parameter in self.predictor.parameters()),
        }
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
        context_mask = torch.arange(4 * TOKENS_PER_TUBELET, device=self.device)[None]
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
        with torch.inference_mode():
            for step in range(3):
                prefix_end = 8 + 2 * (step + 1)
                encoded = self.target_encoder(candidates[:, :, :prefix_end])
                encoded = F.layer_norm(encoded, (encoded.shape[-1],))
                target_steps.append(encoded[:, -TOKENS_PER_TUBELET:])
        targets = torch.stack(target_steps, dim=1)
        return targets[0:1], targets[1:2]

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
            raise RuntimeError("V-JEPA 2 ViT-H produced non-finite values")
        result = CandidateErrors(
            errors[0].cpu().numpy(), errors[1].cpu().numpy(), self.frame_spans
        )
        if self.memory_mode == "safe" and self.device.type == "cuda":
            self.target_encoder.cpu()
            self._clear_cuda()
        return result

    def provenance(self) -> dict:
        return {
            "model": "V-JEPA 2 ViT-H/16, 256px, FPC64",
            "architecture_builder": MODEL_NAME,
            "not_vjepa_1": True,
            "not_vjepa_2_ac": True,
            "official_repository": OFFICIAL_REPOSITORY,
            "official_repository_revision": OFFICIAL_REPOSITORY_REVISION,
            "checkpoint": str(CHECKPOINT),
            "checkpoint_bytes": CHECKPOINT_BYTES,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "checkpoint_identity": self.checkpoint_identity,
            "configuration_identity": self.config_identity,
            "strict_states": self.state_reports,
            "parameter_counts": self.parameter_counts,
            "distance": (
                "L1 in layer-normalized target-encoder latent space, "
                "per two-frame tubelet"
            ),
            "target_causality": (
                "candidate prefix ends at the scored tubelet; no later candidate "
                "frame is encoded"
            ),
            "candidate_use": "post-prediction target encoding only",
            "prediction_input": "shared history frames 12-19 only",
            "spatial_preprocessing": {
                "resize_shortest_edge": RESIZE_SHORTEST_EDGE,
                "center_crop": [CROP_SIZE, CROP_SIZE],
                "resampling": "Pillow bilinear",
                "normalization_mean": list(IMAGENET_MEAN),
                "normalization_std": list(IMAGENET_STD),
            },
            "score_adjustment": "none",
            "selection_or_calibration_on_sp_labels": False,
            "memory_mode": self.memory_mode,
        }
