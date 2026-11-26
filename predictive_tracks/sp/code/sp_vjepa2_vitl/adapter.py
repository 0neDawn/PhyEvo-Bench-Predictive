"""Official V-JEPA 2 ViT-L adapter for the frozen SP protocol.

The temporal prediction, causal candidate encoding, native distance, and
official preprocessing are intentionally identical to the audited ViT-H run.
Only the officially paired architecture and checkpoint are changed.
"""

from __future__ import annotations

from release_paths import project_path as _release_path

import copy
import json
from pathlib import Path
import sys

import torch

from sp_core.common import clean_state, empty_construction, load_strict, sha256
from sp_vjepa2_vith.adapter import (
    Adapter as AuditedVJEPA2Adapter,
    CROP_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    RESIZE_SHORTEST_EDGE,
)


VJEPA_SOURCE_ROOT = Path(_release_path('third_party/vjepa2'))
SP_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = SP_ROOT / "checkpoints/vjepa2-vitl-fpc64-256"
CHECKPOINT = MODEL_ROOT / "original/model.pth"
CONFIG = MODEL_ROOT / "config.json"
PROCESSOR_CONFIG = MODEL_ROOT / "video_preprocessor_config.json"

OFFICIAL_REPOSITORY = "facebook/vjepa2-vitl-fpc64-256"
OFFICIAL_REPOSITORY_REVISION = "b3c1679b7c34d3255ef3547f27c7b226aefab26f"
CHECKPOINT_BYTES = 5_127_726_842
CHECKPOINT_SHA256 = "5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012"
MODEL_NAME = "vit_large"


def verify_checkpoint(path: Path = CHECKPOINT, *, verify_hash: bool = True) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing V-JEPA 2 ViT-L checkpoint: {path}")
    observed_bytes = path.stat().st_size
    if observed_bytes != CHECKPOINT_BYTES:
        raise RuntimeError(
            "Incomplete or unexpected V-JEPA 2 ViT-L checkpoint: "
            f"observed {observed_bytes} bytes, expected {CHECKPOINT_BYTES}"
        )
    observed_hash = sha256(path) if verify_hash else None
    if verify_hash and observed_hash != CHECKPOINT_SHA256:
        raise RuntimeError(
            "V-JEPA 2 ViT-L checkpoint SHA-256 mismatch: "
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
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    processor = json.loads(PROCESSOR_CONFIG.read_text(encoding="utf-8"))
    expected_model = {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
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


class Adapter(AuditedVJEPA2Adapter):
    model_name = "vjepa2_vitl_official"

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

    def provenance(self) -> dict:
        return {
            "model": "V-JEPA 2 ViT-L/16, 256px, FPC64",
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
