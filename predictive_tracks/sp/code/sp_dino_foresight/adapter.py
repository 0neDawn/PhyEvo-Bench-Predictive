"""Official DINO-Foresight latent-feature predictor adapter.

The predictor sees the shared history once.  Its six autoregressive PCA-DINOv2
feature forecasts are then compared symmetrically with the two candidate
futures.  Candidate frames and benchmark labels are unavailable while the
forecast is produced.
"""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from sp_benchmark.adapter_contract import CandidateErrors
from sp_core.common import ROOT, sha256, strict_state_report


DINO_FORESIGHT_ROOT = ROOT / "third_party/DINO-Foresight"
DINO_FORESIGHT_DEPS = ROOT / "third_party/dino_foresight_python_deps"
DINO_FORESIGHT_REVISION = "d98f765d2b743ec90fb38bc657a39c07c7ca4ab0"
DINO_FORESIGHT_CHECKPOINT = (
    DINO_FORESIGHT_ROOT / "pretrained/dinof_highres.ckpt"
)
DINO_FORESIGHT_CHECKPOINT_SHA256 = (
    "4f90652bd2992186a4672a278f8d7cfd2710c38d453abfed3887604b8a4536c2"
)
PCA_CHECKPOINT = (
    DINO_FORESIGHT_ROOT
    / "pretrained/dinov2_pca_448_l[2_5_8_11]_1152.pth"
)
PCA_CHECKPOINT_SHA256 = (
    "16e897a6f09223192d26684418a157acbe08c9a74c56569707d63b7f15766f2e"
)
DINO_V2_ROOT = ROOT / "third_party/dinov2"
DINO_V2_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINO_V2_CHECKPOINT = (
    DINO_V2_ROOT / "pretrained/dinov2_vitb14_reg4_pretrain.pth"
)
DINO_V2_CHECKPOINT_SHA256 = (
    "73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71"
)

HISTORY_FRAMES = 8
NATIVE_CONTEXT = 5
FORECAST_FRAMES = 6
IMAGE_SIZE = (448, 896)
FRAME_SPANS = np.asarray([[frame, frame] for frame in range(20, 26)])


def _args() -> argparse.Namespace:
    """Return the official high-resolution inference configuration."""

    return argparse.Namespace(
        sequence_length=NATIVE_CONTEXT,
        batch_size=1,
        img_size=IMAGE_SIZE,
        feature_extractor="dino",
        dinov2_variant="vitb14_reg",
        d_layers=[2, 5, 8, 11],
        hidden_dim=1152,
        heads=8,
        layers=12,
        dropout=0.1,
        loss_type="SmoothL1",
        beta_smoothl1=0.1,
        masking="simple_replace",
        train_mask_mode="full_mask",
        seperable_attention=True,
        seperable_window_size=1,
        train_mask_frames=1,
        output_activation="none",
        use_first_last=False,
        use_fc_bias=True,
        pca_ckpt=str(PCA_CHECKPOINT),
        crop_feats=False,
        sliding_window_inference=False,
        single_step_sample_train=True,
        step=1,
        vis_attn=False,
        eval_modality=None,
        evaluate_baseline=False,
        eval_midterm=False,
        eval_mode=True,
        high_res_adapt=False,
        predict_residuals=False,
        num_classes=19,
        use_bn=False,
        use_cls=False,
        nfeats=256,
        dpt_out_channels=[128, 256, 512, 512],
        head_ckpt=None,
        lr=0.0,
        max_steps=0,
    )


@contextlib.contextmanager
def _local_dinov2_loader():
    """Redirect the official torch-hub call to pinned local source and weights."""

    original = torch.hub.load

    def load(repo_or_dir, model, *args, **kwargs):
        if repo_or_dir != "facebookresearch/dinov2" or model != "dinov2_vitb14_reg":
            return original(repo_or_dir, model, *args, **kwargs)
        kwargs.pop("pretrained", None)
        local = original(
            str(DINO_V2_ROOT), model, source="local", pretrained=False, **kwargs
        )
        state = torch.load(
            DINO_V2_CHECKPOINT, map_location="cpu", weights_only=True, mmap=True
        )
        incompatible = local.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"DINOv2 strict-load failure: {incompatible}")
        return local

    torch.hub.load = load
    try:
        yield
    finally:
        torch.hub.load = original


def _official_tensor(frames: np.ndarray) -> torch.Tensor:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected [T,H,W,3] RGB video, got {frames.shape}")
    values = []
    for frame in frames:
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).resize(
            (IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.Resampling.BILINEAR
        )
        value = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy())
        values.append(value.permute(2, 0, 1).float().div_(255.0))
    result = torch.stack(values)
    mean = result.new_tensor([0.485, 0.456, 0.406])[:, None, None]
    std = result.new_tensor([0.229, 0.224, 0.225])[:, None, None]
    return (result - mean) / std


class Adapter:
    """One-frame-native DINO-Foresight adapter with six dense rollouts."""

    model_name = "dino_foresight_cityscapes"
    frame_spans = FRAME_SPANS

    def __init__(self, device: str, memory_mode: str = "safe") -> None:
        del memory_mode
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError("DINO-Foresight official high-resolution model requires CUDA")
        required = (
            DINO_FORESIGHT_CHECKPOINT,
            PCA_CHECKPOINT,
            DINO_V2_CHECKPOINT,
            DINO_FORESIGHT_ROOT / "src/dino_f.py",
            DINO_V2_ROOT / "hubconf.py",
        )
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(path)
        if sha256(DINO_FORESIGHT_CHECKPOINT) != DINO_FORESIGHT_CHECKPOINT_SHA256:
            raise RuntimeError("DINO-Foresight checkpoint hash mismatch")
        if sha256(PCA_CHECKPOINT) != PCA_CHECKPOINT_SHA256:
            raise RuntimeError("DINO-Foresight PCA checkpoint hash mismatch")
        if sha256(DINO_V2_CHECKPOINT) != DINO_V2_CHECKPOINT_SHA256:
            raise RuntimeError("DINOv2 ViT-B/14-reg checkpoint hash mismatch")

        for source in (DINO_FORESIGHT_DEPS, DINO_FORESIGHT_ROOT, DINO_V2_ROOT):
            source_text = str(source)
            if source_text not in sys.path:
                sys.path.insert(0, source_text)
        from src.dino_f import Dino_f  # type: ignore[import-not-found]

        with _local_dinov2_loader():
            self.model = Dino_f(_args())
        payload = torch.load(
            DINO_FORESIGHT_CHECKPOINT,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        state = payload["state_dict"]
        self.strict_report = strict_state_report(self.model, state)
        if not self.strict_report["strict_match"]:
            raise RuntimeError(
                "Official DINO-Foresight checkpoint is not a strict architectural match: "
                f"{self.strict_report}"
            )
        incompatible = self.model.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"DINO-Foresight strict-load failure: {incompatible}")
        self.checkpoint_epoch = int(payload.get("epoch", -1))
        del payload, state

        self.model.to(self.device).eval().requires_grad_(False)
        self.prediction_calls = 0
        self._amp_dtype = torch.float16

    def _encode_frames(self, frames: np.ndarray) -> torch.Tensor:
        """Encode frames one at a time to bound peak memory."""

        inputs = _official_tensor(frames)
        encoded = []
        for frame in inputs:
            frame = frame.unsqueeze(0).to(self.device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=self._amp_dtype):
                features = self.model.extract_features(frame)
                pca = self.model.pca_transform(features.float())
            encoded.append(
                pca.reshape(1, IMAGE_SIZE[0] // 14, IMAGE_SIZE[1] // 14, -1)
            )
        return torch.cat(encoded, dim=0)

    def _forecast(self, history: np.ndarray) -> torch.Tensor:
        context = self._encode_frames(history[-NATIVE_CONTEXT:]).unsqueeze(0)
        predictions = []
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=self._amp_dtype
        ):
            for _ in range(FORECAST_FRAMES):
                masked, mask = self.model.get_mask_tokens(
                    context, mode="full_mask", mask_frames=1
                )
                _, output = self.model.forward(context, masked, mask)
                next_feature = output[:, -1].float()
                predictions.append(next_feature.squeeze(0))
                context = torch.cat((context[:, 1:], next_feature.unsqueeze(1)), dim=1)
        prediction = torch.stack(predictions)
        if tuple(prediction.shape[:3]) != (FORECAST_FRAMES, 32, 64):
            raise RuntimeError(f"Unexpected forecast shape: {tuple(prediction.shape)}")
        if not torch.isfinite(prediction).all():
            raise RuntimeError("DINO-Foresight produced non-finite predictions")
        return prediction

    def _candidate_errors(
        self, prediction: torch.Tensor, candidate: np.ndarray
    ) -> np.ndarray:
        errors = []
        for index in range(FORECAST_FRAMES):
            target = self._encode_frames(candidate[index : index + 1])[0]
            error = F.smooth_l1_loss(
                prediction[index], target, reduction="mean", beta=0.1
            )
            errors.append(float(error.cpu()))
            del target
        result = np.asarray(errors, dtype=np.float64)
        if not np.isfinite(result).all() or (result < 0).any():
            raise RuntimeError("DINO-Foresight candidate errors are invalid")
        return result

    def score_pair(
        self, possible_rgb: np.ndarray, impossible_rgb: np.ndarray
    ) -> CandidateErrors:
        if possible_rgb.shape != impossible_rgb.shape:
            raise ValueError("Candidate videos have different shapes")
        if possible_rgb.shape[0] != HISTORY_FRAMES + FORECAST_FRAMES:
            raise ValueError(f"Expected 14 benchmark frames, got {possible_rgb.shape[0]}")
        if not np.array_equal(
            possible_rgb[:HISTORY_FRAMES], impossible_rgb[:HISTORY_FRAMES]
        ):
            raise RuntimeError("Candidate histories are not identical")

        # The only forecast is completed before either candidate future is encoded.
        prediction = self._forecast(possible_rgb[:HISTORY_FRAMES])
        self.prediction_calls += 1
        possible_error = self._candidate_errors(
            prediction, possible_rgb[HISTORY_FRAMES:]
        )
        impossible_error = self._candidate_errors(
            prediction, impossible_rgb[HISTORY_FRAMES:]
        )
        return CandidateErrors(possible_error, impossible_error, self.frame_spans)

    def provenance(self) -> dict:
        return {
            "model": "DINO-Foresight (DINOv2 ViT-B/14-reg)",
            "pretraining_dataset": (
                "Cityscapes for DINO-Foresight/PCA; LVD-142M for DINOv2"
            ),
            "official_repository": "https://github.com/Sta8is/DINO-Foresight",
            "official_source_revision": DINO_FORESIGHT_REVISION,
            "dinov2_repository": "https://github.com/facebookresearch/dinov2",
            "dinov2_source_revision": DINO_V2_REVISION,
            "architecture": (
                "DINOv2 ViT-B/14-reg layers [2,5,8,11], PCA-1152, "
                "12-layer masked feature transformer"
            ),
            "official_training_temporal_sampling": (
                "Cityscapes training uses randomized stride 2-4; official validation "
                "uses stride 3"
            ),
            "benchmark_history": (
                "shared frames 12-19 are available; the official five-frame context "
                "uses the final five frames 15-19"
            ),
            "benchmark_predictions": (
                "six dense autoregressive one-frame steps mapped to frames 20-25"
            ),
            "preprocessing": (
                "official forced resize to 448x896 and ImageNet normalization"
            ),
            "prediction_space": "PCA-compressed multi-layer DINOv2 feature space",
            "distance": "per-frame SmoothL1(beta=0.1) mean in PCA-DINOv2 space",
            "frame_spans": self.frame_spans.tolist(),
            "native_prediction_step_frames": 1,
            "joint_hit_tolerance_frames": 1,
            "candidate_use": "post-prediction symmetric latent-distance scoring only",
            "labels_available_to_predictor": False,
            "checkpoint": str(DINO_FORESIGHT_CHECKPOINT),
            "checkpoint_bytes": DINO_FORESIGHT_CHECKPOINT.stat().st_size,
            "checkpoint_sha256": DINO_FORESIGHT_CHECKPOINT_SHA256,
            "checkpoint_epoch": self.checkpoint_epoch,
            "pca_checkpoint": str(PCA_CHECKPOINT),
            "pca_checkpoint_bytes": PCA_CHECKPOINT.stat().st_size,
            "pca_checkpoint_sha256": PCA_CHECKPOINT_SHA256,
            "dinov2_checkpoint": str(DINO_V2_CHECKPOINT),
            "dinov2_checkpoint_bytes": DINO_V2_CHECKPOINT.stat().st_size,
            "dinov2_checkpoint_sha256": DINO_V2_CHECKPOINT_SHA256,
            "strict_state": self.strict_report,
            "precision": "PyTorch autocast float16 inference; PCA/loss accumulation float32",
            "pca_runtime": "project-isolated scikit-learn 1.5.2 (official requirement)",
        }
