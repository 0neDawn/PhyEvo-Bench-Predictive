"""DDLP OBJ3D adapter: one decoded prediction, two post-hoc RGB-L1 scores."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import numpy as np
import torch

from sp_benchmark.adapter_contract import CandidateErrors
from sp_core.common import ROOT, load_strict, video_tensor


DDLP_ROOT = ROOT / "ddlp"
CONFIG = DDLP_ROOT / "checkpoints/diffuse-ddlp-obj3d128/ddlp_hparams.json"
CHECKPOINT = DDLP_ROOT / "checkpoints/diffuse-ddlp-obj3d128/obj3d128_ddlp.pth"
CHECKPOINT_SHA256 = "4f5b416c44d6288e03656e899a00ae0370cfab8f0aa7d16a8cb794057066283d"


def _build_model(config: dict):
    sys.path.insert(0, str(DDLP_ROOT))
    from models import ObjectDynamicsDLP

    return ObjectDynamicsDLP(
        cdim=config["ch"], enc_channels=config["enc_channels"],
        prior_channels=config["prior_channels"], image_size=config["image_size"],
        n_kp=config["n_kp"], learned_feature_dim=config["learned_feature_dim"],
        pad_mode=config["pad_mode"], sigma=config["sigma"], dropout=config["dropout"],
        patch_size=config["patch_size"], n_kp_enc=config["n_kp_enc"],
        n_kp_prior=config["n_kp_prior"], kp_range=config["kp_range"],
        kp_activation=config["kp_activation"], anchor_s=config["anchor_s"],
        use_resblock=config["use_resblock"], timestep_horizon=config["timestep_horizon"],
        predict_delta=config["predict_delta"], scale_std=config["scale_std"],
        offset_std=config["offset_std"], obj_on_alpha=config["obj_on_alpha"],
        obj_on_beta=config["obj_on_beta"], pint_heads=config["pint_heads"],
        pint_layers=config["pint_layers"], pint_dim=config["pint_dim"],
        use_correlation_heatmaps=config["use_correlation_heatmaps"],
        enable_enc_attn=config["enable_enc_attn"],
        filtering_heuristic=config["filtering_heuristic"],
    )


def _install_torch26_compatibility(model) -> None:
    def encode_bg_features(module, x, masks=None):
        batch_size = x.shape[0]
        x_in = x * masks if masks is not None else x
        _, cnn_features = module.bg_cnn_enc(x_in)
        bg_enc_out = module.bg_enc(cnn_features.reshape(batch_size, -1))
        return bg_enc_out.chunk(2, dim=-1)

    model.bg_module.encode_bg_features = types.MethodType(
        encode_bg_features, model.bg_module
    )


class Adapter:
    model_name = "ddlp_obj3d"
    frame_spans = np.asarray([[frame, frame] for frame in range(20, 26)])

    def __init__(self, device: str, memory_mode: str = "safe") -> None:
        del memory_mode
        self.device = torch.device(device)
        self.prediction_calls = 0
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        if config.get("ds") != "obj3d128":
            raise RuntimeError(f"DDLP config is not OBJ3D: {config.get('ds')}")
        self.model = _build_model(config)
        state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        self.state_report = load_strict(self.model, state)
        _install_torch26_compatibility(self.model)
        self.model.to(self.device).eval().requires_grad_(False)

    def score_pair(self, possible_rgb: np.ndarray, impossible_rgb: np.ndarray) -> CandidateErrors:
        possible = video_tensor(possible_rgb, 128).to(self.device)
        impossible = video_tensor(impossible_rgb, 128).to(self.device)
        # DDLP requires a tensor with history+requested-output length, but sample()
        # reads only [:cond_steps]. Candidate pixels are never placed in this input.
        model_input = torch.zeros_like(possible)
        model_input[:, :8] = possible[:, :8]
        with torch.inference_mode():
            prediction = self.model.sample(
                model_input, num_steps=6, cond_steps=8, deterministic=True
            )[:, 8:]
        self.prediction_calls += 1
        if not torch.isfinite(prediction).all():
            raise RuntimeError("DDLP produced non-finite prediction")
        possible_error = (prediction - possible[:, 8:]).abs().mean(dim=(0, 2, 3, 4))
        impossible_error = (prediction - impossible[:, 8:]).abs().mean(dim=(0, 2, 3, 4))
        return CandidateErrors(
            possible_error.cpu().numpy(),
            impossible_error.cpu().numpy(),
            self.frame_spans,
        )

    def provenance(self) -> dict:
        return {
            "model": "DDLP ObjectDynamicsDLP",
            "pretraining_dataset": "OBJ3D-128",
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "config": str(CONFIG),
            "strict_state": self.state_report,
            "distance": "mean absolute error in decoded RGB [0,1], per original frame",
            "candidate_use": "post-prediction scoring only",
        }

