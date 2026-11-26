"""SAVi + Transformer adapter using the official frozen OBJ3D checkpoints.

This is an additive adapter.  It reuses the already audited OCVP preprocessing,
rollout, decoded-RGB scoring, and provenance implementation without modifying
the existing OCVP-Seq or SAVi+LSTM adapters.
"""

from __future__ import annotations

import copy

import torch

from sp_core.common import load_strict, sha256

from .common import (
    FORECAST_FRAMES,
    NATIVE_CONTEXT,
    OFFICIAL_EXPERIMENT,
    SAVI_CHECKPOINT,
    SAVI_CHECKPOINT_SHA256,
    SAVI_CONFIG,
    OfficialOBJ3DAdapter,
    _load_json,
    _prepare_official_imports,
)


TRANSFORMER_CONFIG = (
    OFFICIAL_EXPERIMENT / "Predictor_Transformer/experiment_params.json"
)
TRANSFORMER_CHECKPOINT = (
    OFFICIAL_EXPERIMENT / "Predictor_Transformer/models/transformer_obj3d.pth"
)
TRANSFORMER_CHECKPOINT_SHA256 = (
    "729e8dcd313b159f24fecfe7907f34e46f352dcb1b65ccba8fd30aa1f45e80d7"
)


class Adapter(OfficialOBJ3DAdapter):
    """Official SAVi tokenizer/decoder with the official Transformer predictor."""

    model_name = "savi_transformer_obj3d"

    def __init__(self, device: str, memory_mode: str = "safe") -> None:
        del memory_mode
        self.definition = {
            "display_name": "SAVi + Transformer (OBJ3D)",
            "official_name": "Transformer",
            "config": TRANSFORMER_CONFIG,
            "checkpoint": TRANSFORMER_CHECKPOINT,
            "checkpoint_sha256": TRANSFORMER_CHECKPOINT_SHA256,
            "model_name": self.model_name,
        }
        self.predictor_key = "savi_transformer"
        self.device = torch.device(device)
        self.prediction_calls = 0

        for path in (
            SAVI_CONFIG,
            SAVI_CHECKPOINT,
            TRANSFORMER_CONFIG,
            TRANSFORMER_CHECKPOINT,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        if sha256(SAVI_CHECKPOINT) != SAVI_CHECKPOINT_SHA256:
            raise RuntimeError("Official OBJ3D SAVi checkpoint hash mismatch")
        if sha256(TRANSFORMER_CHECKPOINT) != TRANSFORMER_CHECKPOINT_SHA256:
            raise RuntimeError("Official OBJ3D Transformer checkpoint hash mismatch")

        _prepare_official_imports()
        from models.Predictors import (  # type: ignore[import-not-found]
            PredictorWrapper,
            VanillaTransformerPredictor,
        )
        from models.SAVi import SAVi  # type: ignore[import-not-found]

        trained = _load_json(TRANSFORMER_CONFIG)
        self._validate_official_config(trained)
        transformer_params = trained["model"]["predictor"]["Transformer"]
        if transformer_params["input_buffer_size"] != NATIVE_CONTEXT:
            raise RuntimeError("Official Transformer input buffer is not five frames")

        inference = copy.deepcopy(trained)
        inference["training_prediction"]["num_preds"] = FORECAST_FRAMES
        inference["training_prediction"]["sample_length"] = (
            NATIVE_CONTEXT + FORECAST_FRAMES
        )

        model_params = inference["model"]["SAVi"]
        self.savi = SAVi(**model_params)
        savi_payload = torch.load(
            SAVI_CHECKPOINT, map_location="cpu", weights_only=False
        )
        self.savi_epoch = int(savi_payload["epoch"])
        self.savi_report = load_strict(
            self.savi, savi_payload["model_state_dict"]
        )

        raw_predictor = VanillaTransformerPredictor(
            num_slots=model_params["num_slots"],
            slot_dim=model_params["slot_dim"],
            num_imgs=NATIVE_CONTEXT + FORECAST_FRAMES,
            token_dim=transformer_params["token_dim"],
            hidden_dim=transformer_params["hidden_dim"],
            num_layers=transformer_params["num_layers"],
            n_heads=transformer_params["n_heads"],
            residual=transformer_params.get("residual", False),
            input_buffer_size=transformer_params.get(
                "input_buffer_size", NATIVE_CONTEXT
            ),
        )
        self.predictor = PredictorWrapper(inference, raw_predictor)
        predictor_payload = torch.load(
            TRANSFORMER_CHECKPOINT, map_location="cpu", weights_only=False
        )
        self.predictor_epoch = int(predictor_payload["epoch"])
        predictor_state = predictor_payload["model_state_dict"]
        if not all(key.startswith("predictor.") for key in predictor_state):
            predictor_state = {
                f"predictor.{key}": value for key, value in predictor_state.items()
            }
        self.predictor_report = load_strict(self.predictor, predictor_state)

        self.savi.to(self.device).eval().requires_grad_(False)
        self.predictor.to(self.device).eval().requires_grad_(False)
        self.num_slots = int(model_params["num_slots"])
        self.slot_dim = int(model_params["slot_dim"])

