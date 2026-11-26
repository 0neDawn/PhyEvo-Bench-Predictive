"""Official OBJ3D OCVP adapters shared by OCVP-Seq and SAVi+LSTM.

The official predictor checkpoints were trained with five context frames and
five forecast frames.  At evaluation time the frozen SAVi tokenizer causally
encodes all eight benchmark history frames; the predictor consumes the final
five slot states and uses the repository's supported autoregressive rollout
override to produce six forecast frames.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
from PIL import Image
import torch

from sp_benchmark.adapter_contract import CandidateErrors
from sp_core.common import ROOT, load_strict, sha256


OCVP_ROOT = ROOT / "third_party/OCVP"
OCVP_SOURCE = OCVP_ROOT / "src"
OFFICIAL_EXPERIMENT = OCVP_ROOT / "pretrained/obj3d/Obj3D"
SAVI_CONFIG = OFFICIAL_EXPERIMENT / "experiment_params.json"
SAVI_CHECKPOINT = OFFICIAL_EXPERIMENT / "models/savi_obj3d.pth"
OFFICIAL_REVISION = "01d0d57adc48efca1e57685a14c96a43cac4be5e"

PREDICTORS = {
    "ocvp_seq": {
        "display_name": "OCVP-Seq (OBJ3D)",
        "official_name": "OCVP-Seq",
        "config": OFFICIAL_EXPERIMENT / "Predictor_OCVPSeq/experiment_params.json",
        "checkpoint": (
            OFFICIAL_EXPERIMENT
            / "Predictor_OCVPSeq/models/OCVPSeq_obj3d.pth"
        ),
        "checkpoint_sha256": (
            "4052ae79fd2808e99720d7da0622594b39598af1e8670f0476dbb4cb8572e5b9"
        ),
        "model_name": "ocvp_seq_obj3d",
    },
    "savi_lstm": {
        "display_name": "SAVi + LSTM (OBJ3D)",
        "official_name": "LSTM",
        "config": OFFICIAL_EXPERIMENT / "Predictor_LSTM/experiment_params.json",
        "checkpoint": OFFICIAL_EXPERIMENT / "Predictor_LSTM/models/lstm_obj3d.pth",
        "checkpoint_sha256": (
            "2e1197dbf191000697649b4ca5fc5750a7d386e293e970c1428d630ea3943b27"
        ),
        "model_name": "savi_lstm_obj3d",
    },
}

SAVI_CHECKPOINT_SHA256 = (
    "985a39306cc1843f779a5b261f9fd74f44a7ae20e7dea49454f96f374d0bb321"
)
FRAME_SPANS = np.asarray([[frame, frame] for frame in range(20, 26)])
HISTORY_FRAMES = 8
NATIVE_CONTEXT = 5
FORECAST_FRAMES = 6
INITIALIZER_SEED_BASE = 140000


def _profiling_dependency_stub() -> None:
    """Provide only the unused fvcore profiling names when fvcore is absent.

    OCVP imports these names at module import time, but inference never calls
    the model-summary helper that uses them.  Keeping the compatibility shim
    here avoids modifying the vendored official source.
    """

    if "fvcore" in sys.modules or importlib.util.find_spec("fvcore") is not None:
        return

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("fvcore is available only for optional model profiling")

    package = types.ModuleType("fvcore")
    nn_module = types.ModuleType("fvcore.nn")
    nn_module.FlopCountAnalysis = unavailable
    nn_module.ActivationCountAnalysis = unavailable
    nn_module.flop_count_str = unavailable
    package.nn = nn_module
    sys.modules["fvcore"] = package
    sys.modules["fvcore.nn"] = nn_module


def _prepare_official_imports() -> None:
    source = str(OCVP_SOURCE)
    if source not in sys.path:
        sys.path.insert(0, source)
    for module_name in ("models", "lib", "CONFIG"):
        module = sys.modules.get(module_name)
        module_file = getattr(module, "__file__", None) if module is not None else None
        if module_file is not None and not Path(module_file).resolve().is_relative_to(
            OCVP_SOURCE.resolve()
        ):
            raise RuntimeError(
                f"Top-level module {module_name!r} was loaded from another project: "
                f"{module_file}"
            )
    _profiling_dependency_stub()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _official_rgb_tensor(frames: np.ndarray) -> torch.Tensor:
    """Apply the official OBJ3D PIL resize and ToTensor-equivalent scaling."""

    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected RGB video [T,H,W,3], got {frames.shape}")
    resized = []
    for frame in frames:
        # The official OBJ3D loader calls RGB PIL.Image.resize((64, 64)) with
        # no resample argument, followed by torchvision.transforms.ToTensor().
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).resize((64, 64))
        resized.append(np.asarray(image, dtype=np.uint8).copy())
    value = torch.from_numpy(np.stack(resized)).permute(0, 3, 1, 2)
    return value.float().div_(255.0).unsqueeze(0)


class OfficialOBJ3DAdapter:
    """One-prediction adapter using official frozen SAVi and predictor weights."""

    frame_spans = FRAME_SPANS

    def __init__(
        self,
        predictor_key: str,
        device: str,
        memory_mode: str = "safe",
    ) -> None:
        del memory_mode
        if predictor_key not in PREDICTORS:
            raise ValueError(f"Unknown OCVP predictor: {predictor_key}")
        self.definition = PREDICTORS[predictor_key]
        self.model_name = self.definition["model_name"]
        self.predictor_key = predictor_key
        self.device = torch.device(device)
        self.prediction_calls = 0

        for path in (
            SAVI_CONFIG,
            SAVI_CHECKPOINT,
            self.definition["config"],
            self.definition["checkpoint"],
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        if sha256(SAVI_CHECKPOINT) != SAVI_CHECKPOINT_SHA256:
            raise RuntimeError("Official OBJ3D SAVi checkpoint hash mismatch")
        if sha256(self.definition["checkpoint"]) != self.definition["checkpoint_sha256"]:
            raise RuntimeError(
                f"Official {self.definition['display_name']} checkpoint hash mismatch"
            )

        _prepare_official_imports()
        from models.Predictors import (  # type: ignore[import-not-found]
            LSTMPredictor,
            OCVPSeq,
            PredictorWrapper,
        )
        from models.SAVi import SAVi  # type: ignore[import-not-found]

        trained = _load_json(self.definition["config"])
        self._validate_official_config(trained)
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

        predictor_params = inference["model"]["predictor"]
        official_name = self.definition["official_name"]
        current = predictor_params[official_name]
        if official_name == "OCVP-Seq":
            raw_predictor = OCVPSeq(
                num_slots=model_params["num_slots"],
                slot_dim=model_params["slot_dim"],
                num_imgs=NATIVE_CONTEXT + FORECAST_FRAMES,
                token_dim=current["token_dim"],
                hidden_dim=current["hidden_dim"],
                num_layers=current["num_layers"],
                n_heads=current["n_heads"],
                residual=current.get("residual", False),
                input_buffer_size=current.get("input_buffer_size", NATIVE_CONTEXT),
            )
        elif official_name == "LSTM":
            raw_predictor = LSTMPredictor(
                slot_dim=model_params["slot_dim"],
                hidden_dim=current["hidden_dim"],
                num_layers=current["num_cells"],
                residual=current.get("residual", True),
            )
        else:  # pragma: no cover - guarded by the frozen registry above
            raise AssertionError(official_name)
        self.predictor = PredictorWrapper(inference, raw_predictor)
        predictor_payload = torch.load(
            self.definition["checkpoint"], map_location="cpu", weights_only=False
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

    def _validate_official_config(self, config: dict) -> None:
        training = config["training_prediction"]
        model = config["model"]
        savi = model["SAVi"]
        checks = {
            "dataset": config["dataset"]["dataset_name"] == "OBJ3D",
            "predictor": (
                model["predictor"]["predictor_name"]
                == self.definition["official_name"]
            ),
            "trained_context": training["num_context"] == NATIVE_CONTEXT,
            "trained_predictions": training["num_preds"] == 5,
            "trained_sample_length": training["sample_length"] == 10,
            "no_teacher_force": training["teacher_force"] is False,
            "slots": savi["num_slots"] == 6,
            "slot_dim": savi["slot_dim"] == 128,
            "resolution": savi["resolution"] == [64, 64],
            "initializer": savi["initializer"] == "LearnedRandom",
        }
        if self.predictor_key == "ocvp_seq":
            checks["input_buffer"] = (
                model["predictor"]["OCVP-Seq"]["input_buffer_size"]
                == NATIVE_CONTEXT
            )
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(f"Official OCVP configuration mismatch: {failed}")

    def _predict_decoded_rgb(self, history: torch.Tensor) -> torch.Tensor:
        if history.shape != (1, HISTORY_FRAMES, 3, 64, 64):
            raise ValueError(f"Unexpected history tensor shape: {tuple(history.shape)}")

        history_slots = []
        next_slots = self.savi.initializer(batch_size=1)
        for step in range(HISTORY_FRAMES):
            features = self.savi.encode(history[:, step])
            slots = self.savi.apply_attention(
                features, predicted_slots=next_slots, step=step
            )
            next_slots = self.savi.predictor(slots)
            history_slots.append(slots)
        history_slots = torch.stack(history_slots, dim=1)

        predicted_slots = self.predictor(history_slots[:, -NATIVE_CONTEXT:])
        expected = (1, FORECAST_FRAMES, self.num_slots, self.slot_dim)
        if tuple(predicted_slots.shape) != expected:
            raise RuntimeError(
                f"Unexpected predictor output {tuple(predicted_slots.shape)} != {expected}"
            )
        flattened = predicted_slots.reshape(
            FORECAST_FRAMES, self.num_slots, self.slot_dim
        )
        decoded = self.savi.decode(flattened)[0]
        decoded = decoded.reshape(1, FORECAST_FRAMES, 3, 64, 64).clamp(0.0, 1.0)
        if not torch.isfinite(decoded).all():
            raise RuntimeError("OCVP produced non-finite decoded predictions")
        return decoded

    def score_pair(
        self,
        possible_rgb: np.ndarray,
        impossible_rgb: np.ndarray,
    ) -> CandidateErrors:
        if possible_rgb.shape != impossible_rgb.shape:
            raise ValueError("Candidate videos have different shapes")
        if possible_rgb.shape[0] != HISTORY_FRAMES + FORECAST_FRAMES:
            raise ValueError(f"Expected 14 benchmark frames, got {possible_rgb.shape[0]}")
        if not np.array_equal(
            possible_rgb[:HISTORY_FRAMES], impossible_rgb[:HISTORY_FRAMES]
        ):
            raise RuntimeError("Candidate histories are not identical")

        # No candidate-future tensor exists yet when the sole forecast is made.
        history = _official_rgb_tensor(possible_rgb[:HISTORY_FRAMES]).to(self.device)
        cuda_devices: list[int] = []
        if self.device.type == "cuda":
            cuda_devices = [
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            ]
        seed = INITIALIZER_SEED_BASE + self.prediction_calls
        with torch.inference_mode(), torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(seed)
            prediction = self._predict_decoded_rgb(history)
        self.prediction_calls += 1

        # Candidate futures are constructed only after prediction and are used
        # exclusively by the symmetric distance calculation below.
        possible = _official_rgb_tensor(possible_rgb[HISTORY_FRAMES:]).to(self.device)
        impossible = _official_rgb_tensor(impossible_rgb[HISTORY_FRAMES:]).to(
            self.device
        )
        with torch.inference_mode():
            possible_error = (prediction - possible).square().mean(dim=(0, 2, 3, 4))
            impossible_error = (prediction - impossible).square().mean(
                dim=(0, 2, 3, 4)
            )
        if not torch.isfinite(possible_error).all() or not torch.isfinite(
            impossible_error
        ).all():
            raise RuntimeError("OCVP candidate errors are non-finite")
        return CandidateErrors(
            possible_error.cpu().numpy(),
            impossible_error.cpu().numpy(),
            self.frame_spans,
        )

    def provenance(self) -> dict:
        predictor_checkpoint = self.definition["checkpoint"]
        return {
            "model": self.definition["display_name"],
            "pretraining_dataset": "OBJ3D for frozen SAVi and frozen predictor",
            "official_repository": (
                "https://github.com/AIS-Bonn/OCVP-object-centric-video-prediction"
            ),
            "official_source_revision": OFFICIAL_REVISION,
            "official_repository_license_note": (
                "No LICENSE/COPYING file was present at the pinned revision; "
                "recorded as source-available, not as a licensed open-source release."
            ),
            "official_training_context": 5,
            "official_training_predictions": 5,
            "benchmark_history": (
                "frozen SAVi causally encodes shared frames 12-19; the predictor "
                "consumes the final five slot states corresponding to frames 15-19"
            ),
            "benchmark_predictions": (
                "six autoregressive frames via the official evaluator's supported "
                "num_preds override"
            ),
            "preprocessing": (
                "official OBJ3D RGB PIL resize((64,64)) followed by ToTensor-equivalent "
                "[0,1] scaling"
            ),
            "distance": (
                "per-frame mean squared error in frozen SAVi decoder RGB [0,1] space"
            ),
            "frame_spans": self.frame_spans.tolist(),
            "candidate_use": "post-prediction decoded-RGB scoring only",
            "labels_available_to_predictor": False,
            "initializer": (
                "official LearnedRandom SAVi initializer with deterministic seed "
                f"{INITIALIZER_SEED_BASE} + zero-based pair index"
            ),
            "savi_config": str(SAVI_CONFIG),
            "savi_checkpoint": str(SAVI_CHECKPOINT),
            "savi_checkpoint_bytes": SAVI_CHECKPOINT.stat().st_size,
            "savi_checkpoint_sha256": SAVI_CHECKPOINT_SHA256,
            "savi_checkpoint_epoch": self.savi_epoch,
            "predictor_config": str(self.definition["config"]),
            "predictor_checkpoint": str(predictor_checkpoint),
            "predictor_checkpoint_bytes": predictor_checkpoint.stat().st_size,
            "predictor_checkpoint_sha256": self.definition["checkpoint_sha256"],
            "predictor_checkpoint_epoch": self.predictor_epoch,
            "strict_states": {
                "savi": self.savi_report,
                "predictor": self.predictor_report,
            },
            "compatibility": (
                "unused fvcore profiling imports are stubbed when fvcore is absent; "
                "official inference modules and checkpoint parameters are unchanged"
            ),
        }
