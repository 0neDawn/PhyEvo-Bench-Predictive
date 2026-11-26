"""Load the untouched official G-SWM code and run an 8-to-6 forecast.

The predictor receives only the shared eight-frame history.  Six blank tensor
slots are appended solely to tell the official recurrent generator how many
future steps to roll out; the blank slots are never encoded as observations.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
import sys
import types
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


DEPLOYMENT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = DEPLOYMENT_DIR.parents[3]
OFFICIAL_REPO = WORKSPACE_ROOT / "third_party/G-SWM"
OFFICIAL_SOURCE = OFFICIAL_REPO / "src"
OBJ3D_CONFIG = OFFICIAL_SOURCE / "configs/obj3d.yaml"
OBJ3D_CHECKPOINT = OFFICIAL_REPO / "pretrained/obj3d.pth"

HISTORY_STEPS = 8
FUTURE_STEPS = 6
NATIVE_SIZE = 64
CHECKPOINT_SHA256 = "83a14770985c11dd4c857eeaaf8981eac504b3b654fdec4cde06372b998ff716"


@dataclass(frozen=True)
class GSWMPrediction:
    """One candidate-independent prediction made from a shared history."""

    rgb: np.ndarray
    z_pres: np.ndarray
    z_depth: np.ndarray
    z_where: np.ndarray
    z_what: np.ndarray
    z_dyna: np.ndarray
    ids: np.ndarray
    seed: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {requested}")
    return device


def _install_unused_attrdict_compatibility_shim() -> None:
    """Satisfy an unused legacy import without changing official source code."""

    if "attrdict" not in sys.modules:
        module = types.ModuleType("attrdict")
        module.AttrDict = dict
        sys.modules["attrdict"] = module


def _configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _as_tchw(frames: np.ndarray | torch.Tensor, expected_steps: int) -> torch.Tensor:
    tensor = torch.as_tensor(frames)
    if tensor.ndim != 4:
        raise ValueError(f"Expected a four-dimensional frame array, got {tuple(tensor.shape)}")

    if tensor.shape[-1] in (1, 3, 4):
        tensor = tensor[..., :3].permute(0, 3, 1, 2)
    elif tensor.shape[1] in (1, 3, 4):
        tensor = tensor[:, :3]
    else:
        raise ValueError("Frames must use THWC or TCHW layout with RGB channels")

    if tensor.shape[0] != expected_steps:
        raise ValueError(f"Expected {expected_steps} frames, got {tensor.shape[0]}")
    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)

    original_dtype = tensor.dtype
    tensor = tensor.to(torch.float32)
    if original_dtype == torch.uint8:
        tensor = tensor / 255.0
    else:
        low = float(tensor.min())
        high = float(tensor.max())
        if low < 0:
            raise ValueError(f"Frame values must be non-negative, observed minimum {low}")
        if high > 1.0:
            if high <= 255.0:
                tensor = tensor / 255.0
            else:
                raise ValueError(f"Frame values exceed supported range: maximum {high}")

    if not torch.isfinite(tensor).all():
        raise ValueError("Frames contain NaN or infinity")
    if tuple(tensor.shape[-2:]) != (NATIVE_SIZE, NATIVE_SIZE):
        tensor = F.interpolate(tensor, size=(NATIVE_SIZE, NATIVE_SIZE), mode="nearest")
    return tensor.contiguous()


def seed_from_history(history: np.ndarray | torch.Tensor) -> int:
    """Derive an order-independent seed from the normalized shared history."""

    normalized = _as_tchw(history, HISTORY_STEPS).numpy()
    digest = hashlib.sha256(normalized.tobytes(order="C")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


class GSWMObj3D:
    """Strict loader and inference wrapper for the official OBJ3D checkpoint."""

    model_name = "G-SWM-OBJ3D"

    def __init__(self, device: str = "auto") -> None:
        self.device = resolve_device(device)
        self._validate_files()
        self.model, self.checkpoint_metadata = self._load_model()

    def _validate_files(self) -> None:
        for path in (OFFICIAL_SOURCE, OBJ3D_CONFIG, OBJ3D_CHECKPOINT):
            if not path.exists():
                raise FileNotFoundError(f"Required G-SWM artifact is missing: {path}")
        observed = sha256_file(OBJ3D_CHECKPOINT)
        if observed != CHECKPOINT_SHA256:
            raise RuntimeError(
                "OBJ3D checkpoint hash mismatch: "
                f"expected {CHECKPOINT_SHA256}, observed {observed}"
            )

    def _load_model(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        source_text = str(OFFICIAL_SOURCE)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        _install_unused_attrdict_compatibility_shim()

        from config import cfg
        from model import get_model

        cfg.merge_from_file(str(OBJ3D_CONFIG))
        model = get_model(cfg)
        try:
            checkpoint = torch.load(
                OBJ3D_CHECKPOINT,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(OBJ3D_CHECKPOINT, map_location="cpu")
        incompatible = model.load_state_dict(checkpoint["model"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"Strict checkpoint load failed: {incompatible}")
        model.eval().to(self.device)
        metadata = {
            "epoch": int(checkpoint["epoch"]),
            "global_step": int(checkpoint["global_step"]),
            "state_tensor_count": len(checkpoint["model"]),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        }
        return model, metadata

    def predict(
        self,
        history: np.ndarray | torch.Tensor,
        seed: int | None = None,
    ) -> GSWMPrediction:
        """Predict six future frames from exactly eight shared history frames."""

        history_tensor = _as_tchw(history, HISTORY_STEPS)
        if seed is None:
            seed = seed_from_history(history_tensor)
        seed = int(seed)
        if seed < 0:
            raise ValueError("Seed must be non-negative")
        _configure_reproducibility(seed)

        sequence = torch.zeros(
            1,
            HISTORY_STEPS + FUTURE_STEPS,
            3,
            NATIVE_SIZE,
            NATIVE_SIZE,
            dtype=torch.float32,
            device=self.device,
        )
        sequence[:, :HISTORY_STEPS] = history_tensor.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            log = self.model.generate(
                sequence,
                cond_steps=HISTORY_STEPS,
                fg_sample=False,
                bg_sample=False,
            )

        def future_numpy(name: str) -> np.ndarray:
            value = log[name][0, HISTORY_STEPS:]
            if not torch.isfinite(value).all():
                raise RuntimeError(f"Non-finite values in G-SWM output: {name}")
            return value.detach().cpu().numpy().copy()

        return GSWMPrediction(
            rgb=future_numpy("recon"),
            z_pres=future_numpy("z_pres"),
            z_depth=future_numpy("z_depth"),
            z_where=future_numpy("z_where"),
            z_what=future_numpy("z_what"),
            z_dyna=future_numpy("z_dyna"),
            ids=future_numpy("ids"),
            seed=seed,
        )

    @staticmethod
    def prepare_candidate(candidate_future: np.ndarray | torch.Tensor) -> np.ndarray:
        return _as_tchw(candidate_future, FUTURE_STEPS).numpy()

    @staticmethod
    def pixel_mse(
        prediction: GSWMPrediction,
        candidate_future: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """Return one native 64x64 RGB prediction error per future step."""

        candidate = GSWMObj3D.prepare_candidate(candidate_future)
        error = np.mean((prediction.rgb - candidate) ** 2, axis=(1, 2, 3))
        if error.shape != (FUTURE_STEPS,) or not np.isfinite(error).all():
            raise RuntimeError("Invalid per-step G-SWM pixel error")
        return error.astype(np.float64, copy=False)
