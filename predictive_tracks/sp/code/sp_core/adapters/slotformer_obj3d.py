"""SlotFormer OBJ3D adapter with its required OBJ3D SAVi tokenizer."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import torch

from sp_benchmark.adapter_contract import CandidateErrors
from sp_core.common import ROOT, load_strict, video_tensor


SLOTFORMER = ROOT / "third_party/SlotFormer"
NERV = ROOT / "third_party/nerv"
DEPS = ROOT / "third_party/slotformer_python_deps"
SAVI_CONFIG = SLOTFORMER / "slotformer/base_slots/configs/savi_obj3d_params.py"
DYNAMICS_CONFIG = SLOTFORMER / (
    "slotformer/video_prediction/configs/slotformer_obj3d_params.py"
)
SAVI_CHECKPOINT = SLOTFORMER / "pretrained/savi_obj3d_params/model_40.pth"
DYNAMICS_CHECKPOINT = SLOTFORMER / "pretrained/slotformer_obj3d_params/model_200.pth"


def _import_config(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(path)
    spec.loader.exec_module(module)
    return module.SlotFormerParams()


def _prepare_imports() -> None:
    sys.path[:0] = [str(DEPS), str(NERV), str(SLOTFORMER)]
    # Avoid eager imports of optional PHYRE dependencies that are unrelated to OBJ3D.
    for package, path in (
        ("slotformer.base_slots", SLOTFORMER / "slotformer/base_slots"),
        ("slotformer.video_prediction", SLOTFORMER / "slotformer/video_prediction"),
    ):
        if package not in sys.modules:
            module = types.ModuleType(package)
            module.__path__ = [str(path)]
            sys.modules[package] = module


class Adapter:
    model_name = "slotformer_obj3d"
    frame_spans = np.asarray([[frame, frame] for frame in range(20, 26)])

    def __init__(self, device: str, memory_mode: str = "safe") -> None:
        del memory_mode
        self.device = torch.device(device)
        self.prediction_calls = 0
        _prepare_imports()
        from slotformer.base_slots.models import build_model as build_savi
        from slotformer.video_prediction.models import build_model as build_dynamics

        savi_params = _import_config(SAVI_CONFIG, "sp_pretest_savi_obj3d")
        dynamics_params = _import_config(DYNAMICS_CONFIG, "sp_pretest_slotformer_obj3d")
        if savi_params.dataset != "obj3d" or dynamics_params.dataset != "obj3d_slots":
            raise RuntimeError("SlotFormer or SAVi config is not OBJ3D")
        dynamics_params.dec_dict = copy.deepcopy(dynamics_params.dec_dict)
        dynamics_params.dec_dict["dec_ckp_path"] = str(SAVI_CHECKPOINT)

        self.savi = build_savi(savi_params)
        savi_state = torch.load(
            SAVI_CHECKPOINT, map_location="cpu", weights_only=False
        )["state_dict"]
        self.savi_report = load_strict(self.savi, savi_state)

        self.dynamics = build_dynamics(dynamics_params)
        dynamics_state = torch.load(
            DYNAMICS_CHECKPOINT, map_location="cpu", weights_only=False
        )["state_dict"]
        self.dynamics_report = load_strict(self.dynamics, dynamics_state)

        self.savi.to(self.device).eval().requires_grad_(False)
        self.dynamics.to(self.device).eval().requires_grad_(False)
        self.savi.testing = True

    def score_pair(self, possible_rgb: np.ndarray, impossible_rgb: np.ndarray) -> CandidateErrors:
        possible = video_tensor(
            possible_rgb, 64, minus_one_to_one=True
        ).to(self.device)
        impossible = video_tensor(
            impossible_rgb, 64, minus_one_to_one=True
        ).to(self.device)
        with torch.inference_mode():
            # The sole prediction path sees only the shared eight-frame history.
            history_slots = self.savi({"img": possible[:, :8]})["post_slots"]
            prediction = self.dynamics.rollout(
                history_slots, pred_len=6, decode=False
            )
        self.prediction_calls += 1

        # Evaluation-only target encodings happen after prediction. SAVi is recurrent,
        # so a target slot at frame t depends only on frames through t, never later ones.
        with torch.inference_mode():
            possible_all = self.savi({"img": possible})["post_slots"]
            impossible_all = self.savi({"img": impossible})["post_slots"]
        if not torch.equal(possible_all[:, :8], impossible_all[:, :8]):
            difference = (possible_all[:, :8] - impossible_all[:, :8]).abs().max().item()
            if difference > 1e-6:
                raise RuntimeError(f"SAVi shared-history slots differ: {difference}")
        targets = (possible_all[:, 8:], impossible_all[:, 8:])
        errors = [
            (prediction - target).square().mean(dim=(0, 2, 3))
            for target in targets
        ]
        if not all(torch.isfinite(item).all() for item in (prediction, *errors)):
            raise RuntimeError("SlotFormer produced non-finite values")
        return CandidateErrors(
            errors[0].cpu().numpy(), errors[1].cpu().numpy(), self.frame_spans
        )

    def provenance(self) -> dict:
        return {
            "model": "SlotFormer with SAVi",
            "pretraining_dataset": "OBJ3D for both SAVi and SlotFormer",
            "official_source_revision": "10a6d9b7bc05ef2397ce2586da3fc0212f8c47aa",
            "savi_checkpoint": str(SAVI_CHECKPOINT),
            "savi_checkpoint_sha256": "c1e791add3c8db5cddc86f9a861cc852966a11d3b2e72d5c1605622f36b9f52b",
            "slotformer_checkpoint": str(DYNAMICS_CHECKPOINT),
            "slotformer_checkpoint_sha256": "52be93dd71a1faae7d310983d88aea971c3899221fb59d32e2d5ced0e1b77f3a",
            "savi_strict_state": self.savi_report,
            "slotformer_strict_state": self.dynamics_report,
            "distance": "slot-wise mean squared error, per original frame",
            "candidate_use": "post-prediction recurrent SAVi encoding only",
        }

