"""OCVP-Seq adapter using the official OBJ3D SAVi and predictor checkpoints."""

from __future__ import annotations

from .common import OfficialOBJ3DAdapter


class Adapter(OfficialOBJ3DAdapter):
    model_name = "ocvp_seq_obj3d"

    def __init__(self, device: str, memory_mode: str = "safe") -> None:
        super().__init__("ocvp_seq", device=device, memory_mode=memory_mode)

