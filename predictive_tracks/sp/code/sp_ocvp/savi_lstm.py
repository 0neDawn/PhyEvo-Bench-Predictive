"""SAVi+LSTM adapter using the official OBJ3D SAVi and LSTM checkpoints."""

from __future__ import annotations

from .common import OfficialOBJ3DAdapter


class Adapter(OfficialOBJ3DAdapter):
    model_name = "savi_lstm_obj3d"

    def __init__(self, device: str, memory_mode: str = "safe") -> None:
        super().__init__("savi_lstm", device=device, memory_mode=memory_mode)

