"""Registry-shaped metadata for additive ViT-L run and audit tools."""

from sp_runner.registry import (
    CODE_ROOT,
    JEPA_PYTHON,
    SP_ROOT,
    Checkpoint,
    ModelSpec,
    TUBELETWISE,
)


CHECKPOINT_PATH = (
    SP_ROOT / "checkpoints/vjepa2-vitl-fpc64-256/original/model.pth"
)
CHECKPOINT_SHA256 = "5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012"

VITL_SPEC = ModelSpec(
    key="vjepa2_vitl",
    display_name="V-JEPA 2 (ViT-L/16 FPC64)",
    python=JEPA_PYTHON,
    module="predictive_tracks.sp.run_vjepa2_vitl",
    model_arg="vjepa2_vitl_official",
    output_name="vjepa2_vitl",
    current_run=SP_ROOT / "runs/current/vjepa2_vitl",
    expected_steps=3,
    expected_frame_spans=TUBELETWISE,
    native_distance="layer-normalized target-encoder latent L1",
    reporting_status="additive ViT-L evaluation with official preprocessing",
    checkpoints=(
        Checkpoint(
            "V-JEPA 2 ViT-L/16 FPC64",
            CHECKPOINT_PATH,
            CHECKPOINT_SHA256,
        ),
    ),
    code_files=(CODE_ROOT / "sp_vjepa2_vitl/adapter.py",),
)
