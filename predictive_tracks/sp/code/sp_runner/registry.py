"""Single source of truth for the currently selected SP model adapters."""

from __future__ import annotations

from release_paths import runtime_python as _release_python, project_path as _release_path

from release_paths import project_path as _release_path

from dataclasses import dataclass
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SP_ROOT = CODE_ROOT.parent
ROOT = SP_ROOT.parents[1]
JEPA_PYTHON = Path(_release_python('other'))
DDLP_PYTHON = Path(_release_python('ddlp'))
PAIR_MANIFEST = CODE_ROOT / "sp_benchmark/manifests/pairs_8to6.csv"
PAIR_MANIFEST_SHA256 = (
    "3d8efd88d767fc080bd92ca983924925b277f7f853fe93f9778ee9e7bbb8f42a"
)


@dataclass(frozen=True)
class Checkpoint:
    role: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    python: Path
    module: str
    model_arg: str
    output_name: str
    current_run: Path
    expected_steps: int
    expected_frame_spans: tuple[tuple[int, int], ...]
    native_distance: str
    reporting_status: str
    checkpoints: tuple[Checkpoint, ...]
    code_files: tuple[Path, ...]


FRAMEWISE = tuple((frame, frame) for frame in range(20, 26))
TUBELETWISE = ((20, 21), (22, 23), (24, 25))


SPECS = (
    ModelSpec(
        key="ddlp_obj3d",
        display_name="DDLP (OBJ3D-128)",
        python=DDLP_PYTHON,
        module="sp_runner.run_one",
        model_arg="ddlp_obj3d",
        output_name="ddlp",
        current_run=SP_ROOT / "runs/current/ddlp",
        expected_steps=6,
        expected_frame_spans=FRAMEWISE,
        native_distance="decoded RGB [0,1] mean absolute error",
        reporting_status="frozen formal adapter",
        checkpoints=(
            Checkpoint(
                "DDLP OBJ3D-128",
                ROOT / "ddlp/checkpoints/diffuse-ddlp-obj3d128/obj3d128_ddlp.pth",
                "4f5b416c44d6288e03656e899a00ae0370cfab8f0aa7d16a8cb794057066283d",
            ),
        ),
        code_files=(CODE_ROOT / "sp_core/adapters/ddlp_obj3d.py",),
    ),
    ModelSpec(
        key="slotformer_obj3d",
        display_name="SlotFormer + SAVi (OBJ3D)",
        python=JEPA_PYTHON,
        module="sp_runner.run_one",
        model_arg="slotformer_obj3d",
        output_name="slotformer",
        current_run=SP_ROOT / "runs/current/slotformer",
        expected_steps=6,
        expected_frame_spans=FRAMEWISE,
        native_distance="slot-wise mean squared error",
        reporting_status="frozen formal adapter",
        checkpoints=(
            Checkpoint(
                "SAVi OBJ3D",
                ROOT / "third_party/SlotFormer/pretrained/savi_obj3d_params/model_40.pth",
                "c1e791add3c8db5cddc86f9a861cc852966a11d3b2e72d5c1605622f36b9f52b",
            ),
            Checkpoint(
                "SlotFormer OBJ3D",
                ROOT / "third_party/SlotFormer/pretrained/slotformer_obj3d_params/model_200.pth",
                "52be93dd71a1faae7d310983d88aea971c3899221fb59d32e2d5ced0e1b77f3a",
            ),
        ),
        code_files=(CODE_ROOT / "sp_core/adapters/slotformer_obj3d.py",),
    ),
    ModelSpec(
        key="vjepa2",
        display_name="V-JEPA 2 (ViT-g/16 FPC64)",
        python=JEPA_PYTHON,
        module="sp_runner.run_one",
        model_arg="vjepa2_corrected",
        output_name="vjepa2",
        current_run=SP_ROOT / "runs/current/vjepa2",
        expected_steps=3,
        expected_frame_spans=TUBELETWISE,
        native_distance="layer-normalized target-encoder latent L1",
        reporting_status="official-preprocessing corrected adapter",
        checkpoints=(
            Checkpoint(
                "V-JEPA 2 ViT-g/16 FPC64",
                Path(
                    _release_path('third_party/vjepa2/vjepa2-vitg-fpc64-256/original/model.pth')
                ),
                "67129f011434e605d894e69f2c8e13d9db118deabe59d54bf6e0fa62c2c5cb8e",
            ),
        ),
        code_files=(
            CODE_ROOT
            / "sp_protocol/adapters/"
            "vjepa2_official_preprocess.py",
        ),
    ),
    ModelSpec(
        key="gswm_obj3d",
        display_name="G-SWM (OBJ3D, predicted-foreground robust)",
        python=JEPA_PYTHON,
        module="sp_runner.run_one",
        model_arg="gswm_foreground_robust",
        output_name="gswm",
        current_run=(
            SP_ROOT / "runs/current/gswm"
        ),
        expected_steps=6,
        expected_frame_spans=FRAMEWISE,
        native_distance="predicted-alpha-weighted RGB MSE with empty-alpha fallback",
        reporting_status="post-hoc native-space sensitivity adapter",
        checkpoints=(
            Checkpoint(
                "G-SWM OBJ3D",
                ROOT / "third_party/G-SWM/pretrained/obj3d.pth",
                "83a14770985c11dd4c857eeaaf8981eac504b3b654fdec4cde06372b998ff716",
            ),
        ),
        code_files=(
            CODE_ROOT
            / "sp_gswm/adapters/"
            "gswm_foreground_predictive_mean.py",
            CODE_ROOT
            / "sp_gswm/adapters/"
            "gswm_foreground_predictive_mean_robust.py",
        ),
    ),
    ModelSpec(
        key="videomaev2",
        display_name="VideoMAE-V2 (decoder-native state)",
        python=JEPA_PYTHON,
        module="sp_runner.run_one",
        model_arg="videomaev2_decoder_native",
        output_name="videomaev2",
        current_run=(
            SP_ROOT / "runs/current/videomaev2"
        ),
        expected_steps=3,
        expected_frame_spans=TUBELETWISE,
        native_distance="layer-normalized official decoder hidden-state MSE",
        reporting_status="post-hoc native-space sensitivity adapter",
        checkpoints=(
            Checkpoint(
                "VideoMAE-V2 ViT-g/14 hybrid 1200e",
                ROOT / "VideoMAEv2/checkpoints/vit_g_hybrid_pt_1200e.pth",
                "5773c881d303426f2f0e906df9842c4ca95fba6a5901849a3e8d9d1ed8f79b5f",
            ),
        ),
        code_files=(
            CODE_ROOT / "sp_videomae/adapter.py",
        ),
    ),
)


BY_KEY = {spec.key: spec for spec in SPECS}
if len(BY_KEY) != len(SPECS):
    raise RuntimeError("Duplicate SP model registry key")
