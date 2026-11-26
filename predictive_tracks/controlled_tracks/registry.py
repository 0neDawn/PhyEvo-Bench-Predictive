"""Nine frozen model adapters, with separate additive entry files."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "predictive_tracks/sp/code"
OUTPUT = ROOT / "predictive_tracks/controlled_tracks/runs"

MODELS = {
    "ocvp_seq": "sp_ocvp.track_orientation_ocvp",
    "savi_transformer": "sp_ocvp.track_orientation_transformer",
    "slotformer": "sp_core.adapters.track_orientation_slotformer",
    "ddlp": "sp_core.adapters.track_orientation_ddlp",
    "gswm": "sp_gswm.adapters.track_orientation",
    "vjepa2_vitg": "sp_protocol.adapters.track_orientation_vjepa2",
    "vjepa2_vitl": "sp_vjepa2_vitl.track_orientation",
    "videomaev2": "sp_videomae.track_orientation",
    "dino_foresight": "sp_dino_foresight.track_orientation",
}
HISTORY_MODELS = {key: module.replace("track_orientation", "track_history")
                  for key, module in MODELS.items()}
MOMENT_MODELS = {key: module.replace("track_orientation", "track_moment")
                for key, module in MODELS.items()}
CONTINUITY_MODELS = {key: module.replace("track_orientation", "track_continuity")
                    for key, module in MODELS.items()}
BOUNDARY_MODELS = {key: module.replace("track_orientation", "track_boundary")
                  for key, module in MODELS.items()}
ORTHOGONAL_MODELS = {key: module.replace("track_orientation", "track_orthogonal")
                    for key, module in MODELS.items()}
PARALLEL_MODELS = {key: module.replace("track_orientation", "track_parallel")
                  for key, module in MODELS.items()}
NATIVE_TOLERANCE = {key: (2 if key in ("vjepa2_vitg", "vjepa2_vitl", "videomaev2")
                         else 1) for key in MODELS}
QUALIFICATIONS = {
    "dino_foresight": "Known baseline masked-position/rollout alignment issue; diagnostic only, never count as validated method improvement.",
    "gswm": "Native history-hash-derived RNG changes under reflection; orientation and posterior-sample effects are confounded. MC sample count remains eight.",
}


def qualification(model, correction="none"):
    if model == "gswm" and correction != "none":
        return "Original history pixels/hash and all eight native posterior sample seeds unchanged; output correction does not introduce the input-view RNG confound."
    return QUALIFICATIONS.get(model)


def archive(track):
    return ROOT / "predictive_tracks" / track / "finnal_result_with_dino_foresight"
