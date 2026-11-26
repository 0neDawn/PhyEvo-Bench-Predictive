"""VideoMAE-V2 evaluated in its official trained output space.

The checkpoint's decoder head predicts normalized RGB patch vectors.  This is
the representation on which VideoMAE-V2 was pretrained, so it is the most
literal interpretation of the paper's model-native prediction space.  The
implementation, temporal geometry, and checkpoint loading are inherited
unchanged from the existing official 16-frame adapter.
"""

from __future__ import annotations

from sp_protocol.adapters.videomaev2_official16 import Adapter as Official16Adapter


class Adapter(Official16Adapter):
    model_name = "videomaev2_official_normalized_patch_output"

    def provenance(self) -> dict:
        value = super().provenance()
        value.update(
            {
                "evaluation_variant": (
                    "official trained decoder-head output under checkpoint-native "
                    "16-frame temporal geometry"
                ),
                "native_output_basis": (
                    "decoder.head is the checkpoint's trained prediction output; "
                    "no intermediate hidden representation is substituted"
                ),
                "distance": (
                    "unweighted MSE between predicted and target normalized RGB "
                    "patch vectors, per two-frame tubelet"
                ),
                "additional_prediction_head": False,
                "category_specific_logic": False,
                "score_weight_or_threshold_fitted": False,
                "outer_metrics_changed": False,
                "selection_timing": (
                    "additive post-hoc protocol comparison after prior benchmark "
                    "results were known"
                ),
                "reporting_status": (
                    "architecture-defined native-output sensitivity; confirmatory "
                    "claim requires a previously untouched holdout"
                ),
            }
        )
        return value
