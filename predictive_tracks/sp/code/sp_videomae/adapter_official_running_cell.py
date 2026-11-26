"""Official-output VideoMAE-V2 with its pretrained running-cell decoder mask.

The checkpoint was trained with the repository's 2x2 running-cell decoder mask
at ratio 0.5.  There are exactly four mask phases.  This adapter evaluates all
four phases and averages the two predictions available for every future token,
yielding one deterministic full-grid prediction.  Phase count, masks, and
uniform aggregation are architecture-defined; no score-dependent choice exists.
"""

from __future__ import annotations

import numpy as np
import torch

from sp_benchmark.adapter_contract import CandidateErrors
from sp_protocol.adapters.videomaev2_official16 import (
    CONTEXT_TUBELETS,
    SCORED_FUTURE_TUBELETS,
    TOKENS_PER_TUBELET,
    TOTAL_TUBELETS,
)
from sp_videomae.adapter_official_output import Adapter as OfficialOutputAdapter


class Adapter(OfficialOutputAdapter):
    model_name = "videomaev2_official_output_running_cell_expectation"

    def __init__(self, device: str, memory_mode: str = "safe") -> None:
        super().__init__(device=device, memory_mode=memory_mode)
        from dataset.masking_generator import RunningCellMaskingGenerator

        generator = RunningCellMaskingGenerator(
            (TOTAL_TUBELETS, 16, 16), mask_ratio=0.5
        )
        self.running_cell_masks = torch.as_tensor(
            generator.all_mask_maps, dtype=torch.bool, device=self.device
        ).reshape(4, TOTAL_TUBELETS * TOKENS_PER_TUBELET)

    def _phase_averaged_prediction(self, model_input: torch.Tensor) -> torch.Tensor:
        total_tokens = TOTAL_TUBELETS * TOKENS_PER_TUBELET
        output_channels = self.model.decoder.head.out_features
        sums = torch.zeros(
            1,
            total_tokens,
            output_channels,
            dtype=model_input.dtype,
            device=self.device,
        )
        counts = torch.zeros(total_tokens, dtype=torch.int64, device=self.device)
        future_start = CONTEXT_TUBELETS * TOKENS_PER_TUBELET

        with torch.inference_mode():
            for phase in range(4):
                decoder_mask = torch.ones(
                    1, total_tokens, dtype=torch.bool, device=self.device
                )
                selected = ~self.running_cell_masks[phase]
                selected[:future_start] = False
                decoder_mask[0, selected] = False
                phase_output = self.model(model_input, self.mask, decoder_mask)
                positions = torch.nonzero(selected, as_tuple=False).flatten()
                if phase_output.shape[1] != positions.numel():
                    raise RuntimeError("Running-cell output/token mapping is inconsistent")
                sums[0, positions] += phase_output[0]
                counts[positions] += 1

        future_counts = counts[future_start:]
        if not torch.all(future_counts == 2):
            raise RuntimeError("Each future token must occur in exactly two of four phases")
        prediction = sums[:, future_start:] / future_counts[None, :, None]
        return prediction[:, : SCORED_FUTURE_TUBELETS * TOKENS_PER_TUBELET]

    def score_pair(
        self, possible_rgb: np.ndarray, impossible_rgb: np.ndarray
    ) -> CandidateErrors:
        possible = self._pad_to_official_16(possible_rgb)
        impossible = self._pad_to_official_16(impossible_rgb)
        model_input = torch.zeros_like(possible)
        model_input[:, :, :8] = possible[:, :, :8]
        prediction = self._phase_averaged_prediction(model_input)
        self.prediction_calls += 1

        errors = []
        with torch.inference_mode():
            for candidate in (possible, impossible):
                all_targets = self._patch_targets(candidate)
                future_targets = all_targets[self.mask].reshape(
                    1,
                    (TOTAL_TUBELETS - CONTEXT_TUBELETS) * TOKENS_PER_TUBELET,
                    -1,
                )
                target = future_targets[
                    :, : SCORED_FUTURE_TUBELETS * TOKENS_PER_TUBELET
                ]
                errors.append(
                    (prediction - target)
                    .square()
                    .reshape(
                        1,
                        SCORED_FUTURE_TUBELETS,
                        TOKENS_PER_TUBELET,
                        -1,
                    )
                    .mean(dim=(0, 2, 3))
                )
        if not all(torch.isfinite(item).all() for item in (prediction, *errors)):
            raise RuntimeError("VideoMAE-V2 running-cell output is non-finite")
        return CandidateErrors(
            errors[0].cpu().numpy(), errors[1].cpu().numpy(), self.frame_spans
        )

    def provenance(self) -> dict:
        value = super().provenance()
        value.update(
            {
                "evaluation_variant": (
                    "official trained decoder-head output with pretrained "
                    "running-cell decoder masking"
                ),
                "decoder_mask": (
                    "official RunningCellMaskingGenerator, 2x2 cell, ratio 0.5"
                ),
                "phase_aggregation": (
                    "uniform mean of all four fixed phases; every future token "
                    "is predicted in exactly two phases"
                ),
                "aggregate_predictions_per_pair": 1,
                "model_forward_passes_per_aggregate_prediction": 4,
                "phase_or_weight_selected_from_scores": False,
                "selection_timing": (
                    "additive post-hoc checkpoint-configuration sensitivity after "
                    "prior benchmark results were known"
                ),
                "reporting_status": (
                    "official-mask ensemble sensitivity; strict single-forward "
                    "comparison should use the official-output adapter"
                ),
            }
        )
        return value
