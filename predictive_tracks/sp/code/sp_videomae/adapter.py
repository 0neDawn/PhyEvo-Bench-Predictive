"""Score the official VideoMAE-V2 decoder's predicted future hidden states.

Unlike a hand-written dynamics head, the predicted states here are the mask-token
states produced by the checkpoint's own future-masked decoder immediately before
its trained pixel projection. Candidate states are produced only after that
prediction, using the same frozen encoder/decoder and causal visible prefixes.
"""

from __future__ import annotations

import numpy as np
import torch

from sp_benchmark.adapter_contract import CandidateErrors
from sp_protocol.adapters.videomaev2_official16 import (
    Adapter as Official16Adapter,
    CONTEXT_TUBELETS,
    SCORED_FUTURE_TUBELETS,
    TOKENS_PER_TUBELET,
    TOTAL_TUBELETS,
)


class Adapter(Official16Adapter):
    model_name = "videomaev2_official_decoder_native_state"

    def _prefix_encoder_mask(self, visible_tubelets: int) -> torch.Tensor:
        mask = torch.ones(
            1,
            TOTAL_TUBELETS * TOKENS_PER_TUBELET,
            dtype=torch.bool,
            device=self.device,
        )
        mask[:, : visible_tubelets * TOKENS_PER_TUBELET] = False
        return mask

    def _future_decode_positions(self) -> torch.Tensor:
        positions = torch.zeros(
            1,
            TOTAL_TUBELETS * TOKENS_PER_TUBELET,
            dtype=torch.bool,
            device=self.device,
        )
        start = CONTEXT_TUBELETS * TOKENS_PER_TUBELET
        stop = (CONTEXT_TUBELETS + SCORED_FUTURE_TUBELETS) * TOKENS_PER_TUBELET
        positions[:, start:stop] = True
        return positions

    def _decoder_hidden(
        self,
        video: torch.Tensor,
        encoder_mask: torch.Tensor,
        decoded_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Reproduce the official forward path and stop before decoder.head."""
        x_visible = self.model.encoder(video, encoder_mask)
        x_visible = self.model.encoder_to_decoder(x_visible)
        batch, _, channels = x_visible.shape
        position = self.model.pos_embed.expand(batch, -1, -1).type_as(video).to(
            video.device
        ).clone().detach()
        visible_position = position[~encoder_mask].reshape(batch, -1, channels)
        decoded_position = position[decoded_positions].reshape(batch, -1, channels)
        values = x_visible + visible_position
        decoded_count = decoded_position.shape[1]
        if decoded_count:
            mask_tokens = self.model.mask_token.expand(
                batch, decoded_count, -1
            ) + decoded_position
            values = torch.cat([values, mask_tokens], dim=1)
        for block in self.model.decoder.blocks:
            values = block(values)
        values = self.model.decoder.norm(values)
        return values[:, -decoded_count:] if decoded_count else values

    def _prediction(self, history_container: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            prediction = self._decoder_hidden(
                history_container,
                self._prefix_encoder_mask(CONTEXT_TUBELETS),
                self._future_decode_positions(),
            )
        return prediction.reshape(
            1,
            SCORED_FUTURE_TUBELETS,
            TOKENS_PER_TUBELET,
            -1,
        )

    def _causal_targets(self, candidate: torch.Tensor) -> torch.Tensor:
        targets = []
        no_decoded_positions = torch.zeros(
            1,
            TOTAL_TUBELETS * TOKENS_PER_TUBELET,
            dtype=torch.bool,
            device=self.device,
        )
        with torch.inference_mode():
            for step in range(SCORED_FUTURE_TUBELETS):
                visible_tubelets = CONTEXT_TUBELETS + step + 1
                hidden = self._decoder_hidden(
                    candidate,
                    self._prefix_encoder_mask(visible_tubelets),
                    no_decoded_positions,
                )
                targets.append(hidden[:, -TOKENS_PER_TUBELET:])
        return torch.stack(targets, dim=1)

    def score_pair(
        self, possible_rgb: np.ndarray, impossible_rgb: np.ndarray
    ) -> CandidateErrors:
        possible = self._pad_to_official_16(possible_rgb)
        impossible = self._pad_to_official_16(impossible_rgb)
        history_container = torch.zeros_like(possible)
        history_container[:, :, :8] = possible[:, :, :8]
        prediction = self._prediction(history_container)
        self.prediction_calls += 1

        errors = []
        for candidate in (possible, impossible):
            target = self._causal_targets(candidate)
            errors.append((prediction - target).square().mean(dim=(0, 2, 3)))
        if not all(torch.isfinite(item).all() for item in (prediction, *errors)):
            raise RuntimeError("VideoMAE-V2 decoder-native states are non-finite")
        return CandidateErrors(
            errors[0].cpu().numpy(), errors[1].cpu().numpy(), self.frame_spans
        )

    def provenance(self) -> dict:
        value = super().provenance()
        value["evaluation_variant"] = (
            "official future-masked decoder hidden state before its trained pixel head"
        )
        value["prediction"] = (
            "checkpoint decoder mask-token states conditioned only on shared history; "
            "no external or hand-written forecasting head"
        )
        value["candidate_target"] = (
            "same frozen encoder and decoder hidden state with a causal prefix ending "
            "at the scored tubelet, computed only after prediction"
        )
        value["distance"] = "MSE in layer-normalized official decoder hidden space"
        value["additional_prediction_head"] = False
        value["candidate_available_to_prediction"] = False
        value["selection_or_calibration_on_sp_labels"] = False
        value["status"] = "post-hoc native-space sensitivity analysis"
        return value
