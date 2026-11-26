from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from .adapter import (
    CHECKPOINT_BYTES,
    CHECKPOINT_SHA256,
    MODEL_NAME,
    verify_checkpoint,
    verify_official_configs,
)
from .definition import VITL_SPEC


class VJEPA2ViTLTests(unittest.TestCase):
    def test_official_architecture_is_large(self) -> None:
        self.assertEqual(MODEL_NAME, "vit_large")
        checks = verify_official_configs()["architecture_checks"]
        self.assertEqual(checks["hidden_size"], 1024)
        self.assertEqual(checks["num_hidden_layers"], 24)
        self.assertEqual(checks["num_attention_heads"], 16)

    def test_partial_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            path.write_bytes(b"partial")
            with self.assertRaisesRegex(RuntimeError, "Incomplete or unexpected"):
                verify_checkpoint(path)

    def test_official_checkpoint_identity_is_pinned(self) -> None:
        self.assertEqual(CHECKPOINT_BYTES, 5_127_726_842)
        self.assertEqual(
            CHECKPOINT_SHA256,
            "5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012",
        )
        self.assertEqual(VITL_SPEC.checkpoints[0].sha256, CHECKPOINT_SHA256)

    def test_temporal_grid_matches_three_future_tubelets(self) -> None:
        self.assertEqual(
            VITL_SPEC.expected_frame_spans,
            ((20, 21), (22, 23), (24, 25)),
        )


if __name__ == "__main__":
    unittest.main()
