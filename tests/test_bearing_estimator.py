from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from train_bearing_estimator import BearingGRU  # noqa: E402


class BearingEstimatorTest(unittest.TestCase):
    def test_variable_length_forward_shape(self) -> None:
        model = BearingGRU(input_size=5, hidden_size=16)
        features = torch.zeros((3, 8, 5), dtype=torch.float32)
        lengths = torch.tensor([8, 5, 3], dtype=torch.long)
        horizons = torch.tensor([[0.1], [0.2], [0.1]], dtype=torch.float32)
        output = model(features, lengths, horizons)
        self.assertEqual(tuple(output.shape), (3, 2))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
