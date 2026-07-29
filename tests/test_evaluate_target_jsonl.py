from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from evaluate_target_jsonl import evaluate  # noqa: E402


class TargetJsonlEvaluationTest(unittest.TestCase):
    def test_scores_same_track_at_requested_horizon(self) -> None:
        rows = [
            {
                "time": 0.0,
                "track_id": 1,
                "pixel": [320.0, 240.0],
                "angle_deg": [0.0, 0.0],
                "offset_angle_deg": 0.0,
                "predicted_pixel": [330.0, 220.0],
                "predicted_angle_deg": [1.0, -2.0],
                "predicted_offset_angle_deg": 2.0,
                "prediction_valid": True,
            },
            {
                "time": 0.4,
                "track_id": 1,
                "pixel": [330.0, 220.0],
                "angle_deg": [1.0, -2.0],
                "offset_angle_deg": 2.0,
                "predicted_pixel": [330.0, 220.0],
                "predicted_angle_deg": [1.0, -2.0],
                "predicted_offset_angle_deg": 2.0,
                "prediction_valid": True,
            },
        ]

        result = evaluate(rows, horizon_s=0.4, tolerance_s=0.01)

        self.assertEqual(result["matched_forecasts"], 1)
        self.assertEqual(result["angular_error_deg"]["median"], 0.0)
        self.assertEqual(result["pixel_error"]["median"], 0.0)
        self.assertEqual(
            result["valid_only"]["offset_angle_error_deg"]["median"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
