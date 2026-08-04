"""Recommended real-time entry point for the empirical servo-angle mode.

The existing ``run_foam_board.py`` entry point remains the legacy geometric
mode. This launcher adds the 2026-08-04 servo/camera calibration and keeps the
three angle definitions separate in the live overlay and JSON output.
"""

import sys
from pathlib import Path

import run_foam_board


SCRIPT_DIR = Path(__file__).resolve().parent
LAB_DIR = (
    Path(sys._MEIPASS)
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")
    else SCRIPT_DIR.parent
)
SERVO_ANGLE_CALIBRATION = (
    LAB_DIR / "configs" / "servo_to_optical_angle_yolo_v8_20260804_v1.json"
)


def main() -> None:
    calibrated_defaults = [
        "--servo-angle-calibration",
        str(SERVO_ANGLE_CALIBRATION),
        # V8 scores on the new calibration recording are lower than on its
        # training set. Keep the legacy launcher's stricter thresholds intact,
        # while this mode relies on two-frame confirmation to accept 0.10+.
        "--conf",
        "0.10",
        "--new-track-conf",
        "0.10",
        "--window-title",
        "Foam Board - Servo Calibrated Realtime",
    ]
    sys.argv = [sys.argv[0], *calibrated_defaults, *sys.argv[1:]]
    run_foam_board.main()


if __name__ == "__main__":
    main()
