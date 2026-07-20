import sys
from pathlib import Path

import yolo_track


SCRIPT_DIR = Path(__file__).resolve().parent
LAB_DIR = SCRIPT_DIR.parent
FOAM_BOARD_MODEL = LAB_DIR / "models" / "foam_board_2p1mm_v7.pt"
CAMERA_CALIBRATION = LAB_DIR / "configs" / "camera_2p1mm_640x480_fisheye.json"


def main():
    defaults = [
        "--model",
        str(FOAM_BOARD_MODEL),
        "--classes",
        "foam_board",
        "--conf",
        "0.20",
        "--new-track-conf",
        "0.40",
        "--new-track-confirm-frames",
        "2",
        "--new-track-min-motion",
        "12",
        "--new-track-candidate-max-age",
        "4",
        "--min-target-speed",
        "80",
        "--stationary-max-frames",
        "8",
        "--coast-frames",
        "3",
        "--iou",
        "0.5",
        "--imgsz",
        "640",
        "--device",
        "0",
        "--predict-seconds",
        "0.2",
        "--min-age",
        "8",
        "--max-match-distance",
        "140",
        "--max-missed",
        "12",
        "--measurement-noise",
        "30",
        "--process-noise",
        "300",
        "--camera-width",
        "640",
        "--camera-height",
        "480",
        "--camera-backend",
        "dshow",
        "--calibration",
        str(CAMERA_CALIBRATION),
        "--skip-unchanged",
    ]
    sys.argv = [sys.argv[0], *defaults, *sys.argv[1:]]
    yolo_track.main()


if __name__ == "__main__":
    main()
