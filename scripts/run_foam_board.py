import sys
from pathlib import Path

import yolo_track


SCRIPT_DIR = Path(__file__).resolve().parent
LAB_DIR = (
    Path(sys._MEIPASS)
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")
    else SCRIPT_DIR.parent
)
FOAM_BOARD_MODEL = LAB_DIR / "models" / "foam_board_2p1mm_v7.pt"
BLUE_FOAM_BOARD_MODEL = LAB_DIR / "models" / "blue_foam_board_v1.pt"
# Use blue foam board model if available; fall back to v7 (black)
DEFAULT_MODEL = BLUE_FOAM_BOARD_MODEL if BLUE_FOAM_BOARD_MODEL.exists() else FOAM_BOARD_MODEL
CAMERA_CALIBRATION = LAB_DIR / "configs" / "camera_2p1mm_640x480_fisheye.json"
CUDA_AVAILABLE = yolo_track.torch.cuda.is_available()
DEFAULT_DEVICE = "0" if CUDA_AVAILABLE else "cpu"
DEFAULT_IMAGE_SIZE = "640" if CUDA_AVAILABLE else "512"
DEFAULT_MIN_AGE = "6" if CUDA_AVAILABLE else "3"
DEFAULT_MATCH_DISTANCE = "140" if CUDA_AVAILABLE else "220"


def main():
    defaults = [
        "--model",
        str(DEFAULT_MODEL),
        "--classes",
        "foam_board",
        "--conf",
        "0.22",
        "--new-track-conf",
        "0.45",
        "--new-track-confirm-frames",
        "2",
        "--new-track-min-motion",
        "10",
        "--new-track-candidate-max-age",
        "3",
        "--min-target-speed",
        "60",
        "--stationary-max-frames",
        "8",
        "--coast-frames",
        "3",
        "--iou",
        "0.5",
        "--imgsz",
        DEFAULT_IMAGE_SIZE,
        "--device",
        DEFAULT_DEVICE,
        "--predict-seconds",
        "0.4",
        "--trajectory-steps",
        "8",
        "--min-age",
        DEFAULT_MIN_AGE,
        "--max-match-distance",
        DEFAULT_MATCH_DISTANCE,
        "--max-missed",
        "12",
        "--measurement-noise",
        "30",
        # Higher process noise in y (vertical) direction — Kalman responds
        # faster to up/down motion while keeping horizontal smooth.
        "--process-noise",
        "150",
        "--process-noise-y",
        "900",
        "--camera-width",
        "640",
        "--camera-height",
        "480",
        "--camera-backend",
        "dshow",
        # Short manual exposure reduces motion blur. Increase gain slightly if
        # the room is too dark; do not restore the previous forced gain=100.
        "--camera-exposure",
        "-7",
        "--camera-gain",
        "20",
        "--calibration",
        str(CAMERA_CALIBRATION),
        # Use ensemble predictor by default — fuses pixel Kalman + angle Kalman
        # + polynomial + ballistic for more reliable 0.4 s trajectory prediction.
        # Switch to --predictor kalman for baseline or --predictor angle_kalman
        # for pure angle-space tracking.
        "--predictor",
        "ensemble",
        "--angle-process-noise",
        "150",
        "--angle-measurement-noise",
        "0.5",
        # Ballistic predictor — models gravity for better long-horizon prediction.
        # Disable with --no-ballistic.
        "--ballistic-gravity",
        "400",
        "--ballistic-ema-alpha",
        "0.30",
        # Reject very large head/wall boxes and tiny texture false positives.
        "--max-box-area",
        "0.18",
        "--min-box-area",
        "0.0005",
        # Raw one-frame detections are not trusted; show only confirmed tracks.
        "--hide-centers",
        "--skip-unchanged",
    ]
    sys.argv = [sys.argv[0], *defaults, *sys.argv[1:]]
    yolo_track.main()


if __name__ == "__main__":
    main()
