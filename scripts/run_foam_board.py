import sys
from pathlib import Path

import yolo_track


SCRIPT_DIR = Path(__file__).resolve().parent
LAB_DIR = (
    Path(sys._MEIPASS)
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")
    else SCRIPT_DIR.parent
)
FOAM_BOARD_MODEL = LAB_DIR / "models" / "foam_board_2p1mm_v8.pt"
FOAM_BOARD_FALLBACK_MODEL = LAB_DIR / "models" / "foam_board_2p1mm_v7.pt"
BLUE_FOAM_BOARD_MODEL = LAB_DIR / "models" / "blue_foam_board_v1.pt"
# Use the blue-target model when it is installed. Otherwise prefer the current
# black foam target V8 model and retain V7 as a portable-package fallback.
DEFAULT_MODEL = (
    BLUE_FOAM_BOARD_MODEL
    if BLUE_FOAM_BOARD_MODEL.exists()
    else FOAM_BOARD_MODEL if FOAM_BOARD_MODEL.exists() else FOAM_BOARD_FALLBACK_MODEL
)
CAMERA_CALIBRATION = LAB_DIR / "configs" / "camera_2p1mm_640x480_fisheye.json"
CUDA_AVAILABLE = yolo_track.torch.cuda.is_available()
DEFAULT_DEVICE = "0" if CUDA_AVAILABLE else "cpu"
DEFAULT_IMAGE_SIZE = "640" if CUDA_AVAILABLE else "512"
DEFAULT_MIN_AGE = "2"
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
        "0",
        "--new-track-candidate-max-age",
        "4",
        "--min-target-speed",
        "0",
        "--stationary-max-frames",
        "8",
        "--coast-frames",
        "0",
        "--iou",
        "0.5",
        "--imgsz",
        DEFAULT_IMAGE_SIZE,
        "--device",
        DEFAULT_DEVICE,
        # Follow the latest confirmed detector center. This accepts stationary,
        # slow and fast hand-held motion without future extrapolation.
        "--target-mode",
        "current",
        "--min-age",
        DEFAULT_MIN_AGE,
        "--max-match-distance",
        DEFAULT_MATCH_DISTANCE,
        "--max-missed",
        "12",
        "--measurement-noise",
        "30",
        "--process-noise",
        "250",
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
        # The optical axis is mounted 30 degrees below the arm's horizontal
        # x-axis. Only targets near the camera center plane are reachable.
        "--camera-down-tilt-deg",
        "30",
        "--lateral-tolerance-deg",
        "5",
        # Do not reject large boxes: a target naturally occupies more of the
        # frame while it is being moved toward the camera.
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
