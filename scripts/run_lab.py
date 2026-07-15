import sys
from pathlib import Path

import yolo_track


SCRIPT_DIR = Path(__file__).resolve().parent
LAB_DIR = SCRIPT_DIR.parent
YOLO11S_MODEL = LAB_DIR / "models" / "yolo11s.pt"


def main():
    defaults = [
        "--model",
        str(YOLO11S_MODEL),
        "--classes",
        "bottle",
        "--conf",
        "0.25",
        "--iou",
        "0.5",
        "--imgsz",
        "640",
        "--device",
        "0",
        "--max-match-distance",
        "140",
        "--max-missed",
        "15",
        "--measurement-noise",
        "40",
        "--process-noise",
        "300",
        "--class-agnostic-tracking",
    ]
    sys.argv = [sys.argv[0], *defaults, *sys.argv[1:]]
    yolo_track.main()


if __name__ == "__main__":
    main()
