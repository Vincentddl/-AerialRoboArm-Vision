import json
import shutil
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
LAB_DIR = SCRIPT_DIR.parent
SOURCE_DATASET = LAB_DIR / "datasets" / "foam_board" / "bootstrap_v2"
RAW_DIR = LAB_DIR / "datasets" / "foam_board_2p1mm" / "raw" / "20260712_230434"
VIDEO_PATH = RAW_DIR / "foam_board_2p1mm_trajectory_20260712_230437_575968.mp4"
OUTPUT_DIR = LAB_DIR / "datasets" / "foam_board_2p1mm" / "v3_stage1"


# Manually reviewed keyframes from independent throw/drop episodes. Boxes include
# the visible motion-blur envelope because that is what the detector receives.
KEYFRAME_BOXES = {
    136: (225, 80, 430, 270),
    236: (255, 35, 405, 220),
    404: (215, 0, 360, 305),
    471: (265, 40, 420, 335),
    608: (190, 0, 330, 405),
    756: (200, 0, 330, 355),
    847: (230, 0, 355, 230),
    884: (195, 0, 335, 295),
    1025: (195, 0, 345, 385),
    1104: (205, 0, 330, 345),
    1247: (210, 0, 335, 405),
    1332: (200, 0, 450, 235),
    1405: (100, 0, 430, 230),
    1503: (90, 0, 390, 190),
    1576: (90, 0, 390, 205),
    1645: (105, 0, 445, 235),
    1736: (245, 10, 435, 320),
    1818: (180, 0, 355, 295),
    1887: (105, 10, 440, 480),
    1966: (105, 45, 500, 480),
    2447: (90, 0, 325, 290),
    2478: (90, 0, 275, 345),
    2567: (135, 75, 365, 250),
    2605: (80, 0, 550, 480),
    3088: (260, 125, 640, 480),
    3153: (210, 0, 525, 240),
    3208: (80, 200, 305, 480),
    4183: (225, 0, 365, 370),
    4215: (120, 0, 295, 315),
    4243: (100, 0, 390, 150),
    4435: (185, 0, 545, 415),
    4689: (125, 0, 400, 345),
    4814: (90, 0, 380, 295),
    4881: (90, 0, 350, 285),
}

TRAIN_NEGATIVE_INDICES = sorted(
    set(np.linspace(2050, 2350, 35).astype(int).tolist() + np.linspace(2700, 2800, 15).astype(int).tolist())
)
VAL_NEGATIVE_INDICES = sorted(
    set(np.linspace(3350, 3950, 20).astype(int).tolist() + np.linspace(4950, 5000, 5).astype(int).tolist())
)


def yolo_label(box, width, height):
    x1, y1, x2, y2 = box
    cx = 0.5 * (x1 + x2) / width
    cy = 0.5 * (y1 + y2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}\n"


def prepare_directories():
    for relative in ("images/train", "images/val", "labels/train", "labels/val", "debug"):
        directory = OUTPUT_DIR / relative
        directory.mkdir(parents=True, exist_ok=True)
        for old_file in directory.glob("*"):
            if old_file.is_file():
                old_file.unlink()


def copy_bootstrap_samples(manifest):
    counts = {"train": 0, "val": 0}
    for split in ("train", "val"):
        for image_path in sorted((SOURCE_DATASET / "images" / split).glob("*.jpg")):
            stem = f"v2_{image_path.stem}"
            label_path = SOURCE_DATASET / "labels" / split / f"{image_path.stem}.txt"
            shutil.copy2(image_path, OUTPUT_DIR / "images" / split / f"{stem}.jpg")
            shutil.copy2(label_path, OUTPUT_DIR / "labels" / split / f"{stem}.txt")
            manifest.append({"source": "bootstrap_v2", "split": split, "image": image_path.name})
            counts[split] += 1
    return counts


def read_frame(cap, frame_index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read frame {frame_index}")
    return frame


def save_new_positive_frames(cap, manifest):
    counts = {"train": 0, "val": 0}
    for frame_index, box in KEYFRAME_BOXES.items():
        split = "train" if frame_index < 3000 else "val"
        frame = read_frame(cap, frame_index)
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = box
        clipped = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
        stem = f"lens21_positive_{frame_index:06d}"
        cv2.imwrite(str(OUTPUT_DIR / "images" / split / f"{stem}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (OUTPUT_DIR / "labels" / split / f"{stem}.txt").write_text(
            yolo_label(clipped, width, height), encoding="utf-8"
        )

        debug = frame.copy()
        cv2.rectangle(debug, (clipped[0], clipped[1]), (clipped[2], clipped[3]), (0, 255, 0), 2)
        cv2.putText(debug, f"{split} frame {frame_index}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.imwrite(str(OUTPUT_DIR / "debug" / f"{stem}.jpg"), debug)
        manifest.append(
            {"source": VIDEO_PATH.name, "type": "positive", "split": split, "frame": frame_index, "box": clipped}
        )
        counts[split] += 1
    return counts


def save_new_negative_frames(cap, manifest):
    counts = {"train": 0, "val": 0}
    for split, indices in (("train", TRAIN_NEGATIVE_INDICES), ("val", VAL_NEGATIVE_INDICES)):
        for frame_index in indices:
            frame = read_frame(cap, frame_index)
            stem = f"lens21_negative_{frame_index:06d}"
            cv2.imwrite(str(OUTPUT_DIR / "images" / split / f"{stem}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (OUTPUT_DIR / "labels" / split / f"{stem}.txt").write_text("", encoding="utf-8")
            manifest.append({"source": VIDEO_PATH.name, "type": "negative", "split": split, "frame": frame_index})
            counts[split] += 1
    return counts


def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(VIDEO_PATH)
    if not SOURCE_DATASET.exists():
        raise FileNotFoundError(SOURCE_DATASET)

    prepare_directories()
    manifest = []
    old_counts = copy_bootstrap_samples(manifest)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {VIDEO_PATH}")
    try:
        positive_counts = save_new_positive_frames(cap, manifest)
        negative_counts = save_new_negative_frames(cap, manifest)
    finally:
        cap.release()

    yaml_path = OUTPUT_DIR / "foam_board_2p1mm.yaml"
    yaml_path.write_text(
        f"path: {OUTPUT_DIR.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: foam_board\n",
        encoding="utf-8",
    )
    summary = {
        "class_name": "foam_board",
        "source_video": str(VIDEO_PATH),
        "old_counts": old_counts,
        "new_positive_counts": positive_counts,
        "new_negative_counts": negative_counts,
        "samples": manifest,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"dataset: {OUTPUT_DIR}")
    print(f"old train/val: {old_counts['train']}/{old_counts['val']}")
    print(f"new positive train/val: {positive_counts['train']}/{positive_counts['val']}")
    print(f"new negative train/val: {negative_counts['train']}/{negative_counts['val']}")
    print(f"yaml: {yaml_path}")


if __name__ == "__main__":
    main()
