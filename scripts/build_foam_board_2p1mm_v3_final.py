import json
import shutil
from pathlib import Path

import cv2

from build_foam_board_2p1mm_v3 import LAB_DIR, VIDEO_PATH, yolo_label


SOURCE_DATASET = LAB_DIR / "datasets" / "foam_board_2p1mm" / "v3_stage1"
OUTPUT_DIR = LAB_DIR / "datasets" / "foam_board_2p1mm" / "v3_final"

HARD_TRAIN_BOXES = {
    1411: (195, 340, 420, 480),
    1819: (185, 115, 460, 380),
    1888: (165, 295, 400, 480),
    1967: (165, 310, 420, 480),
    2413: (140, 105, 375, 480),
    2606: (125, 170, 535, 480),
    2861: (70, 175, 310, 480),
    4882: (90, 260, 480, 455),
}


def prepare_directories():
    for relative in ("images/train", "images/val", "labels/train", "labels/val", "debug"):
        directory = OUTPUT_DIR / relative
        directory.mkdir(parents=True, exist_ok=True)
        for old_file in directory.glob("*"):
            if old_file.is_file():
                old_file.unlink()


def copy_stage1():
    counts = {"train": 0, "val": 0}
    for split in ("train", "val"):
        for image_path in sorted((SOURCE_DATASET / "images" / split).glob("*.jpg")):
            label_path = SOURCE_DATASET / "labels" / split / f"{image_path.stem}.txt"
            shutil.copy2(image_path, OUTPUT_DIR / "images" / split / image_path.name)
            shutil.copy2(label_path, OUTPUT_DIR / "labels" / split / label_path.name)
            counts[split] += 1
    return counts


def add_hard_frames():
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {VIDEO_PATH}")
    records = []
    try:
        for frame_index, box in HARD_TRAIN_BOXES.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"failed to read frame {frame_index}")
            height, width = frame.shape[:2]
            clipped = (max(0, box[0]), max(0, box[1]), min(width, box[2]), min(height, box[3]))
            stem = f"lens21_hard_{frame_index:06d}"
            cv2.imwrite(str(OUTPUT_DIR / "images" / "train" / f"{stem}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (OUTPUT_DIR / "labels" / "train" / f"{stem}.txt").write_text(
                yolo_label(clipped, width, height), encoding="utf-8"
            )
            debug = frame.copy()
            cv2.rectangle(debug, (clipped[0], clipped[1]), (clipped[2], clipped[3]), (0, 255, 0), 2)
            cv2.imwrite(str(OUTPUT_DIR / "debug" / f"{stem}.jpg"), debug)
            records.append({"frame": frame_index, "box": clipped})
    finally:
        cap.release()
    return records


def main():
    prepare_directories()
    copied = copy_stage1()
    hard_frames = add_hard_frames()
    yaml_path = OUTPUT_DIR / "foam_board_2p1mm.yaml"
    yaml_path.write_text(
        f"path: {OUTPUT_DIR.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: foam_board\n",
        encoding="utf-8",
    )
    manifest = {
        "source_dataset": str(SOURCE_DATASET),
        "copied_counts": copied,
        "hard_train_frames": hard_frames,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"dataset: {OUTPUT_DIR}")
    print(f"copied train/val: {copied['train']}/{copied['val']}")
    print(f"hard train frames: {len(hard_frames)}")
    print(f"yaml: {yaml_path}")


if __name__ == "__main__":
    main()
