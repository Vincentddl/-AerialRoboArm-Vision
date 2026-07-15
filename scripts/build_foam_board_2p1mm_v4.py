import json
import shutil
from pathlib import Path

import cv2

from build_foam_board_2p1mm_v3 import LAB_DIR, yolo_label


SOURCE_DATASET = LAB_DIR / "datasets" / "foam_board_2p1mm" / "v3_final"
RAW_DIR = LAB_DIR / "datasets" / "foam_board_2p1mm" / "raw" / "20260713_152652"
VIDEO_PATH = RAW_DIR / "foam_board_2p1mm_trajectory_20260713_152655_332571.mp4"
OUTPUT_DIR = LAB_DIR / "datasets" / "foam_board_2p1mm" / "v4"


# Reviewed target boxes from independent motions in the new metal-panel scene.
# The box follows the visible blur envelope when fast motion stretches the target.
TRAIN_POSITIVE_BOXES = {
    296: (59, 375, 162, 480),
    346: (302, 362, 419, 480),
    398: (259, 356, 403, 480),
    670: (294, 366, 379, 479),
    944: (223, 358, 547, 480),
    1039: (283, 4, 473, 304),
    1076: (307, 363, 475, 479),
    1140: (455, 185, 640, 480),
    1182: (396, 0, 611, 294),
    1396: (438, 361, 551, 480),
    1502: (392, 7, 574, 324),
    1726: (365, 4, 552, 192),
    1784: (409, 186, 640, 480),
    1835: (532, 0, 640, 178),
    2245: (257, 358, 411, 479),
    2377: (251, 0, 496, 261),
}

VAL_POSITIVE_BOXES = {
    2668: (233, 360, 494, 480),
    2733: (293, 357, 449, 480),
    2777: (342, 0, 468, 266),
    2943: (240, 280, 405, 480),
    3011: (174, 357, 399, 480),
    3134: (118, 359, 289, 480),
    3200: (230, 356, 472, 479),
    3241: (350, 357, 536, 480),
}

# These ranges were reviewed as target-free. They include the slatted panel,
# dark right edge, operator, hand, and phone that caused V3 false detections.
TRAIN_NEGATIVE_INDICES = sorted(
    set(range(0, 181, 9))
    | set(range(3470, 3601, 7))
    | {814, 1699, 1866, 2029, 2841}
)
VAL_NEGATIVE_INDICES = list(range(3700, 3901, 10))


def prepare_directories():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for relative in ("images/train", "images/val", "labels/train", "labels/val", "debug/train", "debug/val"):
        (OUTPUT_DIR / relative).mkdir(parents=True, exist_ok=True)


def copy_source_dataset():
    counts = {"train": 0, "val": 0}
    for split in ("train", "val"):
        for image_path in sorted((SOURCE_DATASET / "images" / split).glob("*.jpg")):
            label_path = SOURCE_DATASET / "labels" / split / f"{image_path.stem}.txt"
            shutil.copy2(image_path, OUTPUT_DIR / "images" / split / image_path.name)
            shutil.copy2(label_path, OUTPUT_DIR / "labels" / split / label_path.name)
            counts[split] += 1
    return counts


def load_frames(indices):
    wanted = set(indices)
    frames = {}
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {VIDEO_PATH}")
    try:
        frame_index = 0
        while wanted:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame_index in wanted:
                frames[frame_index] = frame.copy()
                wanted.remove(frame_index)
            frame_index += 1
    finally:
        cap.release()
    if wanted:
        raise RuntimeError(f"failed to read frames: {sorted(wanted)}")
    return frames


def save_sample(frame, split, frame_index, box=None):
    sample_type = "positive" if box is not None else "negative"
    stem = f"metal_scene_{sample_type}_{frame_index:06d}"
    height, width = frame.shape[:2]
    image_path = OUTPUT_DIR / "images" / split / f"{stem}.jpg"
    label_path = OUTPUT_DIR / "labels" / split / f"{stem}.txt"
    cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    debug = frame.copy()
    if box is None:
        label_path.write_text("", encoding="utf-8")
        cv2.putText(debug, "NEGATIVE", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    else:
        x1, y1, x2, y2 = box
        clipped = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
        label_path.write_text(yolo_label(clipped, width, height), encoding="utf-8")
        cv2.rectangle(debug, (clipped[0], clipped[1]), (clipped[2], clipped[3]), (0, 255, 0), 2)
        cv2.putText(debug, "foam_board", (clipped[0], max(24, clipped[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.imwrite(str(OUTPUT_DIR / "debug" / split / f"{stem}.jpg"), debug)
    return {"frame": frame_index, "split": split, "type": sample_type, "box": box}


def main():
    if not SOURCE_DATASET.exists():
        raise FileNotFoundError(SOURCE_DATASET)
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(VIDEO_PATH)

    prepare_directories()
    copied = copy_source_dataset()
    requested = (
        set(TRAIN_POSITIVE_BOXES)
        | set(VAL_POSITIVE_BOXES)
        | set(TRAIN_NEGATIVE_INDICES)
        | set(VAL_NEGATIVE_INDICES)
    )
    frames = load_frames(requested)
    samples = []

    for frame_index, box in TRAIN_POSITIVE_BOXES.items():
        samples.append(save_sample(frames[frame_index], "train", frame_index, box))
    for frame_index, box in VAL_POSITIVE_BOXES.items():
        samples.append(save_sample(frames[frame_index], "val", frame_index, box))
    for frame_index in TRAIN_NEGATIVE_INDICES:
        samples.append(save_sample(frames[frame_index], "train", frame_index))
    for frame_index in VAL_NEGATIVE_INDICES:
        samples.append(save_sample(frames[frame_index], "val", frame_index))

    yaml_path = OUTPUT_DIR / "foam_board_2p1mm.yaml"
    yaml_path.write_text(
        f"path: {OUTPUT_DIR.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: foam_board\n",
        encoding="utf-8",
    )
    manifest = {
        "source_dataset": str(SOURCE_DATASET),
        "source_video": str(VIDEO_PATH),
        "copied_counts": copied,
        "new_positive_counts": {"train": len(TRAIN_POSITIVE_BOXES), "val": len(VAL_POSITIVE_BOXES)},
        "new_negative_counts": {"train": len(TRAIN_NEGATIVE_INDICES), "val": len(VAL_NEGATIVE_INDICES)},
        "samples": samples,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"dataset: {OUTPUT_DIR}")
    print(f"copied train/val: {copied['train']}/{copied['val']}")
    print(f"new positives train/val: {len(TRAIN_POSITIVE_BOXES)}/{len(VAL_POSITIVE_BOXES)}")
    print(f"new negatives train/val: {len(TRAIN_NEGATIVE_INDICES)}/{len(VAL_NEGATIVE_INDICES)}")
    print(f"yaml: {yaml_path}")


if __name__ == "__main__":
    main()
