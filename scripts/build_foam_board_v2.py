import json
from pathlib import Path

import cv2
import numpy as np

from build_foam_board_bootstrap import KEYFRAME_BOXES as VALIDATION_BOXES
from build_foam_board_bootstrap import yolo_label


SCRIPT_DIR = Path(__file__).resolve().parent
LAB_DIR = SCRIPT_DIR.parent
RAW_DIR = LAB_DIR / "datasets" / "foam_ball" / "raw" / "20260712_173758"
TRAIN_VIDEO = RAW_DIR / "foam_ball_trajectory_20260712_174042_358573.mp4"
VALIDATION_VIDEO = RAW_DIR / "foam_ball_trajectory_20260712_174030_244752.mp4"
NEGATIVE_VIDEO = RAW_DIR / "foam_ball_trajectory_20260712_173801_266933.mp4"
OUTPUT_DIR = LAB_DIR / "datasets" / "foam_board" / "bootstrap_v2"

TRAIN_BOXES = {
    0: (343, 214, 473, 480),
    50: (364, 232, 492, 480),
    100: (308, 154, 437, 413),
    150: (286, 129, 408, 378),
    200: (165, 120, 376, 312),
    250: (160, 119, 376, 334),
    300: (218, 138, 426, 436),
    350: (176, 293, 454, 480),
    400: (190, 306, 458, 480),
    450: (205, 306, 458, 480),
    500: (270, 315, 480, 480),
    525: (350, 318, 540, 455),
    550: (321, 0, 560, 165),
    575: (301, 383, 438, 480),
    625: (324, 0, 504, 208),
    650: (264, 12, 356, 210),
    675: (268, 65, 358, 268),
    700: (316, 205, 475, 384),
}


def interpolate_box(keyframes, frame_index):
    indices = sorted(keyframes)
    if frame_index <= indices[0]:
        return np.asarray(keyframes[indices[0]], dtype=np.float64)
    if frame_index >= indices[-1]:
        return np.asarray(keyframes[indices[-1]], dtype=np.float64)
    for left, right in zip(indices, indices[1:]):
        if left <= frame_index <= right:
            alpha = (frame_index - left) / (right - left)
            left_box = np.asarray(keyframes[left], dtype=np.float64)
            right_box = np.asarray(keyframes[right], dtype=np.float64)
            return left_box * (1.0 - alpha) + right_box * alpha
    raise RuntimeError(f"failed to interpolate frame {frame_index}")


def read_frame(cap, frame_index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read frame {frame_index}")
    return frame


def save_labeled_frames(video_path, keyframes, segments, split, stride, prefix, manifest):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    saved = 0
    try:
        for segment_start, segment_end in segments:
            for frame_index in range(segment_start, segment_end + 1, stride):
                frame = read_frame(cap, frame_index)
                height, width = frame.shape[:2]
                box = interpolate_box(keyframes, frame_index)
                stem = f"{prefix}_{frame_index:06d}"
                cv2.imwrite(str(OUTPUT_DIR / "images" / split / f"{stem}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                (OUTPUT_DIR / "labels" / split / f"{stem}.txt").write_text(
                    yolo_label(box, width, height),
                    encoding="utf-8",
                )
                if frame_index in keyframes:
                    debug = frame.copy()
                    x1, y1, x2, y2 = [int(round(v)) for v in box]
                    cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.imwrite(str(OUTPUT_DIR / "debug" / f"{stem}.jpg"), debug)
                manifest.append(
                    {"type": "positive", "split": split, "video": video_path.name, "frame": frame_index, "box_xyxy": box.tolist()}
                )
                saved += 1
    finally:
        cap.release()
    return saved


def save_negative_frames(manifest):
    cap = cv2.VideoCapture(str(NEGATIVE_VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open negative video: {NEGATIVE_VIDEO}")
    saved = {"train": 0, "val": 0}
    indices = np.linspace(205, 2300, 50).astype(int)
    try:
        for sample_index, frame_index in enumerate(indices):
            split = "val" if sample_index % 5 == 0 else "train"
            frame = read_frame(cap, int(frame_index))
            stem = f"negative_{int(frame_index):06d}"
            cv2.imwrite(str(OUTPUT_DIR / "images" / split / f"{stem}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (OUTPUT_DIR / "labels" / split / f"{stem}.txt").write_text("", encoding="utf-8")
            manifest.append({"type": "negative", "split": split, "video": NEGATIVE_VIDEO.name, "frame": int(frame_index)})
            saved[split] += 1
    finally:
        cap.release()
    return saved


def main():
    for relative in ("images/train", "images/val", "labels/train", "labels/val", "debug"):
        directory = OUTPUT_DIR / relative
        directory.mkdir(parents=True, exist_ok=True)
        for old_file in directory.glob("*"):
            if old_file.is_file():
                old_file.unlink()

    manifest = []
    positive_train = save_labeled_frames(
        TRAIN_VIDEO,
        TRAIN_BOXES,
        segments=[(0, 575), (625, 700)],
        split="train",
        stride=5,
        prefix="train_motion",
        manifest=manifest,
    )
    positive_val = save_labeled_frames(
        VALIDATION_VIDEO,
        VALIDATION_BOXES,
        segments=[(0, 213)],
        split="val",
        stride=6,
        prefix="val_vertical",
        manifest=manifest,
    )
    negative_counts = save_negative_frames(manifest)

    yaml_path = OUTPUT_DIR / "foam_board.yaml"
    yaml_path.write_text(
        f"path: {OUTPUT_DIR.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: foam_board\n",
        encoding="utf-8",
    )
    summary = {
        "class_name": "foam_board",
        "positive_train": positive_train,
        "positive_val": positive_val,
        "negative_train": negative_counts["train"],
        "negative_val": negative_counts["val"],
        "samples": manifest,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"dataset: {OUTPUT_DIR}")
    print(f"positive train/val: {positive_train}/{positive_val}")
    print(f"negative train/val: {negative_counts['train']}/{negative_counts['val']}")
    print(f"yaml: {yaml_path}")


if __name__ == "__main__":
    main()
