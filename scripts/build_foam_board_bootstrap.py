import argparse
import json
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
LAB_DIR = SCRIPT_DIR.parent
DEFAULT_RAW_DIR = LAB_DIR / "datasets" / "foam_ball" / "raw" / "20260712_173758"
DEFAULT_POSITIVE_VIDEO = DEFAULT_RAW_DIR / "foam_ball_trajectory_20260712_174030_244752.mp4"
DEFAULT_NEGATIVE_VIDEO = DEFAULT_RAW_DIR / "foam_ball_trajectory_20260712_173801_266933.mp4"
DEFAULT_OUTPUT = LAB_DIR / "datasets" / "foam_board" / "bootstrap_v1"

# Manually reviewed foam-board boxes in the short positive video, stored as x1, y1, x2, y2.
KEYFRAME_BOXES = {
    0: (333, 229, 455, 467),
    20: (338, 220, 456, 467),
    40: (326, 128, 469, 404),
    60: (285, 131, 407, 392),
    75: (268, 124, 383, 345),
    80: (287, 125, 405, 370),
    90: (287, 136, 387, 386),
    100: (289, 127, 406, 362),
    120: (285, 130, 410, 373),
    140: (270, 120, 400, 384),
    150: (256, 96, 373, 421),
    155: (219, 96, 372, 421),
    160: (220, 112, 376, 431),
    165: (194, 94, 352, 393),
    170: (221, 120, 379, 419),
    175: (268, 142, 410, 449),
    180: (303, 170, 423, 480),
    200: (309, 176, 436, 480),
    213: (320, 177, 455, 463),
}


def interpolate_box(frame_index):
    frames = sorted(KEYFRAME_BOXES)
    if frame_index <= frames[0]:
        return np.asarray(KEYFRAME_BOXES[frames[0]], dtype=np.float64)
    if frame_index >= frames[-1]:
        return np.asarray(KEYFRAME_BOXES[frames[-1]], dtype=np.float64)

    for left, right in zip(frames, frames[1:]):
        if left <= frame_index <= right:
            alpha = (frame_index - left) / (right - left)
            left_box = np.asarray(KEYFRAME_BOXES[left], dtype=np.float64)
            right_box = np.asarray(KEYFRAME_BOXES[right], dtype=np.float64)
            return left_box * (1.0 - alpha) + right_box * alpha
    raise RuntimeError(f"failed to interpolate frame {frame_index}")


def yolo_label(box, width, height):
    x1, y1, x2, y2 = box
    x1 = float(np.clip(x1, 0, width - 1))
    y1 = float(np.clip(y1, 0, height - 1))
    x2 = float(np.clip(x2, x1 + 1, width))
    y2 = float(np.clip(y2, y1 + 1, height))
    cx = 0.5 * (x1 + x2) / width
    cy = 0.5 * (y1 + y2) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return f"0 {cx:.8f} {cy:.8f} {box_width:.8f} {box_height:.8f}\n"


def read_frame(cap, frame_index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read frame {frame_index}")
    return frame


def save_positive_samples(video_path, output_dir, stride, validation_start):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open positive video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved = {"train": 0, "val": 0}
    manifest = []
    try:
        for frame_index in range(0, min(frame_count, 214), stride):
            split = "val" if frame_index >= validation_start else "train"
            frame = read_frame(cap, frame_index)
            height, width = frame.shape[:2]
            box = interpolate_box(frame_index)
            stem = f"positive_{frame_index:06d}"
            image_path = output_dir / "images" / split / f"{stem}.jpg"
            label_path = output_dir / "labels" / split / f"{stem}.txt"
            cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            label_path.write_text(yolo_label(box, width, height), encoding="utf-8")

            if frame_index % 15 == 0 or frame_index == 213:
                debug = frame.copy()
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(debug, "foam_board", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imwrite(str(output_dir / "debug" / f"{stem}.jpg"), debug)

            saved[split] += 1
            manifest.append({"type": "positive", "split": split, "frame": frame_index, "box_xyxy": box.tolist()})
    finally:
        cap.release()
    return saved, manifest


def save_negative_samples(video_path, output_dir, count):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open negative video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first_frame = max(1, frame_count // 20)
    last_frame = min(2300, max(1, frame_count - frame_count // 20 - 1))
    indices = np.linspace(first_frame, last_frame, count).astype(int)
    saved = {"train": 0, "val": 0}
    manifest = []
    try:
        for sample_index, frame_index in enumerate(indices):
            split = "val" if sample_index % 5 == 0 else "train"
            frame = read_frame(cap, int(frame_index))
            stem = f"negative_{int(frame_index):06d}"
            image_path = output_dir / "images" / split / f"{stem}.jpg"
            label_path = output_dir / "labels" / split / f"{stem}.txt"
            cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            label_path.write_text("", encoding="utf-8")
            saved[split] += 1
            manifest.append({"type": "negative", "split": split, "frame": int(frame_index)})
    finally:
        cap.release()
    return saved, manifest


def main():
    parser = argparse.ArgumentParser(description="Build the first foam-board YOLO dataset from reviewed video frames.")
    parser.add_argument("--positive-video", default=str(DEFAULT_POSITIVE_VIDEO))
    parser.add_argument("--negative-video", default=str(DEFAULT_NEGATIVE_VIDEO))
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--stride", type=int, default=3, help="Positive-video frame stride.")
    parser.add_argument("--validation-start", type=int, default=150, help="Positive frames at or after this go to val.")
    parser.add_argument("--negative-count", type=int, default=30)
    args = parser.parse_args()

    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.negative_count < 2:
        raise ValueError("--negative-count must be at least 2")

    positive_video = Path(args.positive_video).resolve()
    negative_video = Path(args.negative_video).resolve()
    output_dir = Path(args.out).resolve()
    for relative in ("images/train", "images/val", "labels/train", "labels/val", "debug"):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)
    for pattern in ("images/*/positive_*.jpg", "images/*/negative_*.jpg", "labels/*/positive_*.txt", "labels/*/negative_*.txt", "debug/positive_*.jpg"):
        for generated_file in output_dir.glob(pattern):
            generated_file.unlink()

    positive_counts, positive_manifest = save_positive_samples(
        positive_video,
        output_dir,
        stride=args.stride,
        validation_start=args.validation_start,
    )
    negative_counts, negative_manifest = save_negative_samples(
        negative_video,
        output_dir,
        count=args.negative_count,
    )

    yaml_path = output_dir / "foam_board.yaml"
    yaml_path.write_text(
        f"path: {output_dir.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: foam_board\n",
        encoding="utf-8",
    )
    manifest = {
        "class_name": "foam_board",
        "positive_video": str(positive_video),
        "negative_video": str(negative_video),
        "positive_counts": positive_counts,
        "negative_counts": negative_counts,
        "samples": positive_manifest + negative_manifest,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"dataset: {output_dir}")
    print(f"positive train/val: {positive_counts['train']}/{positive_counts['val']}")
    print(f"negative train/val: {negative_counts['train']}/{negative_counts['val']}")
    print(f"yaml: {yaml_path}")
    print(f"debug: {output_dir / 'debug'}")


if __name__ == "__main__":
    main()
