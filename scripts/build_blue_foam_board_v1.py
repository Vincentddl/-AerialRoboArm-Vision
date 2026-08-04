"""Build a YOLO detection dataset for blue foam boards.

This script follows the same pattern as ``build_foam_board_2p1mm_v7.py`` but
is designed for blue foam board data collected via ``capture_dataset.py``.

Workflow
--------
1. Run ``capture_dataset.py --object blue_foam_board --static-mode`` to collect
   static-pose images (target: 200+).
2. Run ``capture_dataset.py --object blue_foam_board --record-video`` to record
   trajectory videos while tossing the blue board (target: 5-10 throws).
3. Review the trajectory video frame-by-frame; identify keyframes where the
   board is clearly visible and annotate bounding boxes ``(x1, y1, x2, y2)``.
   Save these as a JSON file (see ``KEYFRAME_SCHEMA`` below).
4. Identify time intervals where the board is definitely absent for negative
   sampling.
5. Edit the constants at the top of this script to point to your data, then
   run it: ``python scripts/build_blue_foam_board_v1.py``.

Keyframe JSON schema
--------------------
.. code-block:: json

    [
        {"frame": 245, "box": [280, 50, 360, 190], "split": "train", "throw_id": "throw_01"},
        {"frame": 252, "box": [290, 60, 350, 170], "split": "train", "throw_id": "throw_01"},
        ...
    ]

Boxes are ``[x1, y1, x2, y2]`` in pixel coordinates. Intermediate frames
are interpolated only within the same ``throw_id`` and a short frame gap.
"""

import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Configuration — edit these before running
# ---------------------------------------------------------------------------

LAB_DIR = Path(__file__).resolve().parents[1]

# Raw data directories (from capture_dataset.py output)
STATIC_DIR: Optional[Path] = LAB_DIR / "datasets" / "blue_foam_board" / "raw" / "static"
TRAJECTORY_VIDEO: Optional[Path] = LAB_DIR / "datasets" / "blue_foam_board" / "raw" / "trajectory" / "blue_foam_board_trajectory.mp4"

# Keyframe annotation file (you create this after reviewing the video)
KEYFRAME_PATH: Optional[Path] = LAB_DIR / "datasets" / "blue_foam_board" / "raw" / "trajectory" / "keyframes.json"

# Output directory
OUTPUT_DIR = LAB_DIR / "datasets" / "blue_foam_board" / "v1"

# Negative frame IDs — visually confirmed target-free frames from the
# trajectory video.  Separate lists for train and validation.
TRAIN_NEGATIVE_FRAME_IDS: Set[int] = set()
VAL_NEGATIVE_FRAME_IDS: Set[int] = set()

# If True, also copy the existing v7 negative frames as additional
# background-only training data.  These are still valid because the
# backgrounds (metal panel, tabletop, etc.) haven't changed.
INCLUDE_V7_NEGATIVES = True
V7_DATASET_DIR = LAB_DIR / "datasets" / "foam_board_2p1mm" / "v7"

# JPEG quality for output images
JPEG_QUALITY = 95
# Never interpolate labels across a long gap or between separate throws.
MAX_INTERPOLATION_GAP = 12

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def yolo_label(box, width, height):
    """Convert a pixel-coordinate box to a YOLO-format label line."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / (2 * width)
    cy = (y1 + y2) / (2 * height)
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}\n"


def interpolate_box(box_a, box_b, alpha):
    """Linearly interpolate between two pixel-coordinate boxes."""
    return tuple(
        int(round(a + alpha * (b - a)))
        for a, b in zip(box_a, box_b)
    )


def prepare_directories():
    """Create a fresh output directory tree."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for rel in ("images/train", "images/val", "labels/train", "labels/val",
                "debug/train", "debug/val"):
        (OUTPUT_DIR / rel).mkdir(parents=True, exist_ok=True)


def load_video_frames(frame_ids):
    """Extract specific frame indices from the trajectory video."""
    wanted = set(frame_ids)
    if not wanted or TRAJECTORY_VIDEO is None or not TRAJECTORY_VIDEO.exists():
        return {}
    frames = {}
    cap = cv2.VideoCapture(str(TRAJECTORY_VIDEO))
    if not cap.isOpened():
        print(f"Warning: could not open video: {TRAJECTORY_VIDEO}")
        return {}
    try:
        idx = 0
        while wanted:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if idx in wanted:
                frames[idx] = frame.copy()
                wanted.remove(idx)
            idx += 1
    finally:
        cap.release()
    if wanted:
        print(f"Warning: {len(wanted)} requested frames not found in video "
              f"(max frame index: {idx - 1})")
    return frames


def load_keyframes():
    """Load and validate the keyframe annotation JSON."""
    if KEYFRAME_PATH is None or not KEYFRAME_PATH.exists():
        return []
    data = json.loads(KEYFRAME_PATH.read_text(encoding="utf-8"))
    seen = set()
    for item in data:
        fid = item["frame"]
        if fid in seen:
            raise ValueError(f"duplicate frame {fid} in keyframes")
        seen.add(fid)
        if item["split"] not in ("train", "val"):
            raise ValueError(f"invalid split for frame {fid}: {item['split']}")
        if len(item["box"]) != 4:
            raise ValueError(f"invalid box for frame {fid}")
    return sorted(data, key=lambda x: x["frame"])


def generate_from_keyframes(keyframes, frames, split):
    """Generate positive samples by interpolating between keyframes.

    For each pair of consecutive keyframes in *split*, frames between them
    get linearly interpolated bounding boxes.  The keyframes themselves
    use their exact annotated boxes.
    """
    split_kf = [k for k in keyframes if k["split"] == split]
    if not split_kf:
        return []

    # Collect all frame IDs we need (keyframes + intermediates)
    needed = set()
    for k in split_kf:
        needed.add(k["frame"])
    valid_pairs = []
    for i in range(len(split_kf) - 1):
        first, second = split_kf[i], split_kf[i + 1]
        f_a, f_b = first["frame"], second["frame"]
        throw_id = first.get("throw_id")
        same_throw = throw_id is not None and throw_id == second.get("throw_id")
        if not same_throw or f_b - f_a > MAX_INTERPOLATION_GAP:
            continue
        valid_pairs.append((first, second))
        for fid in range(f_a + 1, f_b):
            needed.add(fid)

    # Load frames from video
    video_frames = load_video_frames(needed)
    if not video_frames:
        return []

    # Build frame_id → box mapping
    fid_box: Dict[int, tuple] = {}
    for k in split_kf:
        fid_box[k["frame"]] = tuple(k["box"])

    # Interpolate
    for a, b in valid_pairs:
        f_a, box_a = a["frame"], tuple(a["box"])
        f_b, box_b = b["frame"], tuple(b["box"])
        span = f_b - f_a
        for fid in range(f_a + 1, f_b):
            alpha = (fid - f_a) / span
            fid_box[fid] = interpolate_box(box_a, box_b, alpha)

    # Save samples
    samples = []
    for fid in sorted(fid_box.keys()):
        if fid not in video_frames:
            continue
        frame = video_frames[fid]
        box = fid_box[fid]
        is_keyframe = any(k["frame"] == fid for k in split_kf)
        source = "manual_keyframe" if is_keyframe else "interpolated"
        sample = save_sample(frame, fid, split, box, source)
        samples.append(sample)
    return samples


def save_sample(frame, frame_id, split, box, annotation_source):
    """Save one image + label + debug visualization."""
    sample_type = "positive" if box is not None else "negative"
    stem = f"blue_{sample_type}_{frame_id:06d}"
    img_path = OUTPUT_DIR / "images" / split / f"{stem}.jpg"
    lbl_path = OUTPUT_DIR / "labels" / split / f"{stem}.txt"
    h, w = frame.shape[:2]

    cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    debug = frame.copy()
    if box is None:
        lbl_path.write_text("", encoding="utf-8")
        cv2.putText(debug, "NEGATIVE", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    else:
        x1, y1, x2, y2 = (round(v) for v in box)
        clipped = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
        if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
            raise ValueError(f"box collapses after clipping for frame {frame_id}: {box}")
        lbl_path.write_text(yolo_label(clipped, w, h), encoding="utf-8")
        cv2.rectangle(debug, (clipped[0], clipped[1]), (clipped[2], clipped[3]),
                      (0, 255, 0), 2)
        cv2.putText(debug, annotation_source,
                    (clipped[0], max(24, clipped[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    cv2.imwrite(str(OUTPUT_DIR / "debug" / split / f"{stem}.jpg"), debug)
    return {
        "frame": frame_id,
        "split": split,
        "type": sample_type,
        "box": list(box) if box else None,
        "annotation_source": annotation_source,
    }


def process_static_images():
    """Process static-mode captures.

    Static images without a matching manual label are skipped. An empty YOLO
    label means "confirmed background"; using it for an unlabeled target image
    would directly train the detector to miss the foam board.

    If you have manually created YOLO-format .txt labels for the static
    images (same stem, in a parallel ``labels/`` directory), place them
    alongside the images and re-run.
    """
    samples = []
    if STATIC_DIR is None or not STATIC_DIR.exists():
        return samples

    image_files = sorted(STATIC_DIR.glob("*.jpg")) + sorted(STATIC_DIR.glob("*.png"))
    if not image_files:
        print("No static images found.")
        return samples

    # Simple split: every 10th image goes to val, rest to train
    # (for better split quality, use farthest-point sampling like v5)
    train_count = 0
    val_count = 0
    for idx, img_path in enumerate(image_files):
        split = "val" if idx % 10 == 0 else "train"
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Warning: could not read {img_path}")
            continue

        stem = f"static_{img_path.stem}"
        out_img = OUTPUT_DIR / "images" / split / f"{stem}.jpg"
        out_lbl = OUTPUT_DIR / "labels" / split / f"{stem}.txt"
        h, w = frame.shape[:2]

        # Check for a pre-existing manual label file
        manual_label = img_path.with_suffix(".txt")
        # Also check a parallel labels/ directory
        alt_label = STATIC_DIR.parent / "labels" / f"{img_path.stem}.txt"

        if manual_label.exists():
            shutil.copy2(str(manual_label), str(out_lbl))
            box = _read_first_box(manual_label, w, h)
            source = "manual_label"
        elif alt_label.exists():
            shutil.copy2(str(alt_label), str(out_lbl))
            box = _read_first_box(alt_label, w, h)
            source = "manual_label"
        else:
            # Unlabeled positives must never be converted into negatives.
            print(f"Skipping unlabeled static image: {img_path.name}")
            continue

        cv2.imwrite(str(out_img), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

        debug = frame.copy()
        if box is not None:
            x1, y1, x2, y2 = (round(v) for v in box)
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(debug, source, (x1, max(24, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        else:
            cv2.putText(debug, "NEEDS LABEL", (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.imwrite(str(OUTPUT_DIR / "debug" / split / f"{stem}.jpg"), debug)

        samples.append({
            "frame": None,
            "split": split,
            "type": "positive" if box else "needs_label",
            "box": list(box) if box else None,
            "annotation_source": source,
            "source_file": str(img_path),
        })
        if split == "train":
            train_count += 1
        else:
            val_count += 1

    print(f"Static images: {train_count} train, {val_count} val")
    return samples


def _read_first_box(label_path, width, height):
    """Read the first YOLO-format box from a label file, convert to pixels."""
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    parts = text.split()
    if len(parts) < 5:
        return None
    cls_id, cx, cy, bw, bh = (float(p) for p in parts[:5])
    cx *= width
    cy *= height
    bw *= width
    bh *= height
    return (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)


def copy_v7_negatives():
    """Copy negative (background-only) frames from the V7 dataset."""
    copied = {"train": 0, "val": 0}
    if not INCLUDE_V7_NEGATIVES or not V7_DATASET_DIR.exists():
        return copied
    for split in ("train", "val"):
        label_dir = V7_DATASET_DIR / "labels" / split
        if not label_dir.exists():
            continue
        for lbl_path in sorted(label_dir.glob("*.txt")):
            text = lbl_path.read_text(encoding="utf-8").strip()
            if text:  # positive sample — skip
                continue
            stem = lbl_path.stem
            img_path = V7_DATASET_DIR / "images" / split / f"{stem}.jpg"
            if not img_path.exists():
                continue
            new_stem = f"v7_negative_{stem}"
            shutil.copy2(str(img_path),
                         str(OUTPUT_DIR / "images" / split / f"{new_stem}.jpg"))
            (OUTPUT_DIR / "labels" / split / f"{new_stem}.txt").write_text(
                "", encoding="utf-8")
            copied[split] += 1
    print(f"Copied V7 negatives: {copied['train']} train, {copied['val']} val")
    return copied


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    prepare_directories()

    samples = []

    # 1. Copy V7 negatives as background data
    v7_neg = copy_v7_negatives()

    # 2. Process static images
    static_samples = process_static_images()
    samples.extend(static_samples)

    # 3. Process trajectory video keyframes + interpolation
    keyframes = load_keyframes()
    if keyframes:
        for split in ("train", "val"):
            traj_samples = generate_from_keyframes(keyframes, None, split)
            samples.extend(traj_samples)

        # Negative frames from the same video
        all_needed = (
            set(TRAIN_NEGATIVE_FRAME_IDS) | set(VAL_NEGATIVE_FRAME_IDS)
        )
        if all_needed and TRAJECTORY_VIDEO and TRAJECTORY_VIDEO.exists():
            neg_frames = load_video_frames(all_needed)
            for fid in sorted(TRAIN_NEGATIVE_FRAME_IDS):
                if fid in neg_frames:
                    sfx = save_sample(neg_frames[fid], fid, "train", None,
                                      "reviewed_negative")
                    samples.append(sfx)
            for fid in sorted(VAL_NEGATIVE_FRAME_IDS):
                if fid in neg_frames:
                    sfx = save_sample(neg_frames[fid], fid, "val", None,
                                      "reviewed_negative")
                    samples.append(sfx)
    else:
        print("No keyframe file found.  Skipping trajectory video processing.")
        print(f"Expected keyframe file at: {KEYFRAME_PATH}")

    # 4. Write YAML config
    yaml_path = OUTPUT_DIR / "blue_foam_board.yaml"
    yaml_path.write_text(
        f"path: {OUTPUT_DIR.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: foam_board\n",
        encoding="utf-8",
    )

    # 5. Tally counts
    def _count(split, stype):
        return sum(
            1 for s in samples
            if s["split"] == split and (
                (stype == "positive" and s["box"] is not None)
                or (stype == "negative" and s["box"] is None)
            )
        )

    pos_train = _count("train", "positive")
    pos_val = _count("val", "positive")
    neg_train = _count("train", "negative") + v7_neg["train"]
    neg_val = _count("val", "negative") + v7_neg["val"]

    total_train = pos_train + neg_train
    total_val = pos_val + neg_val

    # 6. Write manifest
    manifest = {
        "static_dir": str(STATIC_DIR) if STATIC_DIR else None,
        "trajectory_video": str(TRAJECTORY_VIDEO) if TRAJECTORY_VIDEO else None,
        "keyframe_file": str(KEYFRAME_PATH) if KEYFRAME_PATH else None,
        "v7_negatives_included": INCLUDE_V7_NEGATIVES,
        "v7_negative_counts": v7_neg,
        "counts": {
            "positive_train": pos_train,
            "positive_val": pos_val,
            "negative_train": neg_train,
            "negative_val": neg_val,
            "total_train": total_train,
            "total_val": total_val,
        },
        "samples": samples,
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nDataset written to: {OUTPUT_DIR}")
    print(f"  positive train/val: {pos_train}/{pos_val}")
    print(f"  negative train/val: {neg_train}/{neg_val}")
    print(f"  total    train/val: {total_train}/{total_val}")
    print(f"  yaml: {yaml_path}")
    print(f"\nNext steps:")
    print(f"  1. Review debug/ images, fix any bad labels")
    print(f"  2. For static images marked NEEDS LABEL: annotate and re-run")
    print(f"  3. Train: yolo detect train model=models/foam_board_2p1mm_v7.pt "
          f"data={yaml_path.as_posix()} imgsz=640 epochs=30 batch=4 lr0=0.001 freeze=10")


if __name__ == "__main__":
    main()
