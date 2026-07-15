import json
import shutil
from pathlib import Path

import cv2


LAB_DIR = Path(__file__).resolve().parents[1]
SOURCE_DATASET = LAB_DIR / "datasets" / "foam_board_2p1mm" / "v5"
RAW_DIR = LAB_DIR / "datasets" / "foam_board_2p1mm" / "raw" / "20260714_002808"
VIDEO_PATH = RAW_DIR / "foam_board_2p1mm_trajectory_20260714_002824_016099.mp4"
CANDIDATE_PATH = RAW_DIR / "inspection" / "dynamic_candidate_selection.json"
OUTPUT_DIR = LAB_DIR / "datasets" / "foam_board_2p1mm" / "v6"


# V5 boxes retained only after reviewing the full-resolution contact sheets.
REVIEWED_MODEL_FRAME_IDS = {
    83, 143, 200, 268, 287, 536, 643, 774, 1028, 1112, 1127, 1486, 1712,
    1973, 2136, 2230, 2253, 2297, 2407, 2901, 3091, 3180, 3215, 3451, 3582,
    3623, 3629, 3666, 3703, 3711, 3771, 3783, 3813, 3900, 3984, 3991, 4077,
    4140, 4155, 4209, 4232, 4238, 4353, 4423, 4493, 4551, 4590, 4598, 4626,
    4679, 4771, 4803, 5076, 5319, 5382, 5548, 5595, 5850, 5962, 5971, 6038,
}

# Clear target frames missed by V5. Boxes follow the visible target or motion-blur envelope.
MANUAL_BOXES = {
    1735: (334, 92, 415, 170),
    1818: (392, 88, 463, 174),
    1968: (314, 164, 420, 252),
    2225: (232, 184, 358, 289),
    3525: (402, 145, 500, 326),
    3685: (194, 342, 303, 454),
    3820: (198, 412, 304, 480),
    3846: (378, 72, 494, 350),
    4121: (286, 0, 402, 84),
    4212: (371, 132, 518, 232),
    4890: (506, 188, 640, 410),
    5666: (266, 362, 305, 438),
    6089: (376, 238, 640, 480),
}

VAL_POSITIVE_IDS = {200, 643, 1818, 1973, 2901, 3525, 3629, 3813, 3846, 4238, 4598, 4890, 5319, 5850}

# Confirmed target-free frames containing hands, white paper, the metal panel, or a phone.
NEGATIVE_FRAME_IDS = {
    354, 1930, 2019, 2260, 3016, 3907, 4167, 4253, 4295, 4362,
    5183, 5255, 5276, 5338, 5362, 5385, 5488, 5516, 5589, 6136,
}
VAL_NEGATIVE_IDS = {354, 3016, 4362, 5516}


def prepare_directories():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for relative in ("images/train", "images/val", "labels/train", "labels/val", "debug/train", "debug/val"):
        (OUTPUT_DIR / relative).mkdir(parents=True, exist_ok=True)


def copy_source_dataset():
    counts = {}
    negative_counts = {}
    for split in ("train", "val"):
        counts[split] = 0
        negative_counts[split] = 0
        for image_path in sorted((SOURCE_DATASET / "images" / split).glob("*.jpg")):
            label_path = SOURCE_DATASET / "labels" / split / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(label_path)
            shutil.copy2(image_path, OUTPUT_DIR / "images" / split / image_path.name)
            shutil.copy2(label_path, OUTPUT_DIR / "labels" / split / label_path.name)
            counts[split] += 1
            negative_counts[split] += not label_path.read_text(encoding="utf-8").strip()
    return counts, negative_counts


def reviewed_model_boxes():
    data = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))
    candidates = {item["frame"]: tuple(round(value) for value in item["box"]) for item in data["positives"]}
    missing = REVIEWED_MODEL_FRAME_IDS - candidates.keys()
    if missing:
        raise RuntimeError(f"reviewed model frames missing from candidate file: {sorted(missing)}")
    return {frame_id: candidates[frame_id] for frame_id in REVIEWED_MODEL_FRAME_IDS}


def load_frames(frame_ids):
    wanted = set(frame_ids)
    frames = {}
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {VIDEO_PATH}")
    try:
        index = 0
        while wanted:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if index in wanted:
                frames[index] = frame.copy()
                wanted.remove(index)
            index += 1
    finally:
        cap.release()
    if wanted:
        raise RuntimeError(f"failed to read frames: {sorted(wanted)}")
    return frames


def yolo_label(box, width, height):
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / (2 * width)
    center_y = (y1 + y2) / (2 * height)
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return f"0 {center_x:.8f} {center_y:.8f} {box_width:.8f} {box_height:.8f}\n"


def save_sample(frame, frame_id, split, box, annotation_source):
    sample_type = "positive" if box is not None else "negative"
    stem = f"dynamic_20260714_{sample_type}_{frame_id:06d}"
    image_path = OUTPUT_DIR / "images" / split / f"{stem}.jpg"
    label_path = OUTPUT_DIR / "labels" / split / f"{stem}.txt"
    height, width = frame.shape[:2]
    cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    debug = frame.copy()
    if box is None:
        label_path.write_text("", encoding="utf-8")
        cv2.putText(debug, "NEGATIVE", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    else:
        x1, y1, x2, y2 = box
        clipped = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
        if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
            raise ValueError(f"invalid box for frame {frame_id}: {box}")
        label_path.write_text(yolo_label(clipped, width, height), encoding="utf-8")
        cv2.rectangle(debug, (clipped[0], clipped[1]), (clipped[2], clipped[3]), (0, 255, 0), 2)
        cv2.putText(debug, annotation_source, (clipped[0], max(24, clipped[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.imwrite(str(OUTPUT_DIR / "debug" / split / f"{stem}.jpg"), debug)
    return {"frame": frame_id, "split": split, "type": sample_type, "box": box, "annotation_source": annotation_source}


def main():
    for required in (SOURCE_DATASET, VIDEO_PATH, CANDIDATE_PATH):
        if not required.exists():
            raise FileNotFoundError(required)

    model_boxes = reviewed_model_boxes()
    positive_boxes = {**model_boxes, **MANUAL_BOXES}
    overlap = set(positive_boxes) & NEGATIVE_FRAME_IDS
    if overlap:
        raise RuntimeError(f"frames marked both positive and negative: {sorted(overlap)}")

    prepare_directories()
    copied_counts, copied_negative_counts = copy_source_dataset()
    frames = load_frames(set(positive_boxes) | NEGATIVE_FRAME_IDS)
    samples = []

    for frame_id, box in sorted(positive_boxes.items()):
        split = "val" if frame_id in VAL_POSITIVE_IDS else "train"
        source = "manual_missed_target" if frame_id in MANUAL_BOXES else "reviewed_v5_box"
        samples.append(save_sample(frames[frame_id], frame_id, split, box, source))
    for frame_id in sorted(NEGATIVE_FRAME_IDS):
        split = "val" if frame_id in VAL_NEGATIVE_IDS else "train"
        samples.append(save_sample(frames[frame_id], frame_id, split, None, "reviewed_negative"))

    yaml_path = OUTPUT_DIR / "foam_board_2p1mm.yaml"
    yaml_path.write_text(
        f"path: {OUTPUT_DIR.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: foam_board\n",
        encoding="utf-8",
    )
    new_counts = {
        "positive_train": len(positive_boxes) - len(VAL_POSITIVE_IDS),
        "positive_val": len(VAL_POSITIVE_IDS),
        "negative_train": len(NEGATIVE_FRAME_IDS) - len(VAL_NEGATIVE_IDS),
        "negative_val": len(VAL_NEGATIVE_IDS),
        "manual_missed_targets": len(MANUAL_BOXES),
    }
    manifest = {
        "source_dataset": str(SOURCE_DATASET),
        "source_video": str(VIDEO_PATH),
        "candidate_file": str(CANDIDATE_PATH),
        "copied_counts": copied_counts,
        "copied_negative_counts": copied_negative_counts,
        "new_counts": new_counts,
        "samples": samples,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"dataset: {OUTPUT_DIR}")
    print(f"copied train/val: {copied_counts['train']}/{copied_counts['val']}")
    print(f"new positives train/val: {new_counts['positive_train']}/{new_counts['positive_val']}")
    print(f"new negatives train/val: {new_counts['negative_train']}/{new_counts['negative_val']}")
    print(f"manual missed targets: {new_counts['manual_missed_targets']}")
    print(f"total train/val: {copied_counts['train'] + new_counts['positive_train'] + new_counts['negative_train']}/{copied_counts['val'] + new_counts['positive_val'] + new_counts['negative_val']}")
    print(f"yaml: {yaml_path}")


if __name__ == "__main__":
    main()
