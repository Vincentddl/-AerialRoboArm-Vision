import json
import shutil
from pathlib import Path

import cv2
import numpy as np


LAB_DIR = Path(__file__).resolve().parents[1]
SOURCE_DATASET = LAB_DIR / "datasets" / "foam_board_2p1mm" / "v4"
RAW_DIR = LAB_DIR / "datasets" / "foam_board_2p1mm" / "raw" / "20260713_214630"
ANNOTATIONS_PATH = RAW_DIR / "inspection" / "static_auto_annotations.json"
OUTPUT_DIR = LAB_DIR / "datasets" / "foam_board_2p1mm" / "v5"
STATIC_TRAIN_COUNT = 125
STATIC_VAL_COUNT = 15


def prepare_directories():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for relative in ("images/train", "images/val", "labels/train", "labels/val", "debug/static_train", "debug/static_val"):
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


def load_samples():
    annotations = json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8"))
    samples = []
    for annotation in annotations:
        image_path = RAW_DIR / annotation["file"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"failed to read image: {image_path}")
        height, width = image.shape[:2]
        x1, y1, x2, y2 = map(int, annotation["xyxy"])
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"invalid box for {image_path.name}: {annotation['xyxy']}")
        samples.append({"path": image_path, "image": image, "box": (x1, y1, x2, y2), "annotation": annotation})
    if len(samples) != 200:
        raise RuntimeError(f"expected 200 static samples, found {len(samples)}")
    return samples


def pca_features(matrix, components):
    matrix = np.asarray(matrix, dtype=np.float32)
    matrix = (matrix - matrix.mean(axis=0)) / (matrix.std(axis=0) + 1e-6)
    _, _, vh = np.linalg.svd(matrix, full_matrices=False)
    return matrix @ vh[:components].T


def sample_features(samples):
    geometry = []
    appearance = []
    for sample in samples:
        image = sample["image"]
        height, width = image.shape[:2]
        x1, y1, x2, y2 = sample["box"]
        box_width = x2 - x1
        box_height = y2 - y1
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        crop = gray[y1:y2, x1:x2]

        dark_points = np.column_stack(np.where(crop < 120)).astype(np.float32)
        angle = 0.0
        if len(dark_points) >= 20:
            angle = cv2.minAreaRect(dark_points[:, ::-1])[2]
        sharpness = cv2.Laplacian(crop, cv2.CV_64F).var()
        geometry.append(
            [
                (x1 + x2) / (2 * width),
                (y1 + y2) / (2 * height),
                box_width / width,
                box_height / height,
                np.log((box_width * box_height) / (width * height) + 1e-6),
                np.log(box_width / box_height),
                np.sin(np.deg2rad(2 * angle)),
                np.cos(np.deg2rad(2 * angle)),
                np.log(sharpness + 1.0),
            ]
        )

        global_view = cv2.resize(gray, (16, 12), interpolation=cv2.INTER_AREA).reshape(-1) / 255.0
        object_view = cv2.resize(crop, (12, 12), interpolation=cv2.INTER_AREA).reshape(-1) / 255.0
        appearance.append(np.concatenate([global_view, object_view]))

    geometry = np.asarray(geometry, dtype=np.float32)
    geometry = (geometry - geometry.mean(axis=0)) / (geometry.std(axis=0) + 1e-6)
    appearance = pca_features(appearance, components=20)
    appearance = (appearance - appearance.mean(axis=0)) / (appearance.std(axis=0) + 1e-6)
    return np.concatenate([1.4 * geometry, appearance], axis=1)


def farthest_point_indices(features, count, allowed=None):
    allowed = np.arange(len(features)) if allowed is None else np.asarray(sorted(allowed), dtype=np.int32)
    center = features[allowed].mean(axis=0)
    first = allowed[np.argmax(np.sum((features[allowed] - center) ** 2, axis=1))]
    selected = [int(first)]
    min_distances = np.sum((features - features[first]) ** 2, axis=1)
    allowed_mask = np.zeros(len(features), dtype=bool)
    allowed_mask[allowed] = True
    allowed_mask[first] = False
    while len(selected) < count:
        candidates = np.where(allowed_mask)[0]
        next_index = int(candidates[np.argmax(min_distances[candidates])])
        selected.append(next_index)
        allowed_mask[next_index] = False
        distance = np.sum((features - features[next_index]) ** 2, axis=1)
        min_distances = np.minimum(min_distances, distance)
    return selected


def select_splits(samples):
    features = sample_features(samples)
    val_indices = farthest_point_indices(features, STATIC_VAL_COUNT)
    # Adjacent captures are near-duplicates, so keep immediate neighbors out of training.
    blocked = {index + offset for index in val_indices for offset in (-1, 0, 1)}
    train_pool = [index for index in range(len(samples)) if index not in blocked]
    if len(train_pool) < STATIC_TRAIN_COUNT:
        raise RuntimeError(f"not enough independent static training samples: {len(train_pool)}")
    train_indices = farthest_point_indices(features, STATIC_TRAIN_COUNT, train_pool)
    return sorted(train_indices), sorted(val_indices)


def yolo_label(box, width, height):
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / (2 * width)
    center_y = (y1 + y2) / (2 * height)
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return f"0 {center_x:.8f} {center_y:.8f} {box_width:.8f} {box_height:.8f}\n"


def save_static_sample(sample, split):
    source_path = sample["path"]
    stem = f"static_{source_path.stem}"
    image_path = OUTPUT_DIR / "images" / split / f"{stem}.jpg"
    label_path = OUTPUT_DIR / "labels" / split / f"{stem}.txt"
    shutil.copy2(source_path, image_path)

    image = sample["image"]
    height, width = image.shape[:2]
    label_path.write_text(yolo_label(sample["box"], width, height), encoding="utf-8")

    debug = image.copy()
    x1, y1, x2, y2 = sample["box"]
    cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(debug, "foam_board", (x1, max(24, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    debug_split = "static_train" if split == "train" else "static_val"
    cv2.imwrite(str(OUTPUT_DIR / "debug" / debug_split / f"{stem}.jpg"), debug)
    return {
        "file": source_path.name,
        "box": sample["box"],
        "annotation_source": sample["annotation"]["source"],
    }


def main():
    for required in (SOURCE_DATASET, RAW_DIR, ANNOTATIONS_PATH):
        if not required.exists():
            raise FileNotFoundError(required)

    samples = load_samples()
    train_indices, val_indices = select_splits(samples)
    prepare_directories()
    copied_counts, copied_negative_counts = copy_source_dataset()

    static_samples = {"train": [], "val": []}
    for split, indices in (("train", train_indices), ("val", val_indices)):
        for index in indices:
            static_samples[split].append(save_static_sample(samples[index], split))

    yaml_path = OUTPUT_DIR / "foam_board_2p1mm.yaml"
    yaml_path.write_text(
        f"path: {OUTPUT_DIR.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: foam_board\n",
        encoding="utf-8",
    )
    manifest = {
        "source_dataset": str(SOURCE_DATASET),
        "static_source": str(RAW_DIR),
        "annotations": str(ANNOTATIONS_PATH),
        "selection_method": "farthest-point sampling over geometry, pose, crop appearance, and global appearance",
        "validation_neighbor_exclusion": 1,
        "copied_counts": copied_counts,
        "copied_negative_counts": copied_negative_counts,
        "static_counts": {"train": len(train_indices), "val": len(val_indices)},
        "static_samples": static_samples,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"dataset: {OUTPUT_DIR}")
    print(f"copied train/val: {copied_counts['train']}/{copied_counts['val']}")
    print(f"copied negatives train/val: {copied_negative_counts['train']}/{copied_negative_counts['val']}")
    print(f"static train/val: {len(train_indices)}/{len(val_indices)}")
    print(f"total train/val: {copied_counts['train'] + len(train_indices)}/{copied_counts['val'] + len(val_indices)}")
    print(f"yaml: {yaml_path}")


if __name__ == "__main__":
    main()
