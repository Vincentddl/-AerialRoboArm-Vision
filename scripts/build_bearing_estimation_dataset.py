from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
TRACKING_DIR = PROJECT_DIR / "tracking"
sys.path.insert(0, str(TRACKING_DIR))

from bearing import BearingMapper  # noqa: E402


DEFAULT_SESSIONS = [
    PROJECT_DIR
    / "datasets"
    / "foam_board_2p1mm_zaxis"
    / "raw"
    / "20260714_223253",
    PROJECT_DIR
    / "datasets"
    / "trajectory_validation"
    / "raw"
    / "session_20260715_231329",
]


@dataclass(frozen=True)
class Observation:
    frame: int
    timestamp: float
    yaw: float
    pitch: float
    confidence: float
    clipped: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build grouped short-horizon bearing-estimation samples."
    )
    parser.add_argument("--session", type=Path, action="append", dest="sessions")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=PROJECT_DIR / "configs" / "camera_2p1mm_640x480_fisheye.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "datasets" / "bearing_estimation" / "v1" / "bearing_sequences.npz",
    )
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--horizons-ms", type=int, nargs="+", default=[100, 200])
    parser.add_argument("--future-tolerance-ms", type=float, default=50.0)
    parser.add_argument("--history-ms", type=float, default=350.0)
    parser.add_argument("--max-history", type=int, default=12)
    parser.add_argument("--min-history", type=int, default=3)
    parser.add_argument("--max-gap-frames", type=int, default=4)
    parser.add_argument("--min-segment-motion-deg", type=float, default=2.0)
    return parser.parse_args()


def find_one(session: Path, pattern: str) -> Path:
    matches = sorted(session.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern} in {session}, found {len(matches)}")
    return matches[0]


def load_session(
    session: Path,
    mapper: BearingMapper,
    confidence: float,
) -> list[Observation]:
    predictions_path = session / "v7_predictions.json"
    if not predictions_path.exists():
        raise FileNotFoundError(f"missing predictions: {predictions_path}")
    timestamp_path = find_one(session, "*.timestamps.jsonl")
    timestamp_records = [
        json.loads(line)
        for line in timestamp_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    raw_times = [float(item["monotonic_seconds"]) for item in timestamp_records]
    start = raw_times[0]
    timestamps = [value - start for value in raw_times]
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    width, height = mapper.image_size
    observations = []

    for record in predictions:
        frame = int(record["frame"])
        if frame >= len(timestamps):
            continue
        candidates = [
            item for item in record.get("boxes", []) if float(item["conf"]) >= confidence
        ]
        if not candidates:
            continue
        selected = max(candidates, key=lambda item: float(item["conf"]))
        box = [float(value) for value in selected["xyxy"]]
        center = (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))
        yaw, pitch = mapper.pixel_to_angles(center)
        clipped = (
            box[0] <= 1.0
            or box[1] <= 1.0
            or box[2] >= width - 1.0
            or box[3] >= height - 1.0
        )
        observations.append(
            Observation(
                frame=frame,
                timestamp=timestamps[frame],
                yaw=yaw,
                pitch=pitch,
                confidence=float(selected["conf"]),
                clipped=clipped,
            )
        )
    return observations


def split_segments(
    observations: Sequence[Observation], max_gap_frames: int
) -> list[list[Observation]]:
    segments: list[list[Observation]] = []
    current: list[Observation] = []
    for observation in observations:
        if current and observation.frame - current[-1].frame > max_gap_frames:
            segments.append(current)
            current = []
        current.append(observation)
    if current:
        segments.append(current)
    return segments


def segment_motion_deg(segment: Sequence[Observation], mapper: BearingMapper) -> float:
    first = (segment[0].yaw, segment[0].pitch)
    return max(
        mapper.angular_error_deg(first, (item.yaw, item.pitch)) for item in segment[1:]
    )


def group_split(group: str) -> int:
    bucket = int(hashlib.sha1(group.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return 0
    if bucket < 85:
        return 1
    return 2


def nearest_future(
    segment: Sequence[Observation],
    anchor_index: int,
    target_time: float,
    tolerance: float,
) -> Observation | None:
    candidates = [
        item
        for item in segment[anchor_index + 1 :]
        if abs(item.timestamp - target_time) <= tolerance
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: abs(item.timestamp - target_time))


def build_samples(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict]:
    mapper = BearingMapper(args.calibration)
    sessions = [path.resolve() for path in (args.sessions or DEFAULT_SESSIONS)]
    feature_rows = []
    lengths = []
    horizons = []
    targets = []
    weights = []
    splits = []
    groups = []
    sources = []
    anchors = []
    segment_count = 0
    accepted_segments = 0

    for session in sessions:
        observations = load_session(session, mapper, args.confidence)
        for local_segment_id, segment in enumerate(
            split_segments(observations, args.max_gap_frames)
        ):
            segment_count += 1
            if len(segment) < args.min_history + 1:
                continue
            if segment_motion_deg(segment, mapper) < args.min_segment_motion_deg:
                continue
            accepted_segments += 1
            group = f"{session.name}:segment-{local_segment_id:04d}"
            split = group_split(group)

            for anchor_index in range(args.min_history - 1, len(segment) - 1):
                anchor = segment[anchor_index]
                history = [
                    item
                    for item in segment[: anchor_index + 1]
                    if anchor.timestamp - item.timestamp <= args.history_ms / 1000.0
                ][-args.max_history :]
                if len(history) < args.min_history:
                    continue
                features = np.zeros((args.max_history, 5), dtype=np.float32)
                for row, item in enumerate(history):
                    features[row] = (
                        item.timestamp - anchor.timestamp,
                        item.yaw - anchor.yaw,
                        item.pitch - anchor.pitch,
                        item.confidence,
                        float(item.clipped),
                    )

                for horizon_ms in args.horizons_ms:
                    future = nearest_future(
                        segment,
                        anchor_index,
                        anchor.timestamp + horizon_ms / 1000.0,
                        args.future_tolerance_ms / 1000.0,
                    )
                    if future is None:
                        continue
                    quality_weight = anchor.confidence * future.confidence
                    if anchor.clipped or future.clipped:
                        quality_weight *= 0.4
                    feature_rows.append(features.copy())
                    lengths.append(len(history))
                    horizons.append(horizon_ms / 1000.0)
                    targets.append((future.yaw - anchor.yaw, future.pitch - anchor.pitch))
                    weights.append(max(quality_weight, 0.05))
                    splits.append(split)
                    groups.append(group)
                    sources.append(session.name)
                    anchors.append(anchor.frame)

    if not feature_rows:
        raise RuntimeError("no estimation samples were produced")

    arrays = {
        "features": np.asarray(feature_rows, dtype=np.float32),
        "lengths": np.asarray(lengths, dtype=np.int64),
        "horizons": np.asarray(horizons, dtype=np.float32).reshape(-1, 1),
        "targets": np.asarray(targets, dtype=np.float32),
        "weights": np.asarray(weights, dtype=np.float32),
        "splits": np.asarray(splits, dtype=np.int64),
        "groups": np.asarray(groups),
        "sources": np.asarray(sources),
        "anchor_frames": np.asarray(anchors, dtype=np.int64),
    }
    split_names = {0: "train", 1: "validation", 2: "test"}
    manifest = {
        "format_version": 1,
        "calibration": str(args.calibration.resolve()),
        "sessions": [str(path) for path in sessions],
        "configuration": {
            "confidence": args.confidence,
            "horizons_ms": args.horizons_ms,
            "future_tolerance_ms": args.future_tolerance_ms,
            "history_ms": args.history_ms,
            "max_history": args.max_history,
            "min_history": args.min_history,
            "max_gap_frames": args.max_gap_frames,
            "min_segment_motion_deg": args.min_segment_motion_deg,
            "feature_order": [
                "relative_time_s",
                "relative_yaw_deg",
                "relative_pitch_deg",
                "detector_confidence",
                "clipped_flag",
            ],
            "target": ["future_delta_yaw_deg", "future_delta_pitch_deg"],
        },
        "total_detected_segments": segment_count,
        "accepted_moving_segments": accepted_segments,
        "samples": len(feature_rows),
        "samples_by_split": {
            split_names[key]: int(np.sum(arrays["splits"] == key)) for key in split_names
        },
        "groups_by_split": {
            split_names[key]: len(set(arrays["groups"][arrays["splits"] == key].tolist()))
            for key in split_names
        },
        "samples_by_horizon_ms": dict(
            sorted(Counter(int(round(value * 1000)) for value in horizons).items())
        ),
        "samples_by_source": dict(sorted(Counter(sources).items())),
        "quality_note": (
            "All accepted short trajectories are included. Edge-clipped targets are retained "
            "with lower sample weight. V7 centers are pseudo-labels, not manual ground truth."
        ),
    }
    return arrays, manifest


def main() -> None:
    args = parse_args()
    arrays, manifest = build_samples(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"saved: {args.output.resolve()}")
    print(f"manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
