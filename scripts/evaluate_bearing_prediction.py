from __future__ import annotations

import argparse
import json
import math
import sys
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
TRACKING_DIR = PROJECT_DIR / "tracking"
DEFAULT_SESSION = (
    PROJECT_DIR
    / "datasets"
    / "foam_board_2p1mm_zaxis"
    / "raw"
    / "20260714_223253"
)

sys.path.insert(0, str(TRACKING_DIR))

from bearing import (  # noqa: E402
    BearingMapper,
    BearingObservation,
    RobustBearingPredictor,
    summarize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate undistorted bearing prediction with recorded V7 detections."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_SESSION / "v7_predictions.json",
    )
    parser.add_argument(
        "--timestamps",
        type=Path,
        default=DEFAULT_SESSION
        / "foam_board_2p1mm_zaxis_trajectory_20260714_223351_701282.timestamps.jsonl",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=PROJECT_DIR / "configs" / "camera_2p1mm_640x480_fisheye.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "outputs" / "bearing_prediction_eval_v7.json",
    )
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument(
        "--horizons-ms",
        type=int,
        nargs="+",
        default=[100, 200],
        help="Prediction horizons. The unsafe 400 ms experiment is opt-in.",
    )
    parser.add_argument("--history-ms", type=float, default=350.0)
    parser.add_argument("--max-history-gap-ms", type=float, default=120.0)
    parser.add_argument("--future-tolerance-ms", type=float, default=50.0)
    parser.add_argument("--min-samples", type=int, default=6)
    parser.add_argument("--min-history-span-ms", type=float, default=100.0)
    parser.add_argument("--min-motion-deg", type=float, default=2.0)
    parser.add_argument("--edge-margin-px", type=float, default=1.0)
    parser.add_argument("--recency-tau-ms", type=float, default=200.0)
    parser.add_argument("--huber-delta-deg", type=float, default=0.75)
    parser.add_argument("--max-angular-speed", type=float, default=600.0)
    parser.add_argument("--max-angular-acceleration", type=float, default=2500.0)
    parser.add_argument(
        "--exclude-clipped-history",
        action="store_true",
        help="Do not use boxes touching an image boundary as predictor observations.",
    )
    return parser.parse_args()


def load_timestamps(path: Path) -> List[float]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    values = [float(record["monotonic_seconds"]) for record in records]
    if not values:
        raise ValueError(f"no timestamps in {path}")
    start = values[0]
    return [value - start for value in values]


def box_center(box: Sequence[float]) -> tuple[float, float]:
    return 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])


def is_clipped(box: Sequence[float], image_size: tuple[int, int], margin: float) -> bool:
    width, height = image_size
    return (
        box[0] <= margin
        or box[1] <= margin
        or box[2] >= width - margin
        or box[3] >= height - margin
    )


def load_observations(
    predictions_path: Path,
    timestamps: Sequence[float],
    mapper: BearingMapper,
    confidence: float,
    edge_margin_px: float,
) -> tuple[List[BearingObservation], Dict[int, dict], dict]:
    frames = json.loads(predictions_path.read_text(encoding="utf-8"))
    observations: List[BearingObservation] = []
    metadata: Dict[int, dict] = {}
    frames_with_multiple = 0
    rejected_low_confidence = 0

    for record in frames:
        frame_index = int(record["frame"])
        if frame_index >= len(timestamps):
            continue
        candidates = [box for box in record.get("boxes", []) if float(box["conf"]) >= confidence]
        rejected_low_confidence += sum(
            1 for box in record.get("boxes", []) if float(box["conf"]) < confidence
        )
        if not candidates:
            continue
        if len(candidates) > 1:
            frames_with_multiple += 1
        selected = max(candidates, key=lambda item: float(item["conf"]))
        bbox = tuple(float(value) for value in selected["xyxy"])
        center = box_center(bbox)
        yaw, pitch = mapper.pixel_to_angles(center)
        clipped = is_clipped(bbox, mapper.image_size, edge_margin_px)
        observation = BearingObservation(
            timestamp=float(timestamps[frame_index]),
            yaw_deg=yaw,
            pitch_deg=pitch,
            score=float(selected["conf"]),
            frame_index=frame_index,
            pixel=center,
        )
        observations.append(observation)
        metadata[frame_index] = {"bbox": list(bbox), "clipped": clipped}

    observations.sort(key=lambda item: item.timestamp)
    stats = {
        "frames_in_predictions": len(frames),
        "selected_detections": len(observations),
        "clipped_selected_detections": sum(
            1 for item in observations if metadata[item.frame_index]["clipped"]
        ),
        "frames_with_multiple_candidates": frames_with_multiple,
        "rejected_low_confidence_boxes": rejected_low_confidence,
    }
    return observations, metadata, stats


def history_for_anchor(
    observations: Sequence[BearingObservation],
    anchor_index: int,
    history_seconds: float,
    max_gap_seconds: float,
    metadata: Dict[int, dict],
    exclude_clipped: bool,
) -> List[BearingObservation]:
    anchor = observations[anchor_index]
    history = [anchor]
    previous = anchor
    for index in range(anchor_index - 1, -1, -1):
        candidate = observations[index]
        if anchor.timestamp - candidate.timestamp > history_seconds:
            break
        if previous.timestamp - candidate.timestamp > max_gap_seconds:
            break
        if not (exclude_clipped and metadata[candidate.frame_index]["clipped"]):
            history.append(candidate)
        previous = candidate
    history.reverse()
    if exclude_clipped and metadata[anchor.frame_index]["clipped"]:
        return []
    return history


def history_motion_deg(mapper: BearingMapper, history: Sequence[BearingObservation]) -> float:
    first = (history[0].yaw_deg, history[0].pitch_deg)
    latest = (history[-1].yaw_deg, history[-1].pitch_deg)
    return mapper.angular_error_deg(first, latest)


def find_future_observation(
    observations: Sequence[BearingObservation],
    timestamps: Sequence[float],
    target_time: float,
    tolerance_seconds: float,
) -> BearingObservation | None:
    index = bisect_left(timestamps, target_time)
    candidates = []
    left = max(0, index - 3)
    right = min(len(observations), index + 4)
    for item in observations[left:right]:
        delta = abs(item.timestamp - target_time)
        if delta <= tolerance_seconds:
            candidates.append((delta, -item.score, item))
    if not candidates:
        return None
    return min(candidates, key=lambda value: (value[0], value[1]))[2]


def evaluate(args: argparse.Namespace) -> dict:
    mapper = BearingMapper(args.calibration)
    frame_timestamps = load_timestamps(args.timestamps)
    observations, metadata, detection_stats = load_observations(
        args.predictions,
        frame_timestamps,
        mapper,
        args.confidence,
        args.edge_margin_px,
    )
    observation_times = [item.timestamp for item in observations]
    predictor = RobustBearingPredictor(
        history_seconds=args.history_ms / 1000.0,
        min_samples=args.min_samples,
        recency_tau_seconds=args.recency_tau_ms / 1000.0,
        huber_delta_deg=args.huber_delta_deg,
        max_angular_speed_deg_s=args.max_angular_speed,
        max_angular_acceleration_deg_s2=args.max_angular_acceleration,
    )

    output = {
        "source": {
            "predictions": str(args.predictions.resolve()),
            "timestamps": str(args.timestamps.resolve()),
            "calibration": str(args.calibration.resolve()),
            "truth": "nearest future V7 detection center within tolerance (pseudo-ground-truth)",
        },
        "configuration": {
            "confidence": args.confidence,
            "history_ms": args.history_ms,
            "max_history_gap_ms": args.max_history_gap_ms,
            "future_tolerance_ms": args.future_tolerance_ms,
            "min_samples": args.min_samples,
            "min_history_span_ms": args.min_history_span_ms,
            "min_motion_deg": args.min_motion_deg,
            "exclude_clipped_history": args.exclude_clipped_history,
            "recency_tau_ms": args.recency_tau_ms,
            "huber_delta_deg": args.huber_delta_deg,
            "max_angular_speed_deg_s": args.max_angular_speed,
            "max_angular_acceleration_deg_s2": args.max_angular_acceleration,
        },
        "detection_stats": detection_stats,
        "horizons": {},
    }

    histories: Dict[int, List[BearingObservation]] = {}
    rejected = defaultdict(int)
    for anchor_index, anchor in enumerate(observations):
        history = history_for_anchor(
            observations,
            anchor_index,
            args.history_ms / 1000.0,
            args.max_history_gap_ms / 1000.0,
            metadata,
            args.exclude_clipped_history,
        )
        if len(history) < args.min_samples:
            rejected["insufficient_samples"] += 1
            continue
        if history[-1].timestamp - history[0].timestamp < args.min_history_span_ms / 1000.0:
            rejected["short_history_span"] += 1
            continue
        if history_motion_deg(mapper, history) < args.min_motion_deg:
            rejected["insufficient_motion"] += 1
            continue
        histories[anchor_index] = history

    output["history_selection"] = {
        "eligible_anchors": len(histories),
        "rejected": dict(rejected),
    }

    for horizon_ms in args.horizons_ms:
        horizon_seconds = horizon_ms / 1000.0
        errors = {
            method: {"angular": [], "yaw_abs": [], "pitch_abs": [], "residual": []}
            for method in predictor.METHODS
        }
        visible_predictions = defaultdict(int)
        method_wins = defaultdict(int)
        matched = 0
        matched_truth_clipped = 0
        records = []

        for anchor_index, history in histories.items():
            anchor = observations[anchor_index]
            truth = find_future_observation(
                observations,
                observation_times,
                anchor.timestamp + horizon_seconds,
                args.future_tolerance_ms / 1000.0,
            )
            if truth is None:
                continue
            matched += 1
            truth_angles = (truth.yaw_deg, truth.pitch_deg)
            if metadata[truth.frame_index]["clipped"]:
                matched_truth_clipped += 1
            record = {
                "anchor_frame": anchor.frame_index,
                "truth_frame": truth.frame_index,
                "actual_horizon_ms": round((truth.timestamp - anchor.timestamp) * 1000.0, 3),
                "truth_clipped": metadata[truth.frame_index]["clipped"],
                "methods": {},
            }
            record_errors = {}
            for method in predictor.METHODS:
                prediction = predictor.predict(history, horizon_seconds, method)
                predicted_angles = (prediction.yaw_deg, prediction.pitch_deg)
                angular_error = mapper.angular_error_deg(predicted_angles, truth_angles)
                yaw_error = abs(prediction.yaw_deg - truth.yaw_deg)
                pitch_error = abs(prediction.pitch_deg - truth.pitch_deg)
                errors[method]["angular"].append(angular_error)
                errors[method]["yaw_abs"].append(yaw_error)
                errors[method]["pitch_abs"].append(pitch_error)
                errors[method]["residual"].append(prediction.residual_deg)
                if mapper.is_visible(predicted_angles):
                    visible_predictions[method] += 1
                record_errors[method] = angular_error
                record["methods"][method] = {
                    "yaw_deg": round(prediction.yaw_deg, 5),
                    "pitch_deg": round(prediction.pitch_deg, 5),
                    "angular_error_deg": round(angular_error, 5),
                    "fit_residual_deg": round(prediction.residual_deg, 5),
                    "predicted_visible": mapper.is_visible(predicted_angles),
                }
            winner = min(record_errors, key=record_errors.get)
            method_wins[winner] += 1
            records.append(record)

        metrics = {}
        for method in predictor.METHODS:
            metrics[method] = {
                "angular_error_deg": summarize(errors[method]["angular"]),
                "yaw_absolute_error_deg": summarize(errors[method]["yaw_abs"]),
                "pitch_absolute_error_deg": summarize(errors[method]["pitch_abs"]),
                "fit_residual_deg": summarize(errors[method]["residual"]),
                "predicted_visible_fraction": round(
                    visible_predictions[method] / max(len(histories), 1), 4
                ),
                "wins": method_wins[method],
            }

        hold_median = metrics["hold"]["angular_error_deg"]["median"]
        comparisons = {}
        for method in ("angular_velocity", "angular_acceleration"):
            candidate = metrics[method]["angular_error_deg"]["median"]
            if hold_median is None or candidate is None or hold_median == 0:
                improvement = None
            else:
                improvement = round(100.0 * (hold_median - candidate) / hold_median, 2)
            comparisons[f"{method}_vs_hold_median_percent"] = improvement

        output["horizons"][str(horizon_ms)] = {
            "eligible": len(histories),
            "matched": matched,
            "future_detection_coverage": round(matched / max(len(histories), 1), 4),
            "matched_truth_clipped": matched_truth_clipped,
            "matched_truth_clipped_fraction": round(matched_truth_clipped / max(matched, 1), 4),
            "metrics": metrics,
            "comparisons": comparisons,
            "records": records,
        }
    return output


def print_summary(result: dict) -> None:
    stats = result["detection_stats"]
    print(
        f"detections: {stats['selected_detections']} selected, "
        f"{stats['clipped_selected_detections']} clipped"
    )
    print(f"eligible moving anchors: {result['history_selection']['eligible_anchors']}")
    for horizon, values in result["horizons"].items():
        print(
            f"\n{horizon} ms: matched {values['matched']}/{values['eligible']} "
            f"({100.0 * values['future_detection_coverage']:.1f}% coverage)"
        )
        print("  method                 median      p90      mean    visible")
        for method, metrics in values["metrics"].items():
            angular = metrics["angular_error_deg"]
            print(
                f"  {method:22s} {angular['median']:7.2f}  {angular['p90']:7.2f}  "
                f"{angular['mean']:7.2f}  {100.0 * metrics['predicted_visible_fraction']:6.1f}%"
            )
        print(
            "  median improvement vs hold: "
            f"velocity {values['comparisons']['angular_velocity_vs_hold_median_percent']}%, "
            f"acceleration {values['comparisons']['angular_acceleration_vs_hold_median_percent']}%"
        )


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print_summary(result)
    print(f"\nsaved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
