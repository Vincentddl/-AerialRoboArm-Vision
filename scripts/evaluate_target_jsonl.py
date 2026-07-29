from __future__ import annotations

import argparse
import bisect
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable


PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved future targets against later detections."
    )
    parser.add_argument("input", type=Path, help="JSONL produced by --save-jsonl")
    parser.add_argument(
        "--horizon-ms",
        type=float,
        default=400.0,
        help="Forecast horizon to score (default: 400 ms)",
    )
    parser.add_argument(
        "--tolerance-ms",
        type=float,
        default=50.0,
        help="Maximum timing difference for future pseudo-ground-truth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "outputs" / "target_prediction_evaluation.json",
    )
    return parser.parse_args()


def load_targets(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        timestamp = float(record["time"])
        for target in record.get("targets", []):
            row = dict(target)
            row["time"] = timestamp
            row["source_line"] = line_number
            rows.append(row)
    return rows


def angle_to_ray(angles: Iterable[float]) -> tuple[float, float, float]:
    yaw_deg, pitch_deg = angles
    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    return (
        math.sin(yaw) * math.cos(pitch),
        math.sin(pitch),
        math.cos(yaw) * math.cos(pitch),
    )


def angular_error_deg(first: Iterable[float], second: Iterable[float]) -> float:
    a = angle_to_ray(first)
    b = angle_to_ray(second)
    dot = sum(x * y for x, y in zip(a, b))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        alpha = position - lower
        return ordered[lower] * (1.0 - alpha) + ordered[upper] * alpha

    return {
        "count": len(ordered),
        "mean": round(sum(ordered) / len(ordered), 4),
        "median": round(percentile(0.5), 4),
        "p90": round(percentile(0.9), 4),
        "max": round(ordered[-1], 4),
    }


def evaluate(rows: list[dict], horizon_s: float, tolerance_s: float) -> dict:
    by_track: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_track[int(row["track_id"])].append(row)
    for track_rows in by_track.values():
        track_rows.sort(key=lambda item: item["time"])

    angular_errors: list[float] = []
    pixel_errors: list[float] = []
    offset_errors: list[float] = []
    valid_angular_errors: list[float] = []
    valid_pixel_errors: list[float] = []
    valid_offset_errors: list[float] = []
    timing_errors_ms: list[float] = []
    matched = 0
    valid_forecasts = 0
    valid_matched = 0
    records = []

    for anchor in rows:
        is_valid = bool(anchor.get("prediction_valid", True))
        valid_forecasts += int(is_valid)
        track_rows = by_track[int(anchor["track_id"])]
        times = [item["time"] for item in track_rows]
        wanted = anchor["time"] + horizon_s
        index = bisect.bisect_left(times, wanted)
        candidates = [
            item
            for item in track_rows[max(0, index - 2) : min(len(track_rows), index + 3)]
            if int(item.get("missed_frames", 0)) == 0
        ]
        if not candidates:
            continue
        truth = min(candidates, key=lambda item: abs(item["time"] - wanted))
        timing_error = abs(truth["time"] - wanted)
        if timing_error > tolerance_s or truth is anchor:
            continue
        matched += 1
        valid_matched += int(is_valid)
        timing_errors_ms.append(timing_error * 1000.0)
        result = {
            "track_id": int(anchor["track_id"]),
            "anchor_time": anchor["time"],
            "truth_time": truth["time"],
            "timing_error_ms": round(timing_error * 1000.0, 3),
            "prediction_valid": is_valid,
        }

        if anchor.get("predicted_angle_deg") is not None and truth.get("angle_deg") is not None:
            error = angular_error_deg(
                anchor["predicted_angle_deg"], truth["angle_deg"]
            )
            angular_errors.append(error)
            if is_valid:
                valid_angular_errors.append(error)
            result["angular_error_deg"] = round(error, 4)

        if anchor.get("predicted_pixel") is not None and truth.get("pixel") is not None:
            error = math.hypot(
                anchor["predicted_pixel"][0] - truth["pixel"][0],
                anchor["predicted_pixel"][1] - truth["pixel"][1],
            )
            pixel_errors.append(error)
            if is_valid:
                valid_pixel_errors.append(error)
            result["pixel_error"] = round(error, 4)

        if (
            anchor.get("predicted_offset_angle_deg") is not None
            and truth.get("offset_angle_deg") is not None
        ):
            error = abs(
                anchor["predicted_offset_angle_deg"] - truth["offset_angle_deg"]
            )
            offset_errors.append(error)
            if is_valid:
                valid_offset_errors.append(error)
            result["offset_angle_error_deg"] = round(error, 4)
        records.append(result)

    return {
        "input_rows": len(rows),
        "track_count": len(by_track),
        "horizon_ms": round(horizon_s * 1000.0, 3),
        "tolerance_ms": round(tolerance_s * 1000.0, 3),
        "matched_forecasts": matched,
        "future_detection_coverage": round(matched / max(len(rows), 1), 4),
        "valid_forecasts": valid_forecasts,
        "valid_matched_forecasts": valid_matched,
        "angular_error_deg": summarize(angular_errors),
        "pixel_error": summarize(pixel_errors),
        "offset_angle_error_deg": summarize(offset_errors),
        "valid_only": {
            "angular_error_deg": summarize(valid_angular_errors),
            "pixel_error": summarize(valid_pixel_errors),
            "offset_angle_error_deg": summarize(valid_offset_errors),
        },
        "timing_error_ms": summarize(timing_errors_ms),
        "truth_warning": (
            "Later detector centers are pseudo-ground-truth. Use manual center labels "
            "and physical position measurements before robot control."
        ),
        "records": records,
    }


def main() -> None:
    args = parse_args()
    rows = load_targets(args.input)
    result = evaluate(
        rows,
        horizon_s=args.horizon_ms / 1000.0,
        tolerance_s=args.tolerance_ms / 1000.0,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = result["angular_error_deg"]
    print(
        f"matched={result['matched_forecasts']}/{result['input_rows']} "
        f"coverage={100.0 * result['future_detection_coverage']:.1f}% "
        f"angular_median={summary['median']} deg p90={summary['p90']} deg"
    )
    print(f"saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
