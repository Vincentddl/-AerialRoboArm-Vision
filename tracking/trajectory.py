from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


@dataclass
class Detection:
    """One detector result in pixel coordinates."""

    bbox: BBox
    score: float
    class_id: int
    label: str = ""

    @property
    def center(self) -> Point:
        x1, y1, x2, y2 = self.bbox
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


@dataclass
class PendingTrack:
    detection: Detection
    first_center: Point
    first_time: float
    last_time: float
    hits: int = 1
    age: int = 1
    missed: int = 0
    max_displacement: float = 0.0
    history: List[Point] = field(default_factory=list)

    def update(self, detection: Detection, timestamp: float) -> None:
        self.detection = detection
        self.last_time = timestamp
        self.hits += 1
        self.age += 1
        self.missed = 0
        self.max_displacement = max(self.max_displacement, _distance(self.first_center, detection.center))
        self.history.append(detection.center)


@dataclass
class TrackState:
    track_id: int
    label: str
    class_id: int
    bbox: BBox
    score: float
    center: Point
    last_time: float
    velocity: Point = (0.0, 0.0)
    acceleration: Point = (0.0, 0.0)
    age: int = 1
    missed: int = 0
    stationary_frames: int = 0
    history: List[Point] = field(default_factory=list)
    state: np.ndarray = field(init=False, repr=False)
    covariance: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.state = np.asarray(
            [
                self.center[0],
                self.center[1],
                self.velocity[0],
                self.velocity[1],
                self.acceleration[0],
                self.acceleration[1],
            ],
            dtype=np.float64,
        )
        self.covariance = np.diag([100.0, 100.0, 10000.0, 10000.0, 100000.0, 100000.0]).astype(np.float64)

    def predict_to(self, timestamp: float, process_noise: float) -> None:
        dt = max(timestamp - self.last_time, 1e-6)
        transition = np.asarray(
            [
                [1.0, 0.0, dt, 0.0, 0.5 * dt * dt, 0.0],
                [0.0, 1.0, 0.0, dt, 0.0, 0.5 * dt * dt],
                [0.0, 0.0, 1.0, 0.0, dt, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dt2 = dt * dt
        dt3 = dt2 * dt
        noise_gain = np.asarray(
            [
                [dt3 / 6.0, 0.0],
                [0.0, dt3 / 6.0],
                [0.5 * dt2, 0.0],
                [0.0, 0.5 * dt2],
                [dt, 0.0],
                [0.0, dt],
            ],
            dtype=np.float64,
        )
        process = process_noise * noise_gain @ noise_gain.T

        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process
        self.last_time = timestamp
        self._sync_from_state(shift_bbox=True)

    def update(self, detection: Detection, timestamp: float, measurement_noise: float, max_history: int) -> None:
        if timestamp > self.last_time:
            self.predict_to(timestamp, process_noise=1.0)

        observation = np.asarray(detection.center, dtype=np.float64)
        observation_model = np.asarray(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        measurement_covariance = np.eye(2, dtype=np.float64) * measurement_noise

        innovation = observation - observation_model @ self.state
        innovation_covariance = observation_model @ self.covariance @ observation_model.T + measurement_covariance
        kalman_gain = self.covariance @ observation_model.T @ np.linalg.inv(innovation_covariance)
        self.state = self.state + kalman_gain @ innovation
        identity = np.eye(6, dtype=np.float64)
        self.covariance = (identity - kalman_gain @ observation_model) @ self.covariance

        self.bbox = detection.bbox
        self.score = detection.score
        self.label = detection.label
        self.class_id = detection.class_id
        self.age += 1
        self.missed = 0
        self._sync_from_state(shift_bbox=False)
        self.history.append(self.center)
        if len(self.history) > max_history:
            self.history = self.history[-max_history:]

    def mark_missed(self, max_history: int) -> None:
        self.missed += 1
        self.history.append(self.center)
        if len(self.history) > max_history:
            self.history = self.history[-max_history:]

    def predict_pixel(self, seconds: float) -> Point:
        return (
            float(self.state[0] + self.state[2] * seconds + 0.5 * self.state[4] * seconds * seconds),
            float(self.state[1] + self.state[3] * seconds + 0.5 * self.state[5] * seconds * seconds),
        )

    def _sync_from_state(self, shift_bbox: bool) -> None:
        old_center = self.center
        self.center = (float(self.state[0]), float(self.state[1]))
        self.velocity = (float(self.state[2]), float(self.state[3]))
        self.acceleration = (float(self.state[4]), float(self.state[5]))
        if shift_bbox:
            dx = self.center[0] - old_center[0]
            dy = self.center[1] - old_center[1]
            x1, y1, x2, y2 = self.bbox
            self.bbox = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


class PixelToWorldMapper:
    """Maps image pixels to robot/world plane coordinates.

    Supported calibration JSON formats:
    1. Homography:
       {"type": "homography", "matrix": [[...], [...], [...]], "unit": "mm"}
    2. Four or more point pairs:
       {"type": "points", "image_points": [[u,v], ...], "world_points": [[x,y], ...], "unit": "mm"}
    3. Simple scale and offset:
       {"type": "scale_offset", "scale": [sx, sy], "offset": [ox, oy], "unit": "mm"}

    If no calibration is provided, world coordinates are returned as pixels.
    """

    def __init__(self, calibration_path: Optional[str] = None):
        self.unit = "px"
        self._matrix: Optional[np.ndarray] = None
        self._scale: Optional[np.ndarray] = None
        self._offset: Optional[np.ndarray] = None
        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs: Optional[np.ndarray] = None
        self._camera_model: Optional[str] = None
        self._angle_offset = np.zeros(2, dtype=np.float64)
        self.image_size: Optional[Tuple[int, int]] = None

        if calibration_path:
            self.load(calibration_path)

    def load(self, calibration_path: str) -> None:
        data = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        camera_model = data.get("model")
        if camera_model in {"pinhole", "fisheye"}:
            camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
            if camera_matrix.shape != (3, 3):
                raise ValueError("camera_matrix must be 3x3")
            self._camera_matrix = camera_matrix
            self._dist_coeffs = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
            self._camera_model = camera_model
            self._angle_offset = np.asarray(data.get("angle_offset_deg", [0.0, 0.0]), dtype=np.float64)
            self.image_size = tuple(int(value) for value in data["image_size"])
            self.unit = "deg"
            return

        calibration_type = data.get("type", "points")
        self.unit = data.get("unit", "mm")

        if calibration_type == "homography":
            matrix = np.asarray(data["matrix"], dtype=np.float64)
            if matrix.shape != (3, 3):
                raise ValueError("homography matrix must be 3x3")
            self._matrix = matrix
            return

        if calibration_type == "points":
            image_points = np.asarray(data["image_points"], dtype=np.float32)
            world_points = np.asarray(data["world_points"], dtype=np.float32)
            if len(image_points) < 4 or len(world_points) < 4:
                raise ValueError("at least four image/world point pairs are required")
            matrix, _ = cv2.findHomography(image_points, world_points, method=0)
            if matrix is None:
                raise ValueError("failed to compute homography from calibration points")
            self._matrix = matrix.astype(np.float64)
            return

        if calibration_type == "scale_offset":
            self._scale = np.asarray(data.get("scale", [1.0, 1.0]), dtype=np.float64)
            self._offset = np.asarray(data.get("offset", [0.0, 0.0]), dtype=np.float64)
            return

        raise ValueError(f"unsupported calibration type: {calibration_type}")

    def to_world(self, point: Point) -> Point:
        xy = np.asarray(point, dtype=np.float64)
        if self._camera_matrix is not None and self._dist_coeffs is not None:
            src = np.asarray([[xy]], dtype=np.float64)
            if self._camera_model == "fisheye":
                normalized = cv2.fisheye.undistortPoints(src, self._camera_matrix, self._dist_coeffs)[0, 0]
            else:
                normalized = cv2.undistortPoints(src, self._camera_matrix, self._dist_coeffs)[0, 0]
            ray_x, ray_y = float(normalized[0]), float(normalized[1])
            yaw = math.degrees(math.atan2(ray_x, 1.0))
            pitch = math.degrees(math.atan2(ray_y, math.sqrt(1.0 + ray_x * ray_x)))
            angles = np.asarray([yaw, pitch], dtype=np.float64) + self._angle_offset
            return (float(angles[0]), float(angles[1]))
        if self._matrix is not None:
            src = np.asarray([[xy]], dtype=np.float64)
            dst = cv2.perspectiveTransform(src, self._matrix)[0, 0]
            return (float(dst[0]), float(dst[1]))
        if self._scale is not None and self._offset is not None:
            dst = xy * self._scale + self._offset
            return (float(dst[0]), float(dst[1]))
        return (float(xy[0]), float(xy[1]))


class TrajectoryEstimator:
    """Kalman-filter tracker for lightweight detector outputs."""

    def __init__(
        self,
        max_match_distance: float = 80.0,
        max_missed: int = 8,
        max_history: int = 40,
        velocity_alpha: float = 0.55,
        process_noise: float = 250.0,
        measurement_noise: float = 25.0,
        new_track_min_score: float = 0.0,
        new_track_confirmation_frames: int = 1,
        new_track_min_motion: float = 0.0,
        new_track_candidate_max_age: int = 4,
        stationary_speed_threshold: float = 0.0,
        stationary_max_frames: int = 8,
        match_classes: bool = True,
        mapper: Optional[PixelToWorldMapper] = None,
    ):
        self.max_match_distance = max_match_distance
        self.max_missed = max_missed
        self.max_history = max_history
        self.velocity_alpha = velocity_alpha
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.new_track_min_score = new_track_min_score
        self.new_track_confirmation_frames = max(1, new_track_confirmation_frames)
        self.new_track_min_motion = max(0.0, new_track_min_motion)
        self.new_track_candidate_max_age = max(self.new_track_confirmation_frames, new_track_candidate_max_age)
        self.stationary_speed_threshold = max(0.0, stationary_speed_threshold)
        self.stationary_max_frames = max(1, stationary_max_frames)
        self.match_classes = match_classes
        self.mapper = mapper or PixelToWorldMapper()
        self._next_id = 1
        self._next_candidate_id = 1
        self._tracks: Dict[int, TrackState] = {}
        self._pending_tracks: Dict[int, PendingTrack] = {}

    @property
    def tracks(self) -> List[TrackState]:
        return list(self._tracks.values())

    def update(self, detections: Iterable[Detection], timestamp: float) -> List[TrackState]:
        detections = list(detections)
        unmatched_detections = set(range(len(detections)))
        unmatched_tracks = set(self._tracks.keys())

        for track in self._tracks.values():
            track.predict_to(timestamp, process_noise=self.process_noise)

        candidates: List[Tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            for det_index, detection in enumerate(detections):
                if self.match_classes and detection.class_id != track.class_id:
                    continue
                distance = _distance(track.center, detection.center)
                if distance <= self.max_match_distance:
                    candidates.append((distance, track_id, det_index))

        for _, track_id, det_index in sorted(candidates, key=lambda item: item[0]):
            if track_id not in unmatched_tracks or det_index not in unmatched_detections:
                continue
            self._tracks[track_id].update(
                detections[det_index],
                timestamp=timestamp,
                measurement_noise=self.measurement_noise,
                max_history=self.max_history,
            )
            unmatched_tracks.remove(track_id)
            unmatched_detections.remove(det_index)

        for track_id in list(unmatched_tracks):
            self._tracks[track_id].mark_missed(max_history=self.max_history)
            if self._tracks[track_id].missed > self.max_missed:
                del self._tracks[track_id]

        unmatched_pending = set(self._pending_tracks.keys())
        pending_candidates: List[Tuple[float, int, int]] = []
        for candidate_id, pending in self._pending_tracks.items():
            for det_index in unmatched_detections:
                detection = detections[det_index]
                if self.match_classes and detection.class_id != pending.detection.class_id:
                    continue
                distance = _distance(pending.detection.center, detection.center)
                if distance <= self.max_match_distance:
                    pending_candidates.append((distance, candidate_id, det_index))

        promoted = []
        for _, candidate_id, det_index in sorted(pending_candidates, key=lambda item: item[0]):
            if candidate_id not in unmatched_pending or det_index not in unmatched_detections:
                continue
            pending = self._pending_tracks[candidate_id]
            pending.update(detections[det_index], timestamp)
            unmatched_pending.remove(candidate_id)
            unmatched_detections.remove(det_index)
            if (
                pending.hits >= self.new_track_confirmation_frames
                and pending.max_displacement >= self.new_track_min_motion
            ):
                promoted.append(candidate_id)

        for candidate_id in promoted:
            pending = self._pending_tracks.pop(candidate_id)
            dt = max(pending.last_time - pending.first_time, 1e-6)
            velocity = (
                (pending.detection.center[0] - pending.first_center[0]) / dt,
                (pending.detection.center[1] - pending.first_center[1]) / dt,
            )
            self._create_track(
                pending.detection,
                timestamp=pending.last_time,
                velocity=velocity,
                age=pending.hits,
                history=pending.history,
            )

        for candidate_id in list(unmatched_pending):
            pending = self._pending_tracks[candidate_id]
            pending.age += 1
            pending.missed += 1
            if pending.missed > 1 or pending.age >= self.new_track_candidate_max_age:
                del self._pending_tracks[candidate_id]

        for candidate_id in list(self._pending_tracks):
            pending = self._pending_tracks[candidate_id]
            if pending.age >= self.new_track_candidate_max_age:
                del self._pending_tracks[candidate_id]

        for det_index in sorted(unmatched_detections):
            detection = detections[det_index]
            if detection.score < self.new_track_min_score:
                continue
            if self.new_track_confirmation_frames == 1 and self.new_track_min_motion == 0.0:
                self._create_track(detection, timestamp=timestamp)
                continue
            self._pending_tracks[self._next_candidate_id] = PendingTrack(
                detection=detection,
                first_center=detection.center,
                first_time=timestamp,
                last_time=timestamp,
                history=[detection.center],
            )
            self._next_candidate_id += 1

        if self.stationary_speed_threshold > 0.0:
            for track_id in list(self._tracks):
                track = self._tracks[track_id]
                speed = math.hypot(track.velocity[0], track.velocity[1])
                if speed < self.stationary_speed_threshold:
                    track.stationary_frames += 1
                else:
                    track.stationary_frames = 0
                if track.stationary_frames >= self.stationary_max_frames:
                    del self._tracks[track_id]

        return self.tracks

    def _create_track(
        self,
        detection: Detection,
        timestamp: float,
        velocity: Point = (0.0, 0.0),
        age: int = 1,
        history: Optional[List[Point]] = None,
    ) -> None:
        track = TrackState(
            track_id=self._next_id,
            label=detection.label,
            class_id=detection.class_id,
            bbox=detection.bbox,
            score=detection.score,
            center=detection.center,
            last_time=timestamp,
            velocity=velocity,
            age=age,
            history=list(history or [detection.center]),
        )
        self._tracks[self._next_id] = track
        self._next_id += 1

    def arm_targets(
        self,
        predict_seconds: float = 0.5,
        min_age: int = 3,
        max_missed: int = 0,
        min_speed: float = 0.0,
    ) -> List[dict]:
        targets = []
        for track in self.tracks:
            if track.age < min_age or track.missed > max_missed:
                continue
            if math.hypot(track.velocity[0], track.velocity[1]) < min_speed:
                continue
            predicted_pixel = track.predict_pixel(predict_seconds)
            current_world = self.mapper.to_world(track.center)
            predicted_world = self.mapper.to_world(predicted_pixel)
            targets.append(
                {
                    "track_id": track.track_id,
                    "label": track.label,
                    "class_id": track.class_id,
                    "score": round(float(track.score), 4),
                    "age": track.age,
                    "missed_frames": track.missed,
                    "pixel": _round_point(track.center),
                    "velocity_px_s": _round_point(track.velocity),
                    "acceleration_px_s2": _round_point(track.acceleration),
                    "predicted_pixel": _round_point(predicted_pixel),
                    "world": _round_point(current_world),
                    "predicted_world": _round_point(predicted_world),
                    "world_unit": self.mapper.unit,
                    "predict_seconds": predict_seconds,
                }
            )
            if self.mapper.unit == "deg":
                targets[-1]["angle_deg"] = _round_point(current_world)
                targets[-1]["predicted_angle_deg"] = _round_point(predicted_world)
        return targets


def draw_tracks(
    image: np.ndarray,
    tracks: Sequence[TrackState],
    mapper: PixelToWorldMapper,
    predict_seconds: float = 0.5,
    min_age: int = 3,
) -> np.ndarray:
    for track in tracks:
        color = _track_color(track.track_id)
        x1, y1, x2, y2 = [int(v) for v in track.bbox]
        cx, cy = [int(v) for v in track.center]

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.circle(image, (cx, cy), 4, color, -1)

        if len(track.history) >= 2:
            points = np.asarray(track.history, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(image, [points], isClosed=False, color=color, thickness=2)

        coast = f" coast={track.missed}" if track.missed else ""
        line1 = (
            f"ID {track.track_id} {track.label}{coast} "
            f"v=({track.velocity[0]:.0f},{track.velocity[1]:.0f})px/s"
        )
        if track.age >= min_age:
            px, py = [int(v) for v in track.predict_pixel(predict_seconds)]
            world_x, world_y = mapper.to_world(track.predict_pixel(predict_seconds))
            cv2.circle(image, (px, py), 5, (0, 0, 255), -1)
            cv2.arrowedLine(image, (cx, cy), (px, py), (0, 0, 255), 2, tipLength=0.25)
            line2 = (
                f"a=({track.acceleration[0]:.0f},{track.acceleration[1]:.0f})px/s2 "
                f"+{predict_seconds:.2f}s=({world_x:.1f},{world_y:.1f}){mapper.unit}"
            )
        else:
            line2 = f"collecting trajectory {track.age}/{min_age} frames"
        text_y = max(20, y1 - 25)
        cv2.putText(image, line1, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.putText(image, line2, (x1, text_y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return image


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _round_point(point: Point) -> List[float]:
    return [round(float(point[0]), 3), round(float(point[1]), 3)]


def _track_color(track_id: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(track_id)
    return tuple(int(v) for v in rng.integers(64, 255, size=3))
