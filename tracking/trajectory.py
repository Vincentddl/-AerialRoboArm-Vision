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
            ],
            dtype=np.float64,
        )
        self.covariance = np.diag([100.0, 100.0, 10000.0, 10000.0]).astype(np.float64)

    def predict_to(
        self, timestamp: float, process_noise_x: float, process_noise_y: float | None = None
    ) -> None:
        if process_noise_y is None:
            process_noise_y = process_noise_x
        dt = max(timestamp - self.last_time, 1e-6)
        # Constant-velocity transition: x ← x + v*dt, v unchanged
        transition = np.asarray(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dt2 = dt * dt
        # Asymmetric process noise: px for x-dim, py for y-dim (vertical)
        noise_gain = np.asarray(
            [
                [0.5 * dt2, 0.0],
                [0.0, 0.5 * dt2],
                [dt, 0.0],
                [0.0, dt],
            ],
            dtype=np.float64,
        )
        # Asymmetric process noise: px for x, py for y (vertical)
        Q_2d = np.diag([float(process_noise_x), float(process_noise_y)]).astype(np.float64)
        process = noise_gain @ Q_2d @ noise_gain.T

        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process
        self.last_time = timestamp
        self._sync_from_state(shift_bbox=True)

    def update(self, detection: Detection, timestamp: float, measurement_noise: float, max_history: int) -> None:
        if timestamp > self.last_time:
            self.predict_to(timestamp, process_noise_x=1.0)

        observation = np.asarray(detection.center, dtype=np.float64)
        observation_model = np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        measurement_covariance = np.eye(2, dtype=np.float64) * measurement_noise

        innovation = observation - observation_model @ self.state
        innovation_covariance = observation_model @ self.covariance @ observation_model.T + measurement_covariance
        kalman_gain = self.covariance @ observation_model.T @ np.linalg.inv(innovation_covariance)
        self.state = self.state + kalman_gain @ innovation
        identity = np.eye(4, dtype=np.float64)
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
        """Predict future pixel position assuming constant velocity.

        Simple linear extrapolation — no acceleration term to drift.
        Well-suited for short horizons (≤ 0.4 s) and pure vertical motion.
        """
        t = float(seconds)
        return (
            float(self.state[0] + self.state[2] * t),
            float(self.state[1] + self.state[3] * t),
        )

    def predict_trajectory(self, seconds: float, steps: int = 8) -> List[Point]:
        """Return a sampled future path, including now and the requested endpoint."""
        if seconds < 0:
            raise ValueError("prediction horizon must be non-negative")
        sample_count = max(1, int(steps))
        return [
            self.predict_pixel(seconds * index / sample_count)
            for index in range(sample_count + 1)
        ]

    def _sync_from_state(self, shift_bbox: bool) -> None:
        old_center = self.center
        self.center = (float(self.state[0]), float(self.state[1]))
        self.velocity = (float(self.state[2]), float(self.state[3]))
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

    Camera calibration formats (pinhole / fisheye) map pixels to bearing
    angles (yaw, pitch) in degrees via undistortion.

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
        self._reprojection_error_px: Optional[float] = None

        if calibration_path:
            self.load(calibration_path)

    def load(self, calibration_path: str) -> None:
        data = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        self._reprojection_error_px = data.get("reprojection_error_px")
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

    def to_pixel(self, angles: Point) -> Point:
        """Reverse-map bearing angles ``(yaw_deg, pitch_deg)`` back to pixels.

        Only supported for camera (pinhole / fisheye) calibrations.
        Inverse of :meth:`to_world` for those models.
        """
        if self._camera_matrix is None or self._dist_coeffs is None:
            raise RuntimeError(
                "to_pixel requires a camera calibration (pinhole or fisheye)"
            )
        raw_yaw = angles[0] - float(self._angle_offset[0])
        raw_pitch = angles[1] - float(self._angle_offset[1])
        yaw_rad = math.radians(raw_yaw)
        pitch_rad = math.radians(raw_pitch)
        ray = np.asarray(
            [
                math.sin(yaw_rad) * math.cos(pitch_rad),
                math.sin(pitch_rad),
                math.cos(yaw_rad) * math.cos(pitch_rad),
            ],
            dtype=np.float64,
        )
        ray = ray / np.linalg.norm(ray)
        point = (ray / ray[2]).reshape(1, 1, 3)
        zero_rvec = np.zeros((3, 1), dtype=np.float64)
        zero_tvec = np.zeros((3, 1), dtype=np.float64)
        if self._camera_model == "fisheye":
            projected, _ = cv2.fisheye.projectPoints(
                point, zero_rvec, zero_tvec, self._camera_matrix, self._dist_coeffs
            )
        else:
            projected, _ = cv2.projectPoints(
                point, zero_rvec, zero_tvec, self._camera_matrix, self._dist_coeffs
            )
        pixel = projected[0, 0]
        return (float(pixel[0]), float(pixel[1]))

    def is_visible(self, angles: Point, margin_px: float = 0.0) -> bool:
        """Return whether a camera bearing projects into the calibrated image."""
        if self._camera_matrix is None or self._dist_coeffs is None or self.image_size is None:
            return False
        yaw = math.radians(angles[0] - float(self._angle_offset[0]))
        pitch = math.radians(angles[1] - float(self._angle_offset[1]))
        if math.cos(yaw) * math.cos(pitch) <= 0.0:
            return False
        u, v = self.to_pixel(angles)
        width, height = self.image_size
        return (
            math.isfinite(u)
            and math.isfinite(v)
            and -margin_px <= u < width + margin_px
            and -margin_px <= v < height + margin_px
        )

    def optical_axis_offset_deg(self, angles: Point) -> float:
        """Return the 3-D angle between a bearing ray and the camera optical axis."""
        if self._camera_matrix is None or self._dist_coeffs is None:
            raise RuntimeError(
                "optical-axis offset requires a camera calibration"
            )
        yaw = math.radians(angles[0] - float(self._angle_offset[0]))
        pitch = math.radians(angles[1] - float(self._angle_offset[1]))
        optical_axis_dot = math.cos(yaw) * math.cos(pitch)
        return math.degrees(
            math.acos(float(np.clip(optical_axis_dot, -1.0, 1.0)))
        )

    @property
    def principal_point(self) -> Optional[Point]:
        """Camera principal point ``(cx, cy)`` in pixels, if calibrated."""
        if self._camera_matrix is not None:
            return (
                float(self._camera_matrix[0, 2]),
                float(self._camera_matrix[1, 2]),
            )
        return None

    @property
    def calibration_info(self) -> str:
        """Human-readable calibration summary for logging."""
        if self._camera_model is not None and self._reprojection_error_px is not None:
            size = f"{self.image_size[0]}x{self.image_size[1]}" if self.image_size else "?"
            return (
                f"camera calibration: {self._camera_model} {size} "
                f"{self._reprojection_error_px:.2f} px reproj error "
                f"(lens distortion already corrected in pixel→angle mapping)"
            )
        if self._camera_model is not None:
            size = f"{self.image_size[0]}x{self.image_size[1]}" if self.image_size else "?"
            return f"camera calibration: {self._camera_model} {size} (distortion corrected)"
        return "no camera calibration loaded"


class TrajectoryEstimator:
    """Kalman-filter tracker for lightweight detector outputs."""

    def __init__(
        self,
        max_match_distance: float = 80.0,
        max_missed: int = 8,
        max_history: int = 40,
        velocity_alpha: float = 0.55,
        process_noise: float = 250.0,
        process_noise_y: float | None = None,
        measurement_noise: float = 25.0,
        new_track_min_score: float = 0.0,
        new_track_confirmation_frames: int = 1,
        new_track_min_motion: float = 0.0,
        new_track_candidate_max_age: int = 4,
        stationary_speed_threshold: float = 0.0,
        stationary_max_frames: int = 8,
        match_classes: bool = True,
        mapper: Optional[PixelToWorldMapper] = None,
        ensemble_predictor=None,
    ):
        self.max_match_distance = max_match_distance
        self.max_missed = max_missed
        self.max_history = max_history
        self.velocity_alpha = velocity_alpha
        self.process_noise = process_noise
        self.process_noise_y = process_noise_y if process_noise_y is not None else process_noise
        self.measurement_noise = measurement_noise
        self.new_track_min_score = new_track_min_score
        self.new_track_confirmation_frames = max(1, new_track_confirmation_frames)
        self.new_track_min_motion = max(0.0, new_track_min_motion)
        self.new_track_candidate_max_age = max(self.new_track_confirmation_frames, new_track_candidate_max_age)
        self.stationary_speed_threshold = max(0.0, stationary_speed_threshold)
        self.stationary_max_frames = max(1, stationary_max_frames)
        self.match_classes = match_classes
        self.mapper = mapper or PixelToWorldMapper()
        self._ensemble_predictor = ensemble_predictor
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
            track.predict_to(timestamp, process_noise_x=self.process_noise, process_noise_y=self.process_noise_y)

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
                self._cleanup_track(track_id)
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
                    self._cleanup_track(track_id)
                    del self._tracks[track_id]

        # Only detector-corrected states are observations. Feeding a missed
        # track's Kalman extrapolation back into the predictor causes drift.
        if self._ensemble_predictor is not None:
            for track in self._tracks.values():
                if track.missed == 0:
                    self._ensemble_predictor.update_with_track(track, timestamp)

        return self.tracks

    def _cleanup_track(self, track_id: int) -> None:
        """Notify ensemble predictor (if any) that a track was dropped."""
        if self._ensemble_predictor is not None:
            self._ensemble_predictor.cleanup_track(track_id)

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
        predict_seconds: float = 0.4,
        min_age: int = 3,
        max_missed: int = 0,
        min_speed: float = 0.0,
        trajectory_steps: int = 8,
        camera_elevation_deg: float = 0.0,
        max_prediction_uncertainty_deg: float = 8.0,
    ) -> List[dict]:
        targets = []
        for track in self.tracks:
            if track.age < min_age or track.missed > max_missed:
                continue
            if math.hypot(track.velocity[0], track.velocity[1]) < min_speed:
                continue
            trajectory_pixel: List[Optional[Point]] = list(
                track.predict_trajectory(predict_seconds, trajectory_steps)
            )
            predicted_pixel: Optional[Point] = trajectory_pixel[-1]
            current_world = self.mapper.to_world(track.center)

            # --- angle / world prediction --------------------------------
            prediction_method = "pixel_kalman"
            prediction_uncertainty_deg: Optional[float] = None
            prediction_error: Optional[str] = None
            prediction_sources: Optional[dict] = None
            if self._ensemble_predictor is not None and self.mapper.unit == "deg":
                try:
                    ensemble = self._ensemble_predictor.predict(
                        track, predict_seconds, trajectory_steps
                    )
                    predicted_world = ensemble["predicted_angle"]
                    trajectory_world = ensemble["trajectory"]
                    prediction_method = ensemble["method"]
                    prediction_uncertainty_deg = ensemble.get("uncertainty_deg")
                    prediction_sources = ensemble.get("sources")
                    trajectory_pixel = []
                    for angles in trajectory_world:
                        if self.mapper.is_visible(angles, margin_px=640.0):
                            trajectory_pixel.append(self.mapper.to_pixel(angles))
                        else:
                            trajectory_pixel.append(None)
                    predicted_pixel = trajectory_pixel[-1]
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                    # Fall back to pixel-Kalman-converted angles
                    if predicted_pixel is None:
                        raise RuntimeError("pixel Kalman prediction is unavailable") from exc
                    predicted_world = self.mapper.to_world(predicted_pixel)
                    trajectory_world = [
                        self.mapper.to_world(point)
                        for point in trajectory_pixel
                        if point is not None
                    ]
                    prediction_method = "pixel_kalman (fallback)"
                    prediction_error = str(exc)
            else:
                if predicted_pixel is None:
                    continue
                predicted_world = self.mapper.to_world(predicted_pixel)
                trajectory_world = [
                    self.mapper.to_world(point)
                    for point in trajectory_pixel
                    if point is not None
                ]

            predicted_visible = (
                self.mapper.is_visible(predicted_world)
                if self.mapper.unit == "deg"
                else predicted_pixel is not None
            )
            uncertainty_ok = (
                prediction_uncertainty_deg is None
                or prediction_uncertainty_deg <= max_prediction_uncertainty_deg
            )
            prediction_valid = (
                track.missed == 0
                and predicted_pixel is not None
                and predicted_visible
                and uncertainty_ok
                and prediction_error is None
            )

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
                    "predicted_pixel": (
                        _round_point(predicted_pixel) if predicted_pixel is not None else None
                    ),
                    "world": _round_point(current_world),
                    "predicted_world": _round_point(predicted_world),
                    "world_unit": self.mapper.unit,
                    "predict_seconds": predict_seconds,
                    "trajectory_pixel": [
                        _round_point(point) if point is not None else None
                        for point in trajectory_pixel
                    ],
                    "trajectory_world": [_round_point(point) for point in trajectory_world],
                    "prediction_method": prediction_method,
                    "prediction_valid": prediction_valid,
                    "predicted_visible": predicted_visible,
                    "prediction_uncertainty_deg": (
                        round(float(prediction_uncertainty_deg), 3)
                        if prediction_uncertainty_deg is not None
                        else None
                    ),
                    "prediction_sources": prediction_sources,
                }
            )
            if prediction_error is not None:
                targets[-1]["prediction_error"] = prediction_error
            if self.mapper.unit == "deg":
                targets[-1]["angle_deg"] = _round_point(current_world)
                targets[-1]["predicted_angle_deg"] = _round_point(predicted_world)
                targets[-1]["predicted_angular_displacement_deg"] = _round_point(
                    (
                        predicted_world[0] - current_world[0],
                        predicted_world[1] - current_world[1],
                    )
                )
                current_axis_offset = self.mapper.optical_axis_offset_deg(
                    current_world
                )
                predicted_axis_offset = self.mapper.optical_axis_offset_deg(
                    predicted_world
                )
                targets[-1]["yaw_offset_deg"] = round(float(current_world[0]), 3)
                targets[-1]["predicted_yaw_offset_deg"] = round(
                    float(predicted_world[0]), 3
                )
                targets[-1]["offset_angle_deg"] = round(current_axis_offset, 3)
                targets[-1]["predicted_offset_angle_deg"] = round(
                    predicted_axis_offset, 3
                )
                targets[-1]["offset_angle_convention"] = (
                    "unsigned 3-D angle from camera optical axis"
                )
        return targets


def draw_tracks(
    image: np.ndarray,
    tracks: Sequence[TrackState],
    mapper: PixelToWorldMapper,
    predict_seconds: float = 0.4,
    min_age: int = 3,
    trajectory_steps: int = 8,
    targets: Optional[Sequence[dict]] = None,
) -> np.ndarray:
    text_color = (0, 0, 255)  # red
    font = cv2.FONT_HERSHEY_DUPLEX
    targets_by_id = {
        int(target["track_id"]): target for target in (targets or [])
    }

    for track in tracks:
        color = _track_color(track.track_id)
        x1, y1, x2, y2 = [int(v) for v in track.bbox]
        cx, cy = [int(v) for v in track.center]

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.circle(image, (cx, cy), 4, color, -1)

        if len(track.history) >= 2:
            points = np.asarray(track.history, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(image, [points], isClosed=False, color=color, thickness=2)

        if track.age >= min_age:
            target = targets_by_id.get(track.track_id)
            raw_path = (
                target.get("trajectory_pixel", [])
                if target is not None
                else track.predict_trajectory(predict_seconds, trajectory_steps)
            )
            future_path = [
                (float(point[0]), float(point[1]))
                for point in raw_path
                if point is not None
            ]
            if len(future_path) >= 2:
                future_points = np.asarray(future_path, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(
                    image,
                    [future_points],
                    isClosed=False,
                    color=(0, 0, 255),
                    thickness=2,
                )
                px, py = [int(v) for v in future_path[-1]]
                if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
                    cv2.circle(image, (px, py), 5, (0, 0, 255), -1)
                    cv2.arrowedLine(
                        image, (cx, cy), (px, py), (0, 0, 255), 2, tipLength=0.25
                    )
            if (
                target is not None
                and "offset_angle_deg" in target
                and "predicted_offset_angle_deg" in target
            ):
                valid_text = "OK" if target.get("prediction_valid") else "INVALID"
                line1 = f"axis_offset={target['offset_angle_deg']:.1f}deg"
                line2 = (
                    f"axis_offset@+{predict_seconds:.1f}s="
                    f"{target['predicted_offset_angle_deg']:.1f}deg {valid_text}"
                )
            else:
                line1 = "axis_offset=calculating..."
                line2 = f"axis_offset@+{predict_seconds:.1f}s=calculating..."
        else:
            line1 = "axis_offset=calculating..."
            line2 = f"axis_offset@+{predict_seconds:.1f}s=calculating..."
        text_y = max(20, y1 - 25)
        cv2.putText(image, line1, (x1, text_y), font, 0.55, text_color, 2)
        cv2.putText(image, line2, (x1, text_y + 20), font, 0.55, text_color, 2)
    return image


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _round_point(point: Point) -> List[float]:
    return [round(float(point[0]), 3), round(float(point[1]), 3)]


def _track_color(track_id: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(track_id)
    return tuple(int(v) for v in rng.integers(64, 255, size=3))
