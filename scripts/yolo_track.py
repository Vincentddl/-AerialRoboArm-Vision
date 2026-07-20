import argparse
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
LAB_DIR = SCRIPT_DIR.parent
TRACKING_DIR = LAB_DIR / "tracking"
MODELS_DIR = LAB_DIR / "models"
OUTPUTS_DIR = LAB_DIR / "outputs"

sys.path.insert(0, str(TRACKING_DIR))

from trajectory import Detection, PixelToWorldMapper, TrajectoryEstimator, draw_tracks  # noqa: E402
from ultralytics import YOLO  # noqa: E402


DEFAULT_MODEL = MODELS_DIR / "yolo11s.pt"
FALLBACK_MODEL = MODELS_DIR / "reference" / "yolo11s.duplicate-root-copy.pt"


def resolve_path(path):
    if not path:
        return ""
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return str(candidate)
    return str(LAB_DIR / candidate)


def load_model(model_path):
    model_path = Path(model_path)
    default_requested = model_path == DEFAULT_MODEL
    if not model_path.exists() and default_requested and DEFAULT_MODEL.exists():
        model_path = DEFAULT_MODEL
    if not model_path.exists() and default_requested and FALLBACK_MODEL.exists():
        model_path = FALLBACK_MODEL
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")
    print(f"Using model: {model_path}")
    return YOLO(str(model_path))


def resolve_device(device):
    if not device:
        return None
    normalized = device.lower()
    wants_cuda = normalized.isdigit() or normalized.startswith("cuda")
    if wants_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "You requested GPU inference, but this Python environment cannot see CUDA.\n"
            f"torch version: {torch.__version__}\n"
            f"torch.version.cuda: {torch.version.cuda}\n"
            "Install a CUDA-enabled PyTorch build, then retry with --device 0.\n"
            "For now, run with --device cpu or omit --device."
        )
    return device


def parse_class_filter(class_filter, names):
    if not class_filter:
        return None

    allowed = set()
    class_items = [item.strip().lower() for item in class_filter.split(",") if item.strip()]
    normalized_names = {idx: str(name).lower() for idx, name in names.items()}

    for item in class_items:
        if item.isdigit():
            allowed.add(int(item))
            continue

        exact_matches = [idx for idx, name in normalized_names.items() if name == item]
        partial_matches = [idx for idx, name in normalized_names.items() if item in name]
        matches = exact_matches or partial_matches
        if not matches:
            examples = ", ".join(str(name) for _, name in list(names.items())[:12])
            raise ValueError(f"unknown class '{item}'. Example classes: {examples} ...")
        allowed.update(matches)

    print("Class filter:", ", ".join(f"{idx}:{names[idx]}" for idx in sorted(allowed)))
    return allowed


CAMERA_BACKENDS = {
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "any": cv2.CAP_ANY,
}


def open_source(source, backend="dshow"):
    if source.isdigit():
        candidates = (
            [("dshow", cv2.CAP_DSHOW), ("msmf", cv2.CAP_MSMF), ("any", cv2.CAP_ANY)]
            if backend == "auto"
            else [(backend, CAMERA_BACKENDS[backend])]
        )
        for backend_name, backend_id in candidates:
            cap = cv2.VideoCapture(int(source), backend_id)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                print(f"Camera source {source} opened with {backend_name}")
                return cap
            cap.release()
        return cap

    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class LatestFrameCapture:
    """Read camera frames in the background and always expose the newest frame."""

    def __init__(self, source, width=0, height=0, backend="dshow"):
        self.cap = open_source(source, backend=backend)
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"failed to open source: {source}")

        self.lock = threading.Lock()
        self.running = True
        self.frame = None
        self.timestamp = 0.0
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self.lock:
                self.frame = frame
                self.timestamp = time.perf_counter()

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, 0.0, None
            return True, self.timestamp, self.frame.copy()

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


class SequentialVideoCapture:
    """Read every frame from a finite video and use video time instead of inference time."""

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"failed to open video: {source}")
        reported_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.fps = reported_fps if reported_fps > 1 else 25.0
        self.frame_index = 0
        self.timestamps = None

        timestamp_path = Path(source).with_suffix(".timestamps.jsonl")
        if timestamp_path.exists():
            values = []
            for line in timestamp_path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                values.append(float(record["monotonic_seconds"]))
            if len(values) >= 2:
                start = values[0]
                self.timestamps = [value - start for value in values]
                duration = self.timestamps[-1]
                if duration > 0:
                    self.fps = (len(self.timestamps) - 1) / duration
                print(f"Using frame timestamps: {timestamp_path} ({self.fps:.3f} FPS effective)")

    def read(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return False, 0.0, None
        if self.timestamps is not None and self.frame_index < len(self.timestamps):
            timestamp = self.timestamps[self.frame_index]
        else:
            timestamp = self.frame_index / self.fps
        self.frame_index += 1
        return True, timestamp, frame

    def release(self):
        self.cap.release()


def result_to_detections(result, allowed_classes=None):
    detections = []
    names = result.names
    boxes = result.boxes
    if boxes is None:
        return detections

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)

    for bbox, score, class_id in zip(xyxy, conf, cls):
        if allowed_classes is not None and class_id not in allowed_classes:
            continue
        detections.append(
            Detection(
                bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                score=float(score),
                class_id=int(class_id),
                label=str(names.get(int(class_id), class_id)),
            )
        )
    return detections


def draw_detection_centers(image, detections):
    for detection in detections:
        cx, cy = [int(v) for v in detection.center]
        cv2.drawMarker(
            image,
            (cx, cy),
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=16,
            thickness=2,
        )
        cv2.putText(
            image,
            f"C({cx},{cy})",
            (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
    return image


def main():
    parser = argparse.ArgumentParser(description="YOLO trash detection with trajectory prediction")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to trained YOLO .pt model")
    parser.add_argument("--source", default="0", help="Camera index, video file, image file, or stream URL")
    parser.add_argument("--classes", default="", help="Class filter, e.g. bottle, Clear plastic bottle, 7, 21")
    parser.add_argument("--list-classes", action="store_true", help="Print model classes and exit")
    parser.add_argument("--conf", type=float, default=0.2, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=512, help="YOLO inference image size")
    parser.add_argument("--device", default="", help="Device, e.g. cpu, 0, cuda:0. Empty means auto")
    parser.add_argument("--predict-seconds", type=float, default=0.2, help="Lead time for short-term prediction")
    parser.add_argument("--min-age", type=int, default=1, help="Frames required before a track is sent as an arm target")
    parser.add_argument("--max-match-distance", type=float, default=120.0, help="Max pixel distance for ID matching")
    parser.add_argument("--max-missed", type=int, default=15, help="Drop a track after this many missed frames")
    parser.add_argument("--process-noise", type=float, default=250.0, help="Kalman process noise; higher follows acceleration faster")
    parser.add_argument("--measurement-noise", type=float, default=25.0, help="Kalman measurement noise; higher smooths detector jitter more")
    parser.add_argument(
        "--new-track-conf",
        type=float,
        default=0.0,
        help="Minimum confidence required to create a new track; lower-confidence detections may still continue one",
    )
    parser.add_argument(
        "--new-track-confirm-frames",
        type=int,
        default=1,
        help="Matched frames required before a detection candidate becomes a track",
    )
    parser.add_argument(
        "--new-track-min-motion",
        type=float,
        default=0.0,
        help="Minimum pixel displacement required before a detection candidate becomes a track",
    )
    parser.add_argument(
        "--new-track-candidate-max-age",
        type=int,
        default=4,
        help="Maximum frames allowed for a candidate to satisfy track confirmation",
    )
    parser.add_argument(
        "--min-target-speed",
        type=float,
        default=0.0,
        help="Minimum track speed in pixels/second before output to the arm",
    )
    parser.add_argument(
        "--stationary-max-frames",
        type=int,
        default=8,
        help="Drop a track after this many consecutive frames below --min-target-speed",
    )
    parser.add_argument(
        "--coast-frames",
        type=int,
        default=0,
        help="Continue output from the motion model for this many consecutive detector misses",
    )
    parser.add_argument("--class-agnostic-tracking", action="store_true", help="Keep the same ID even if YOLO class changes between similar trash classes")
    parser.add_argument("--camera-width", type=int, default=640, help="Requested camera width")
    parser.add_argument("--camera-height", type=int, default=480, help="Requested camera height")
    parser.add_argument(
        "--camera-backend",
        choices=["auto", "dshow", "msmf", "any"],
        default="dshow",
        help="Windows camera backend for numeric sources",
    )
    parser.add_argument("--skip-unchanged", action="store_true", help="Skip duplicate camera frames from the latest-frame reader")
    parser.add_argument("--hide-centers", action="store_true", help="Hide immediate detection centroids")
    parser.add_argument("--print-targets", action="store_true", help="Print target JSON to terminal")
    parser.add_argument("--calibration", default="", help="Optional pixel-to-robot-plane calibration JSON")
    parser.add_argument("--save-jsonl", default="", help="Optional path to append arm target JSON lines")
    parser.add_argument("--output", default="", help="Optional output image/video path")
    parser.add_argument("--no-window", action="store_true", help="Run without cv2.imshow")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model = load_model(resolve_path(args.model))
    if args.list_classes:
        for class_id, name in model.names.items():
            print(f"{class_id}: {name}")
        return

    allowed_classes = parse_class_filter(args.classes, model.names)
    mapper = PixelToWorldMapper(resolve_path(args.calibration) if args.calibration else None)
    estimator = TrajectoryEstimator(
        max_match_distance=args.max_match_distance,
        max_missed=args.max_missed,
        process_noise=args.process_noise,
        measurement_noise=args.measurement_noise,
        new_track_min_score=args.new_track_conf,
        new_track_confirmation_frames=args.new_track_confirm_frames,
        new_track_min_motion=args.new_track_min_motion,
        new_track_candidate_max_age=args.new_track_candidate_max_age,
        stationary_speed_threshold=args.min_target_speed,
        stationary_max_frames=args.stationary_max_frames,
        match_classes=not args.class_agnostic_tracking,
        mapper=mapper,
    )

    source = args.source
    source_path = Path(source)
    is_image = source_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    is_video = source_path.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv", ".m4v", ".wmv", ".webm"}
    capture = None
    writer = None
    calibration_size_checked = False
    inference_device_reported = False
    jsonl_file = open(args.save_jsonl, "a", encoding="utf-8") if args.save_jsonl else None

    try:
        if is_image:
            frame = cv2.imread(source)
            if frame is None:
                raise RuntimeError(f"failed to read image: {source}")
            frames = [(time.perf_counter(), frame)]
        elif is_video:
            capture = SequentialVideoCapture(source)
            frames = None
        else:
            capture = LatestFrameCapture(
                source,
                width=args.camera_width,
                height=args.camera_height,
                backend=args.camera_backend,
            )
            frames = None

        last_timestamp = None
        while True:
            if frames is None:
                ok, timestamp, frame = capture.read()
                if not ok:
                    if is_video:
                        break
                    time.sleep(0.005)
                    continue
                if args.skip_unchanged and last_timestamp is not None and timestamp == last_timestamp:
                    time.sleep(0.001)
                    continue
                last_timestamp = timestamp
                if frame is None:
                    break
            else:
                if not frames:
                    break
                timestamp, frame = frames.pop(0)

            if not calibration_size_checked and mapper.image_size is not None:
                actual_size = (frame.shape[1], frame.shape[0])
                if actual_size != mapper.image_size:
                    raise RuntimeError(
                        f"calibration expects {mapper.image_size[0]}x{mapper.image_size[1]}, "
                        f"but source is {actual_size[0]}x{actual_size[1]}"
                    )
                print(f"Camera calibration: {mapper.image_size[0]}x{mapper.image_size[1]} ({mapper.unit})")
                calibration_size_checked = True

            start = time.perf_counter()
            result = model.predict(
                source=frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=device,
                verbose=False,
            )[0]
            if not inference_device_reported:
                print(f"Inference device: {next(model.model.parameters()).device}")
                inference_device_reported = True
            detections = result_to_detections(result, allowed_classes)
            tracks = estimator.update(detections, timestamp)
            arm_targets = estimator.arm_targets(
                predict_seconds=args.predict_seconds,
                min_age=args.min_age,
                max_missed=args.coast_frames,
                min_speed=args.min_target_speed,
            )
            active_tracks = [track for track in tracks if track.missed <= args.coast_frames]
            infer_ms = (time.perf_counter() - start) * 1000.0

            if jsonl_file and arm_targets:
                jsonl_file.write(json.dumps({"time": timestamp, "targets": arm_targets}, ensure_ascii=False) + "\n")
                jsonl_file.flush()

            if args.print_targets and arm_targets:
                print(json.dumps({"time": round(timestamp, 3), "targets": arm_targets}, ensure_ascii=False))

            vis = draw_tracks(
                frame.copy(),
                active_tracks,
                mapper,
                predict_seconds=args.predict_seconds,
                min_age=args.min_age,
            )
            if not args.hide_centers:
                vis = draw_detection_centers(vis, detections)
            cv2.putText(
                vis,
                f"YOLO trajectory {infer_ms:.1f}ms | det {len(detections)} | tracks {len(tracks)}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            if args.output and not is_image:
                if writer is None:
                    h, w = vis.shape[:2]
                    fps = getattr(capture, "fps", capture.cap.get(cv2.CAP_PROP_FPS)) if capture is not None else 25
                    fallback_fps = 30.0 if args.source.isdigit() else 25.0
                    fps = fps if fps and fps > 1 else fallback_fps
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
                writer.write(vis)

            if not args.no_window:
                window_name = "YOLO Object Trajectory"
                cv2.imshow(window_name, vis)
                key = cv2.waitKey(0 if is_image else 1) & 0xFF
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key in (27, ord("q")):
                    break

            if is_image:
                if args.output:
                    cv2.imwrite(args.output, vis)
                break
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if jsonl_file is not None:
            jsonl_file.close()
        if not args.no_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
