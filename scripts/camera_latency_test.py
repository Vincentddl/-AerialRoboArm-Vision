from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np


BACKENDS = {
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "any": cv2.CAP_ANY,
}


def parse_source(value: str) -> Union[int, str]:
    return int(value) if value.isdigit() else value


def backend_candidates(name: str):
    if name == "auto":
        return [("dshow", cv2.CAP_DSHOW), ("msmf", cv2.CAP_MSMF), ("any", cv2.CAP_ANY)]
    return [(name, BACKENDS[name])]


def open_camera(source: str, backend: str, width: int, height: int, fps: float):
    parsed_source = parse_source(source)
    if not isinstance(parsed_source, int):
        cap = cv2.VideoCapture(parsed_source)
        if not cap.isOpened():
            raise RuntimeError(f"failed to open source: {source}")
        return cap, "file"

    errors = []
    for backend_name, backend_id in backend_candidates(backend):
        cap = cv2.VideoCapture(parsed_source, backend_id)
        if not cap.isOpened():
            cap.release()
            errors.append(backend_name)
            continue

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps > 0:
            cap.set(cv2.CAP_PROP_FPS, fps)

        ok, frame = cap.read()
        if ok and frame is not None:
            return cap, backend_name
        cap.release()
        errors.append(backend_name)

    tried = ", ".join(errors)
    raise RuntimeError(f"failed to open camera source {source}; tried: {tried}")


def wall_clock_text(value: Optional[datetime] = None) -> str:
    value = value or datetime.now()
    return value.strftime("%H:%M:%S.") + f"{value.microsecond // 1000:03d}"


def put_text_with_background(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.65,
    thickness: int = 2,
) -> None:
    font = cv2.FONT_HERSHEY_DUPLEX
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(image, (x - 5, y - height - 6), (x + width + 5, y + baseline + 5), (0, 0, 0), -1)
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def make_reference_clock(width: int, height: int, value: datetime) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    text = wall_clock_text(value)
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 1.75
    thickness = 3
    (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
    while text_width > width - 30 and scale > 0.8:
        scale -= 0.05
        (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
    origin = ((width - text_width) // 2, (height + text_height) // 2)
    cv2.putText(canvas, text, origin, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(canvas, "REFERENCE", (14, 28), font, 0.6, (0, 220, 255), 1, cv2.LINE_AA)
    return canvas


@dataclass
class FrameSnapshot:
    frame: np.ndarray
    received_wall: datetime
    received_perf: float
    sequence: int


class LatestFrameCapture:
    """Continuously drain the camera queue and retain only the newest frame."""

    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self.lock = threading.Lock()
        self.running = True
        self.snapshot: Optional[FrameSnapshot] = None
        self.sequence = 0
        self.thread = threading.Thread(target=self._reader, name="latency-camera-reader", daemon=True)
        self.thread.start()

    def _reader(self) -> None:
        while self.running:
            ok, frame = self.cap.read()
            received_perf = time.perf_counter()
            received_wall = datetime.now()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            with self.lock:
                self.sequence += 1
                self.snapshot = FrameSnapshot(frame, received_wall, received_perf, self.sequence)

    def read(self) -> Optional[FrameSnapshot]:
        with self.lock:
            if self.snapshot is None:
                return None
            item = self.snapshot
            return FrameSnapshot(item.frame.copy(), item.received_wall, item.received_perf, item.sequence)

    def release(self) -> None:
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    scale = height / image.shape[0]
    width = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def save_evidence(
    output_dir: Path,
    reference: np.ndarray,
    feed: np.ndarray,
    snapshot: FrameSnapshot,
    now_wall: datetime,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_height = max(reference.shape[0], feed.shape[0])
    reference_resized = resize_to_height(reference, target_height)
    feed_resized = resize_to_height(feed, target_height)
    combined = np.hstack([reference_resized, feed_resized])
    stamp = now_wall.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    image_path = output_dir / f"latency_{stamp}.png"
    metadata_path = image_path.with_suffix(".json")
    if not cv2.imwrite(str(image_path), combined):
        raise RuntimeError(f"failed to save screenshot: {image_path}")
    metadata = {
        "saved_at": now_wall.astimezone().isoformat(timespec="milliseconds"),
        "frame_received_at": snapshot.received_wall.astimezone().isoformat(timespec="milliseconds"),
        "frame_sequence": snapshot.sequence,
        "note": "Subtract the clock visible inside the camera image from the RX overlay.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return image_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Display a PC reference clock beside a timestamped UVC feed for end-to-end latency tests."
    )
    parser.add_argument("--source", default="0", help="Camera index or video source; USB Video is usually 0 here.")
    parser.add_argument("--backend", choices=["auto", "dshow", "msmf", "any"], default="auto")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=0.0, help="Requested camera FPS; zero keeps the device default.")
    parser.add_argument("--reference-width", type=int, default=680)
    parser.add_argument("--reference-height", type=int, default=240)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "latency_tests"),
    )
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0:
        raise ValueError("camera width and height must be positive")
    if args.reference_width <= 0 or args.reference_height <= 0:
        raise ValueError("reference window dimensions must be positive")

    cap, backend_name = open_camera(args.source, args.backend, args.width, args.height, args.fps)
    capture = LatestFrameCapture(cap)
    output_dir = Path(args.output_dir).resolve()

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
    print(
        f"Camera: source={args.source}, backend={backend_name}, "
        f"size={actual_width}x{actual_height}, reported_fps={actual_fps:.2f}"
    )
    print("Aim the UVC camera at the REFERENCE clock window.")
    print("Latency = RX overlay time - clock time visible inside the camera image.")
    print("Keys: s=save evidence image, q/Esc=quit")

    reference_name = "Latency Reference Clock"
    feed_name = "UVC Camera Feed"
    cv2.namedWindow(reference_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(feed_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(reference_name, args.reference_width, args.reference_height)
    cv2.resizeWindow(feed_name, actual_width, actual_height)
    cv2.moveWindow(reference_name, 10, 40)
    cv2.moveWindow(feed_name, args.reference_width + 30, 40)

    last_snapshot: Optional[FrameSnapshot] = None
    last_reference: Optional[np.ndarray] = None
    last_feed: Optional[np.ndarray] = None

    try:
        while True:
            now_wall = datetime.now()
            now_perf = time.perf_counter()
            reference = make_reference_clock(args.reference_width, args.reference_height, now_wall)
            cv2.imshow(reference_name, reference)

            snapshot = capture.read()
            if snapshot is not None:
                feed = snapshot.frame
                age_ms = max(0.0, (now_perf - snapshot.received_perf) * 1000.0)
                put_text_with_background(
                    feed,
                    f"RX {wall_clock_text(snapshot.received_wall)}",
                    (10, 28),
                    (0, 255, 0),
                )
                put_text_with_background(
                    feed,
                    f"NOW {wall_clock_text(now_wall)}  AGE {age_ms:.1f} ms  FRAME {snapshot.sequence}",
                    (10, 58),
                    (0, 220, 255),
                    scale=0.55,
                    thickness=1,
                )
                cv2.imshow(feed_name, feed)
                last_snapshot = snapshot
                last_reference = reference.copy()
                last_feed = feed.copy()

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s") and last_snapshot is not None and last_reference is not None and last_feed is not None:
                saved = save_evidence(output_dir, last_reference, last_feed, last_snapshot, now_wall)
                print(f"Saved: {saved}")

            if cv2.getWindowProperty(reference_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if cv2.getWindowProperty(feed_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
