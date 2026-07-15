import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
LAB_DIR = SCRIPT_DIR.parent

BACKENDS = {
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "any": cv2.CAP_ANY,
}


def parse_source(source):
    return int(source) if str(source).isdigit() else source


def backend_candidates(name):
    if name == "auto":
        return [("dshow", cv2.CAP_DSHOW), ("msmf", cv2.CAP_MSMF), ("any", cv2.CAP_ANY)]
    return [(name, BACKENDS[name])]


def open_source(source, backend, width, height):
    parsed_source = parse_source(source)
    candidates = backend_candidates(backend) if isinstance(parsed_source, int) else [("any", cv2.CAP_ANY)]
    failures = []

    for backend_name, backend_id in candidates:
        cap = cv2.VideoCapture(parsed_source, backend_id)
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            failures.append(f"{backend_name}: open failed")
            cap.release()
            continue

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap, backend_name, frame
            time.sleep(0.02)

        failures.append(f"{backend_name}: opened but no frames")
        cap.release()

    detail = "; ".join(failures) or "no backend candidates"
    raise RuntimeError(f"failed to read source {source} ({detail})")


def safe_name(value):
    cleaned = re.sub(r"[^\w-]+", "_", value.strip(), flags=re.UNICODE).strip("_")
    return cleaned or "object"


def estimate_frame_rate(cap, duration=1.2):
    start = time.perf_counter()
    count = 0
    latest_frame = None
    while time.perf_counter() - start < duration:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        latest_frame = frame
        count += 1
    elapsed = time.perf_counter() - start
    return (count / elapsed if count >= 2 and elapsed > 0 else 0.0), latest_frame


def frame_signature(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (5, 5), 0)


def frame_difference(first, second):
    if first is None or second is None:
        return float("inf")
    return float(np.mean(cv2.absdiff(first, second)))


def sharpness_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def resolve_output_dir(output, object_name):
    if output:
        path = Path(output)
        return path if path.is_absolute() else LAB_DIR / path
    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LAB_DIR / "datasets" / object_name / "raw" / session


def next_sequence(output_dir, prefix):
    largest = 0
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    for path in output_dir.glob(f"{prefix}_*.jpg"):
        match = pattern.match(path.stem)
        if match:
            largest = max(largest, int(match.group(1)))
    return largest + 1


class VideoRecorder:
    def __init__(self, output_dir, prefix, fps):
        self.output_dir = output_dir
        self.prefix = prefix
        self.fps = fps
        self.writer = None
        self.metadata_file = None
        self.video_path = None
        self.frame_index = 0

    @property
    def is_recording(self):
        return self.writer is not None

    def start(self, frame):
        if self.is_recording:
            return self.video_path

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.video_path = self.output_dir / f"{self.prefix}_trajectory_{timestamp}.mp4"
        metadata_path = self.video_path.with_suffix(".timestamps.jsonl")
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.video_path), fourcc, self.fps, (width, height))
        if not self.writer.isOpened():
            self.writer.release()
            self.writer = None
            raise RuntimeError(f"failed to create video: {self.video_path}")

        self.metadata_file = metadata_path.open("w", encoding="utf-8")
        self.frame_index = 0
        print(f"video recording started: {self.video_path}")
        return self.video_path

    def write(self, frame):
        if not self.is_recording:
            return
        self.writer.write(frame)
        record = {
            "frame_index": self.frame_index,
            "captured_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "monotonic_seconds": time.perf_counter(),
        }
        self.metadata_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.frame_index += 1

    def stop(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.metadata_file is not None:
            self.metadata_file.close()
            self.metadata_file = None
        if self.video_path is not None:
            print(f"video recording stopped: {self.video_path} ({self.frame_index} frames)")
        stopped_path = self.video_path
        self.video_path = None
        self.frame_index = 0
        return stopped_path


def draw_status(
    frame,
    saved,
    automatic,
    interval,
    recording,
    source,
    backend,
    flash_message,
    capture_status="",
    quality_text="",
):
    display = frame.copy()
    mode = f"AUTO {interval:.2f}s" if automatic else "MANUAL"
    lines = [
        f"source {source} | {backend} | {display.shape[1]}x{display.shape[0]}",
        f"saved {saved} | images {mode} | video {'REC' if recording else 'OFF'}",
        "s save | r image auto | Space/v start-stop video | q/Esc quit",
    ]
    if flash_message:
        lines.append(flash_message)
    elif capture_status:
        lines.append(capture_status)
    if quality_text:
        lines.append(quality_text)

    for index, line in enumerate(lines):
        y = 26 + index * 26
        cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        color = (0, 255, 255) if index < 3 else (0, 255, 0)
        cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1)
    return display


def main():
    parser = argparse.ArgumentParser(description="Capture raw object-detection images from a camera.")
    parser.add_argument("--source", default="0", help="Camera index, video file, or stream URL.")
    parser.add_argument(
        "--backend",
        choices=["auto", "dshow", "msmf", "any"],
        default="auto",
        help="Windows camera backend. Use dshow first for USB capture devices.",
    )
    parser.add_argument("--object", default="foam_ball", help="Object/session name used in folders and filenames.")
    parser.add_argument("--out", default="", help="Output directory. Relative paths are resolved from the project root.")
    parser.add_argument("--width", type=int, default=640, help="Requested camera width.")
    parser.add_argument("--height", type=int, default=480, help="Requested camera height.")
    parser.add_argument("--interval", type=float, default=0.25, help="Seconds between images in automatic mode.")
    parser.add_argument("--auto-start", action="store_true", help="Start automatic capture immediately.")
    parser.add_argument(
        "--static-mode",
        action="store_true",
        help="Save a sharp image after the scene settles and reject near-duplicates.",
    )
    parser.add_argument("--settle-seconds", type=float, default=0.35, help="Stable time required in static mode.")
    parser.add_argument("--max-motion", type=float, default=1.0, help="Maximum frame change considered stable.")
    parser.add_argument("--min-change", type=float, default=3.0, help="Minimum change from the last saved static image.")
    parser.add_argument("--min-sharpness", type=float, default=150.0, help="Minimum Laplacian sharpness in static mode.")
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Stop automatic capture when the output directory reaches this total; zero disables.",
    )
    parser.add_argument("--record-video", action="store_true", help="Start raw video recording immediately.")
    parser.add_argument(
        "--video-fps",
        type=float,
        default=0.0,
        help="Saved video FPS. Zero measures live-camera FPS and falls back to the reported rate or 25.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality from 1 to 100.")
    args = parser.parse_args()

    if args.interval <= 0:
        raise ValueError("--interval must be greater than zero")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")
    if args.video_fps < 0:
        raise ValueError("--video-fps cannot be negative")
    if args.settle_seconds < 0:
        raise ValueError("--settle-seconds cannot be negative")
    if args.max_motion < 0 or args.min_change < 0 or args.min_sharpness < 0:
        raise ValueError("static-mode quality thresholds cannot be negative")
    if args.max_images < 0:
        raise ValueError("--max-images cannot be negative")

    object_name = safe_name(args.object)
    output_dir = resolve_output_dir(args.out, object_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"

    cap, backend_name, frame = open_source(args.source, args.backend, args.width, args.height)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
    estimated_fps = 0.0
    is_live_camera = isinstance(parse_source(args.source), int)
    if is_live_camera or actual_fps <= 1:
        estimated_fps, latest_frame = estimate_frame_rate(cap)
        if latest_frame is not None:
            frame = latest_frame

    session_data = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "object": args.object,
        "filename_prefix": object_name,
        "source": args.source,
        "backend": backend_name,
        "width": actual_width,
        "height": actual_height,
        "reported_fps": actual_fps,
        "estimated_fps": estimated_fps,
        "capture_interval_seconds": args.interval,
        "requested_video_fps": args.video_fps,
        "capture_mode": "static" if args.static_mode else "standard",
        "settle_seconds": args.settle_seconds,
        "max_motion": args.max_motion,
        "min_change": args.min_change,
        "min_sharpness": args.min_sharpness,
        "max_images": args.max_images,
    }
    (output_dir / "session.json").write_text(json.dumps(session_data, indent=2, ensure_ascii=False), encoding="utf-8")

    sequence = next_sequence(output_dir, object_name)
    saved = sequence - 1
    session_saved = 0
    automatic = args.auto_start
    if args.video_fps:
        video_fps = args.video_fps
        video_fps_source = "command line"
    elif is_live_camera and estimated_fps > 1:
        video_fps = estimated_fps
        video_fps_source = "measured camera rate"
    elif actual_fps > 1:
        video_fps = actual_fps
        video_fps_source = "camera reported rate"
    elif estimated_fps > 1:
        video_fps = estimated_fps
        video_fps_source = "measured fallback rate"
    else:
        video_fps = 25.0
        video_fps_source = "fallback"
    recorder = VideoRecorder(output_dir, object_name, video_fps)
    next_auto_save = time.monotonic()
    flash_message = ""
    flash_until = 0.0
    previous_signature = None
    last_saved_signature = None
    stable_since = time.monotonic()
    capture_status = ""
    current_motion = float("inf")
    current_change = float("inf")
    current_sharpness = 0.0

    print(f"Camera: source={args.source}, backend={backend_name}, size={actual_width}x{actual_height}, fps={actual_fps:.2f}")
    print(f"Output: {output_dir}")
    print(f"Video FPS: {video_fps:.2f} ({video_fps_source})")
    print("Keys: s=save, r=automatic images on/off, Space/v/V=start/stop video recording, q/Esc=quit")
    if args.static_mode:
        print("Static mode: move the target to a new pose, hold it still, and wait for one saved message.")
        print(f"Static limits: settle={args.settle_seconds:.2f}s, motion<={args.max_motion:.2f}, change>={args.min_change:.2f}, sharpness>={args.min_sharpness:.1f}")

    def save_image(raw_frame, trigger, quality=None):
        nonlocal sequence, saved, session_saved, flash_message, flash_until
        filename = f"{object_name}_{sequence:06d}.jpg"
        image_path = output_dir / filename
        ok = cv2.imwrite(str(image_path), raw_frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        if not ok:
            raise RuntimeError(f"failed to save image: {image_path}")

        captured_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
        record = {
            "file": filename,
            "captured_at": captured_at,
            "monotonic_seconds": time.perf_counter(),
            "trigger": trigger,
            "source": args.source,
            "backend": backend_name,
            "width": int(raw_frame.shape[1]),
            "height": int(raw_frame.shape[0]),
            "object": args.object,
            "capture_mode": "static" if args.static_mode else "standard",
        }
        if quality:
            record["quality"] = quality
        with metadata_path.open("a", encoding="utf-8") as metadata_file:
            metadata_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        sequence += 1
        saved += 1
        session_saved += 1
        flash_message = f"saved {filename}"
        flash_until = time.monotonic() + 1.0
        print(flash_message)

    try:
        if args.record_video:
            recorder.start(frame)
        while True:
            now = time.monotonic()
            recorder.write(frame)
            signature = frame_signature(frame) if args.static_mode else None
            if args.static_mode:
                current_motion = frame_difference(previous_signature, signature)
                current_change = frame_difference(last_saved_signature, signature)
                current_sharpness = sharpness_score(frame)
                if current_motion > args.max_motion:
                    stable_since = now
                    capture_status = "MOVING"
                elif now - stable_since < args.settle_seconds:
                    capture_status = "SETTLING"
                elif current_sharpness < args.min_sharpness:
                    capture_status = "BLURRY"
                elif current_change < args.min_change:
                    capture_status = "MOVE TO A NEW POSE"
                else:
                    capture_status = "READY"

                if automatic and now >= next_auto_save and capture_status == "READY":
                    save_image(
                        frame,
                        "auto_static",
                        {
                            "sharpness": round(current_sharpness, 3),
                            "motion": round(current_motion, 3),
                            "change_from_previous": None if last_saved_signature is None else round(current_change, 3),
                            "settled_seconds": round(now - stable_since, 3),
                        },
                    )
                    last_saved_signature = signature.copy()
                    next_auto_save = now + args.interval
                    capture_status = "SAVED - MOVE TO A NEW POSE"
                    if args.max_images and saved >= args.max_images:
                        automatic = False
                        capture_status = f"COMPLETE {saved}/{args.max_images}"
                        print(capture_status)
                previous_signature = signature
            elif automatic and now >= next_auto_save:
                save_image(frame, "auto")
                next_auto_save = now + args.interval

            visible_flash = flash_message if now < flash_until else ""
            display = draw_status(
                frame,
                saved,
                automatic,
                args.interval,
                recorder.is_recording,
                args.source,
                backend_name,
                visible_flash,
                capture_status if args.static_mode else "",
                (
                    f"sharp {current_sharpness:.0f} | motion {current_motion:.2f} | change {current_change:.2f}"
                    if args.static_mode
                    else ""
                ),
            )
            cv2.imshow("Object Dataset Capture", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                quality = None
                if args.static_mode:
                    quality = {
                        "sharpness": round(current_sharpness, 3),
                        "motion": round(current_motion, 3),
                        "change_from_previous": None if last_saved_signature is None else round(current_change, 3),
                        "settled_seconds": round(now - stable_since, 3),
                    }
                save_image(frame, "manual", quality)
                if args.static_mode:
                    last_saved_signature = signature.copy()
            elif key == ord("r"):
                automatic = not automatic
                next_auto_save = time.monotonic()
                if args.static_mode:
                    stable_since = next_auto_save
                print(f"automatic capture: {'on' if automatic else 'off'}")
            elif key in (ord("v"), ord("V"), ord(" ")):
                if recorder.is_recording:
                    stopped_path = recorder.stop()
                    flash_message = f"video saved {stopped_path.name}"
                else:
                    started_path = recorder.start(frame)
                    flash_message = f"recording {started_path.name}"
                flash_until = time.monotonic() + 1.5

            ok, next_frame = cap.read()
            if not ok or next_frame is None:
                time.sleep(0.01)
                continue
            frame = next_frame
    finally:
        recorder.stop()
        cap.release()
        cv2.destroyAllWindows()

    print(f"Finished. Saved {saved} images in {output_dir}")


if __name__ == "__main__":
    main()
