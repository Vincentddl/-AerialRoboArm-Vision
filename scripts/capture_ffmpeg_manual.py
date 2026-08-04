import argparse
import json
import queue
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


def safe_name(value):
    cleaned = "".join(char if char.isalnum() or char in "-_" else "_" for char in value.strip())
    return cleaned.strip("_") or "capture"


def resolve_output_dir(value, object_name):
    if value:
        path = Path(value)
        return path if path.is_absolute() else PROJECT_DIR / path
    return PROJECT_DIR / "datasets" / object_name / "raw"


def parse_angle_sequence(value):
    if not value.strip():
        return []
    angles = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        angles.append(float(item))
    if not angles:
        raise ValueError("--angles must contain at least one numeric angle")
    return angles


class JpegPipeReader:
    def __init__(self, stream):
        self.stream = stream
        self.buffer = bytearray()

    def read(self):
        while True:
            start = self.buffer.find(b"\xff\xd8")
            if start >= 0:
                end = self.buffer.find(b"\xff\xd9", start + 2)
                if end >= 0:
                    packet = bytes(self.buffer[start : end + 2])
                    del self.buffer[: end + 2]
                    return packet
                if start > 0:
                    del self.buffer[:start]
            elif len(self.buffer) > 2:
                del self.buffer[:-2]

            chunk = self.stream.read(65536)
            if not chunk:
                return None
            self.buffer.extend(chunk)


class JpegFramePump:
    """Read the FFmpeg pipe off the UI thread so the preview stays responsive."""

    def __init__(self, reader, max_queue_size=250):
        self.reader = reader
        self.frames = queue.Queue(maxsize=max_queue_size)
        self.error = None
        self.finished = False
        self.thread = threading.Thread(target=self._run, name="jpeg-frame-pump", daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        try:
            while True:
                packet = self.reader.read()
                if packet is None:
                    break
                self.frames.put(packet)
        except BaseException as exc:
            self.error = exc
        finally:
            self.finished = True


class ManualRecorder:
    def __init__(self, ffmpeg, output_dir, prefix, fps):
        self.ffmpeg = ffmpeg
        self.output_dir = output_dir
        self.prefix = prefix
        self.fps = fps
        self.raw_file = None
        self.raw_path = None
        self.video_path = None
        self.timestamps_path = None
        self.events_path = None
        self.timestamps_file = None
        self.events_file = None
        self.log_file = None
        self.frame_count = 0
        self.started_monotonic = None

    @property
    def active(self):
        return self.raw_file is not None

    def start(self):
        if self.active:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        stem = f"{self.prefix}_{stamp}"
        self.video_path = self.output_dir / f"{stem}.mkv"
        self.raw_path = self.output_dir / f"{stem}.mjpeg.partial"
        self.timestamps_path = self.output_dir / f"{stem}.timestamps.jsonl"
        self.events_path = self.output_dir / f"{stem}.angles.jsonl"
        log_path = self.output_dir / f"{stem}.ffmpeg.log"
        self.timestamps_file = self.timestamps_path.open("w", encoding="utf-8", buffering=1)
        self.events_file = self.events_path.open("w", encoding="utf-8", buffering=1)
        self.log_file = log_path.open("w", encoding="utf-8")
        # Write validated camera JPEG packets directly while recording. Feeding a
        # second FFmpeg process synchronously can block the UI while that process
        # probes a damaged first analog-link frame. The raw stream is remuxed on S.
        self.raw_file = self.raw_path.open("wb", buffering=1024 * 1024)
        self.frame_count = 0
        self.started_monotonic = time.monotonic()

    def write(self, jpeg_packet):
        if not self.active:
            return
        self.raw_file.write(jpeg_packet)
        now = time.monotonic()
        record = {
            "frame_index": self.frame_count,
            "captured_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "monotonic_seconds": now,
            "elapsed_seconds": now - self.started_monotonic,
        }
        self.timestamps_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.frame_count += 1

    def record_event(self, event, **values):
        if not self.active or self.events_file is None:
            return
        now = time.monotonic()
        record = {
            "event": event,
            "frame_index": self.frame_count,
            "captured_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "elapsed_seconds": now - self.started_monotonic,
            **values,
        }
        self.events_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def stop(self):
        if not self.active:
            return None
        raw_file = self.raw_file
        self.raw_file = None
        try:
            raw_file.close()
            self.timestamps_file.close()
            self.timestamps_file = None
            self.events_file.close()
            self.events_file = None

            if self.frame_count <= 0:
                raise RuntimeError("recording stopped without receiving a valid frame")

            command = [
                self.ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-probesize",
                "10M",
                "-analyzeduration",
                "10M",
                "-f",
                "mjpeg",
                "-framerate",
                f"{self.fps:g}",
                "-i",
                str(self.raw_path),
                "-an",
                "-c:v",
                "copy",
                "-f",
                "matroska",
                str(self.video_path),
            ]
            creationflags = (
                subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
            )
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=self.log_file,
                creationflags=creationflags,
                check=False,
            )
            if result.returncode:
                raise RuntimeError(f"recording FFmpeg exited with code {result.returncode}")
            self.raw_path.unlink()
        finally:
            if not raw_file.closed:
                raw_file.close()
            if self.timestamps_file is not None:
                self.timestamps_file.close()
                self.timestamps_file = None
            if self.events_file is not None:
                self.events_file.close()
                self.events_file = None
            self.log_file.close()
            self.log_file = None
        return self.video_path


def draw_status(frame, recording, fps, recorded_frames, elapsed, guide=None):
    display = frame.copy()
    status = "REC" if recording else "PREVIEW - NOT RECORDING"
    color = (0, 0, 255) if recording else (0, 255, 255)
    lines = [
        status,
        f"USB Video | 640x480 MJPEG | requested {fps:g} FPS",
        f"R: start   S: stop/save   Q/Esc: exit",
    ]
    if recording:
        lines.append(f"recorded {recorded_frames} frames | {elapsed:.1f} s")
    for index, line in enumerate(lines):
        y = 28 + index * 27
        cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4)
        cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1)
    if recording:
        cv2.circle(display, (display.shape[1] - 24, 24), 9, color, -1)

    if guide:
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 128), (display.shape[1], 266), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.78, display, 0.22, 0, display)
        state = guide["state"]
        state_color = {
            "idle": (0, 255, 255),
            "move": (0, 255, 255),
            "hold": (0, 165, 255),
            "done": (0, 255, 0),
            "finished": (0, 255, 0),
        }[state]
        angle_text = f"g {guide['angle']:g} deg"
        header = f"STEP {guide['step']:02d}/{guide['count']:02d}    COMMAND: {angle_text}"
        if state == "idle":
            message = "Press R to record, then move the servo to this angle"
        elif state == "move":
            message = f"MOVE TO {angle_text} | when stable, press SPACE"
        elif state == "hold":
            message = f"HOLD STILL  {guide['hold_elapsed']:.1f} / {guide['hold_seconds']:.1f} s"
        elif state == "done" and guide["next_angle"] is not None:
            message = f"CAPTURED | press SPACE for next: g {guide['next_angle']:g} deg"
        elif state == "done":
            message = "LAST ANGLE CAPTURED | press SPACE to finish sequence"
        else:
            message = "SEQUENCE COMPLETE | press S to save"
        guide_lines = [header, message, "SPACE: confirm/next    B: previous angle"]
        for index, line in enumerate(guide_lines):
            y = 160 + index * 43
            scale = 0.82 if index == 0 else 0.66
            cv2.putText(display, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
            cv2.putText(display, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, state_color, 2)
    return display


def main():
    parser = argparse.ArgumentParser(
        description="Manual 50 FPS DirectShow preview and lossless MJPEG recording."
    )
    parser.add_argument("--device", default="USB Video", help="DirectShow video device name.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--object", default="arm_motion_slow_50fps")
    parser.add_argument("--out", default="datasets/arm_motion_slow_50fps/raw")
    parser.add_argument(
        "--angles",
        default="",
        help="Optional comma-separated guided servo angles, for example -30,-35,-40.",
    )
    parser.add_argument("--hold-seconds", type=float, default=3.0)
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0 or args.fps <= 0 or args.hold_seconds <= 0:
        raise ValueError("width, height, fps, and hold-seconds must be greater than zero")
    guide_angles = parse_angle_sequence(args.angles)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not found on PATH")

    object_name = safe_name(args.object)
    output_dir = resolve_output_dir(args.out, object_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    session_path = output_dir / "manual_capture_session.json"
    session_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "device": args.device,
                "width": args.width,
                "height": args.height,
                "fps": args.fps,
                "format": "MJPEG",
                "control": {
                    "start": "R",
                    "confirm_or_next_angle": "Space",
                    "previous_angle": "B",
                    "stop_save": "S",
                    "exit": ["Q", "Esc"],
                },
                "angle_guide": {
                    "angles_deg": guide_angles,
                    "hold_seconds": args.hold_seconds,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    producer_command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "dshow",
        "-video_size",
        f"{args.width}x{args.height}",
        "-framerate",
        f"{args.fps:g}",
        "-vcodec",
        "mjpeg",
        "-i",
        f"video={args.device}",
        "-an",
        "-c:v",
        "copy",
        "-f",
        "image2pipe",
        "pipe:1",
    ]
    creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
    producer = subprocess.Popen(
        producer_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        creationflags=creationflags,
    )
    reader = JpegPipeReader(producer.stdout)
    frame_pump = JpegFramePump(reader)
    frame_pump.start()
    recorder = ManualRecorder(ffmpeg, output_dir, object_name, args.fps)
    window_name = "Manual 50 FPS Capture"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    last_frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    guide_index = 0
    guide_state = "idle"
    guide_hold_started = None

    try:
        while True:
            try:
                packet = frame_pump.frames.get(timeout=0.02)
            except queue.Empty:
                packet = None

            if packet is None and frame_pump.finished:
                error = producer.stderr.read().decode("utf-8", errors="replace").strip()
                if frame_pump.error:
                    raise RuntimeError(str(frame_pump.error)) from frame_pump.error
                raise RuntimeError(error or "capture FFmpeg stopped producing frames")

            packet_is_valid = False
            if packet is not None:
                frame = cv2.imdecode(np.frombuffer(packet, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    last_frame = frame
                    packet_is_valid = True

            if recorder.active and packet_is_valid:
                recorder.write(packet)
            elapsed = (
                time.monotonic() - recorder.started_monotonic if recorder.active else 0.0
            )
            if guide_angles and recorder.active and guide_state == "hold":
                hold_elapsed = time.monotonic() - guide_hold_started
                if hold_elapsed >= args.hold_seconds:
                    guide_state = "done"
                    recorder.record_event(
                        "hold_completed",
                        angle_deg=guide_angles[guide_index],
                        step=guide_index + 1,
                    )
            else:
                hold_elapsed = 0.0

            guide = None
            if guide_angles:
                guide = {
                    "state": guide_state,
                    "step": guide_index + 1,
                    "count": len(guide_angles),
                    "angle": guide_angles[guide_index],
                    "next_angle": (
                        guide_angles[guide_index + 1]
                        if guide_index + 1 < len(guide_angles)
                        else None
                    ),
                    "hold_elapsed": min(hold_elapsed, args.hold_seconds),
                    "hold_seconds": args.hold_seconds,
                }
            display = draw_status(
                last_frame,
                recorder.active,
                args.fps,
                recorder.frame_count,
                elapsed,
                guide,
            )
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("r"), ord("R")) and not recorder.active:
                recorder.start()
                if guide_angles:
                    guide_index = 0
                    guide_state = "move"
                    guide_hold_started = None
                    recorder.record_event(
                        "guide_started",
                        angle_deg=guide_angles[guide_index],
                        step=guide_index + 1,
                    )
            elif key == ord(" ") and recorder.active and guide_angles:
                if guide_state == "move":
                    guide_state = "hold"
                    guide_hold_started = time.monotonic()
                    recorder.record_event(
                        "hold_started",
                        angle_deg=guide_angles[guide_index],
                        step=guide_index + 1,
                    )
                elif guide_state == "done":
                    if guide_index + 1 < len(guide_angles):
                        guide_index += 1
                        guide_state = "move"
                        guide_hold_started = None
                        recorder.record_event(
                            "move_prompt",
                            angle_deg=guide_angles[guide_index],
                            step=guide_index + 1,
                        )
                    else:
                        guide_state = "finished"
                        recorder.record_event("sequence_completed", steps=len(guide_angles))
            elif key in (ord("b"), ord("B")) and recorder.active and guide_angles:
                guide_index = max(0, guide_index - 1)
                guide_state = "move"
                guide_hold_started = None
                recorder.record_event(
                    "step_back",
                    angle_deg=guide_angles[guide_index],
                    step=guide_index + 1,
                )
            elif key in (ord("s"), ord("S")) and recorder.active:
                if guide_angles:
                    recorder.record_event(
                        "recording_stopped",
                        guide_state=guide_state,
                        completed_steps=(guide_index + 1 if guide_state in {"done", "finished"} else guide_index),
                    )
                saved_path = recorder.stop()
                print(f"Saved: {saved_path}", flush=True)
            elif key in (ord("q"), ord("Q"), 27):
                break
    finally:
        if recorder.active:
            saved_path = recorder.stop()
            print(f"Saved: {saved_path}", flush=True)
        producer.terminate()
        try:
            producer.wait(timeout=5)
        except subprocess.TimeoutExpired:
            producer.kill()
            producer.wait(timeout=5)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
