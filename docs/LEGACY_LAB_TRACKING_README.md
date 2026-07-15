# Lab Tracking Runtime

This folder contains the current PC-side lab experiment runtime.

## Layout

```text
lab_tracking/
  scripts/      runnable entry scripts
  tracking/     Kalman trajectory and pixel-to-world mapping
  models/       local model weights
  models/reference/ optional reference weights
  configs/      calibration examples
  assets/       media and calibration sample assets
  docs/         notes and usage docs
  outputs/      suggested output folder for json/video/image results
  archive/      old root-level experiments kept out of the runtime path
```

## Current Runtime

- Detector: YOLO11s
- Default class: `bottle`
- Tracker: constant-acceleration Kalman filter using real frame timestamps
- Robot output: optional JSONL arm targets

## Foam Board Bootstrap Model

The first local `foam_board` model was trained from the July 12 capture session and is intended for pipeline testing in
the same scene. Run live detection and 0.5-second pixel/angle prediction with:

```powershell
& "D:\app\miniconda\envs\python312\python.exe" ".\lab_tracking\scripts\run_foam_board.py" --source 0
```

Defaults:

```text
model: lab_tracking/models/foam_board_2p1mm_v3.pt
class: foam_board
detector confidence: 0.20
new-track confidence: 0.40
short detector-miss coasting: 3 frames
camera: source 0, DirectShow, 640x480
calibration: configs/camera_2p1mm_640x480_fisheye.json
prediction horizon: 0.5 seconds
trajectory model: constant acceleration (x, y, vx, vy, ax, ay)
```

The angle output is relative to the camera optical axis. Add the measured camera-to-base mounting rotation before using
it as a robot-base angle.

The v3 model fine-tunes the bootstrap weights with the fixed 2.1 mm lens, reviewed throw/drop keyframes, motion-blur
hard examples, and same-camera negative frames. Treat it as a scene-specific detector until it is validated in other
lighting and backgrounds.

## Run

From `D:\Study_data\ultralytics-8.3.218`:

```powershell
py -3 lab_tracking\scripts\run_lab.py --source 0
```

## Capture A Custom Object Dataset

Use the dedicated capture script to save raw, unannotated images from a USB camera. From the project root:

```powershell
& "D:\app\miniconda\envs\python312\python.exe" ".\lab_tracking\scripts\capture_dataset.py" --source 0 --backend dshow --object foam_ball
```

Try `--source 1` if the first camera is the laptop webcam. If DirectShow cannot read the USB device, retry with
`--backend msmf` or `--backend auto`.

Controls:

```text
s: save one raw image
r: start or stop automatic capture
v: start or stop raw video recording
q or Esc: finish and release the camera
```

Automatic capture defaults to one image every 0.25 seconds. Change it with `--interval`:

```powershell
& "D:\app\miniconda\envs\python312\python.exe" ".\lab_tracking\scripts\capture_dataset.py" --source 0 --backend dshow --object foam_ball --interval 0.2 --auto-start
```

For trajectory experiments, start continuous raw video recording immediately and optionally save one labeling image
every 0.2 seconds at the same time:

```powershell
& "D:\app\miniconda\envs\python312\python.exe" ".\lab_tracking\scripts\capture_dataset.py" --source 0 --backend dshow --object foam_ball --record-video --auto-start --interval 0.2
```

Each session is saved under:

```text
lab_tracking/datasets/foam_ball/raw/YYYYMMDD_HHMMSS/
```

The folder contains raw JPEG images, MP4 trajectory videos, per-video `*.timestamps.jsonl` files, `session.json`, and
image `metadata.jsonl`. Preview overlays are not written into images or videos. Keep raw sessions unmodified, then copy
selected images into train/validation folders after labeling.

Show all YOLO11s classes:

```powershell
py -3 lab_tracking\scripts\run_lab.py --list-classes
```

Save arm targets:

```powershell
py -3 lab_tracking\scripts\run_lab.py --source 0 --save-jsonl lab_tracking\outputs\arm_targets.jsonl --print-targets
```

Use calibration:

```powershell
py -3 lab_tracking\scripts\run_lab.py --source 0 --calibration configs\calibration.example.json --save-jsonl lab_tracking\outputs\arm_targets.jsonl
```

## Foam Board Note

YOLO11s pretrained on COCO can detect `bottle`, but it does not include a `foam_board` class. Foam board detection needs a small custom dataset and fine-tuning later.

## Organized Files

- Active model: `models/yolo11s.pt`
- Reference models: `models/reference/`
- Calibration examples: `configs/`
- Calibration sample data: `assets/calibration_data/`
- Sample media: `assets/media/`
- Old one-off scripts: `archive/root_experiments/`
