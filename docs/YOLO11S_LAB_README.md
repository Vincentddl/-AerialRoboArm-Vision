# YOLO11s Lab Tracking

This entry runs YOLO11s on the PC side and reuses the existing Kalman trajectory tracker.

## Run

```powershell
cd D:\目标检测
py -3 yolo11s_lab_track.py --source 0
```

The default setup is:

- model: `ultralytics-8.3.218\yolo11s.pt`
- class filter: `bottle`
- input size: `640`
- device: `0`
- tracker: Kalman filter from `FastestDet-main\trajectory.py`

## Notes

YOLO11s pretrained on COCO can detect `bottle`, so water bottles are available immediately.

COCO does not include a `foam board` class. Without collecting and training a small custom dataset, foam board detection will not be reliable. The recommended next step is to label a small two-class dataset:

- `water_bottle`
- `foam_board`

Then fine-tune from `yolo11s.pt`.

## Useful Commands

Run water-bottle tracking:

```powershell
py -3 yolo11s_lab_track.py --source 0
```

Lower confidence if bottles are missed:

```powershell
py -3 yolo11s_lab_track.py --source 0 --conf 0.15
```

Save robot-arm targets:

```powershell
py -3 yolo11s_lab_track.py --source 0 --save-jsonl arm_targets.jsonl --print-targets
```

Use calibration:

```powershell
py -3 yolo11s_lab_track.py --source 0 --calibration FastestDet-main\calibration.example.json --save-jsonl arm_targets.jsonl
```
