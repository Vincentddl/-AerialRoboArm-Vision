# Local Archive Inventory

Generated during the local merge on 2026-07-15.

## Sources

```text
D:/Study_data/camera_calibration
D:/Study_data/ultralytics-8.3.218/lab_tracking
```

The source directories were copied, not moved. They remain available as rollback references.

## Merged Project

```text
D:/Study_data/AerialRoboArm-Vision
total files: 9476
total size: approximately 7.064 GB
```

Top-level local content at merge time:

| Path | Files | Size |
| --- | ---: | ---: |
| `calibration/` | 1152 | 481.85 MB |
| `datasets/` | 7837 | 546.27 MB |
| `models/` | 12 | 194.61 MB |
| `runs/` | 195 | 444.00 MB |
| `outputs/` | 229 | 5489.46 MB |
| `archive/` | 7 | 63.12 MB |
| `assets/` | 2 | 14.37 MB |

## Active Artifacts

- Camera calibration: `configs/camera_2p1mm_640x480_fisheye.json`
- Calibration source set: `calibration/images_2p1mm_final/`
- Detector weight: `models/foam_board_2p1mm_v7.pt`
- V7 dataset: `datasets/foam_board_2p1mm/v7/`
- Z-axis raw recording: `datasets/foam_board_2p1mm_zaxis/raw/20260714_223253/`
- Historical training run: `runs/foam_board_2p1mm_v7/`
- Historical trajectory evaluation: `outputs/v7_zaxis_current_prediction_eval.json`

## Verification

- All 8248 non-cache files from `lab_tracking` are present.
- All camera calibration files are present except the intentionally omitted editor-only `.vscode/settings.json`.
- Recomputed fisheye calibration used 53 valid images and reproduced RMS `0.228637` and reprojection error `0.229 px`.
- V7 smoke inference loaded the merged model and used CUDA.
- Full Z-axis replay used the copied frame timestamps at `30.019 FPS` effective.

Large local artifacts are intentionally excluded by `.gitignore`; this document records their presence without attempting to store them in ordinary Git history.
