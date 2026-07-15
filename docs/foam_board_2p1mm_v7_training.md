# Foam Board 2.1 mm V7 Training

## Source Review

- Video: `datasets/foam_board_2p1mm_zaxis/raw/20260714_223253/foam_board_2p1mm_zaxis_trajectory_20260714_223351_701282.mp4`
- Frames: 2050 at 640x480
- Timestamp duration: 68.256 seconds, 30.019 FPS effective
- Encoded duration: 78.80 seconds, 26.015 FPS reported by the capture device

The first part of the recording mainly contains hand-held target poses. The later part contains repeated fast vertical
passes. Only visually reviewed free-motion detections and confirmed target-free frames were added to V7. Positive
training and validation samples come from separate throw intervals so adjacent frames cannot cross the split.

This recording is an incremental object-detection source. It is not treated as a trajectory dynamics training set:
there are only about eight released motions, no physical Z-position calibration, and some motion remains hand-driven.

## Dataset

- Source dataset: `datasets/foam_board_2p1mm/v6`
- Builder: `scripts/build_foam_board_2p1mm_v7.py`
- Existing V6 train/validation: 516/143
- New positive train/validation: 67/14
- New negative train/validation: 33/10
- Final train/validation: 616/167
- Final negative train/validation: 184/70

The reviewed selection is stored as `v7_candidate_selection.json` beside the raw video. V6 boxes were retained only
after full contact-sheet inspection. The 43 new negative frames contain the guide board, hand, fixed panel, or empty
scene without the foam target.

## Training

- Starting weights: `models/foam_board_2p1mm_v6.pt`
- Final weights: `models/foam_board_2p1mm_v7.pt`
- Run: `runs/foam_board_2p1mm_v7`
- Best epoch: 8; early stop at epoch 16
- Device: NVIDIA GeForce RTX 4070 Laptop GPU
- Image size: 640; batch: 4
- Optimizer: AdamW; initial learning rate: 0.0002

V7 mixed validation metrics:

| Precision | Recall | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: |
| 0.911 | 0.846 | 0.884 | 0.502 |

## Cross-Checks

On the original V6 validation set, V7 obtained P=0.897, R=0.844, mAP50=0.871, and mAP50-95=0.481. The original
V6 result on its mixed validation set was P=0.841, R=0.831, mAP50=0.831, and mAP50-95=0.467.

On recording frames 1800-2049, which were not used for V7 training, detection coverage changed as follows:

| Confidence | V6 detected frames | V7 detected frames |
| ---: | ---: | ---: |
| 0.20 | 27/250 | 32/250 |
| 0.40 | 22/250 | 30/250 |

All 43 reviewed target-free frames had zero detections from both V6 and V7 at confidence thresholds 0.10, 0.20,
and 0.40. The complete comparison is stored in `v7_evaluation.json` beside the source video.

V7 improves high-confidence dynamic detections, but frame coverage is not trajectory accuracy. A reliable Z-axis
predictor still requires more released trajectories, physical Z calibration, timestamps, and independent landing or
position ground truth.
