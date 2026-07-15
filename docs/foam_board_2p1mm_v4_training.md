# Foam Board 2.1 mm V4 Training

## Source

- Video: `datasets/foam_board_2p1mm/raw/20260713_152652/foam_board_2p1mm_trajectory_20260713_152655_332571.mp4`
- Resolution: 640x480
- Frames: 3907
- Duration: 129.865 seconds
- Timestamp records: 3907

The recording contains clean metal-panel backgrounds, operator/phone hard negatives, multiple target distances,
partial-frame targets, and fast motion blur. Frames with severe analog-link corruption were excluded from training.

## Dataset

V4 extends the reviewed V3 dataset with samples from the new scene. All new labels were visually reviewed.

- Train: 315 images
- Validation: 110 images
- New positives: 16 train, 8 validation
- New negatives: 45 train, 21 validation
- Dataset: `datasets/foam_board_2p1mm/v4`
- Builder: `scripts/build_foam_board_2p1mm_v4.py`

## Model

- Starting weights: `models/foam_board_2p1mm_v3.pt`
- Final weights: `models/foam_board_2p1mm_v4.pt`
- Training run: `runs/foam_board_2p1mm_v4`
- Best epoch: 2; early stopping completed after epoch 10

V4 validation metrics:

- Precision: 0.925
- Recall: 0.908
- mAP50: 0.971
- mAP50-95: 0.567

On the original V3 validation set, V4 achieved precision 0.979, recall 0.994, mAP50 0.994, and mAP50-95 0.593,
so the new-scene adaptation did not regress the previous scene.

## Static Background Gate

The fixed dark panel at the right edge is visually similar to the black foam target. Single-frame training alone could
not reliably distinguish it, so the runtime now confirms target motion before producing a robot target:

- Two matching frames required to create a track
- At least 12 px candidate displacement
- At least 80 px/s track speed before arm output
- Track removed after eight consecutive low-speed frames

Regression results with the default runtime configuration:

- Pure background: 207 frames, zero arm-target records
- Fast-motion clip: 621 frames, 172 output timestamps, 191 target records, 13 track IDs

The previous V3 weights remain in `models/foam_board_2p1mm_v3.pt` for rollback.
