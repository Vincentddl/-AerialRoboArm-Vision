# Foam Board 2.1 mm V3 Training

## Source

- Video: `datasets/foam_board_2p1mm/raw/20260712_230434/foam_board_2p1mm_trajectory_20260712_230437_575968.mp4`
- Resolution: 640x480
- Frames: 5010
- Timestamp records: 5010
- Effective frame rate: 30.003 FPS
- Duration: 166.949 seconds

## Dataset

The final dataset combines the reviewed bootstrap v2 samples with 2.1 mm lens keyframes, same-camera backgrounds, and
eight manually reviewed high-motion hard examples.

- Train: 254 images, including 90 backgrounds
- Validation: 81 images, including 35 backgrounds
- Dataset YAML: `datasets/foam_board_2p1mm/v3_final/foam_board_2p1mm.yaml`
- Dataset builder: `scripts/build_foam_board_2p1mm_v3.py`
- Hard-example builder: `scripts/build_foam_board_2p1mm_v3_final.py`

Severely blurred frames were included only when the visible object extent could be labeled consistently. Frames where
the object extent was not recoverable were excluded rather than mislabeled as background.

## Final Model

- Weights: `models/foam_board_2p1mm_v3.pt`
- Starting weights: `models/foam_board_bootstrap_v2.pt`
- Validation precision: 0.981
- Validation recall: 0.913
- Validation mAP50: 0.989
- Validation mAP50-95: 0.592

## Stress Tests

Ten high-motion frames were checked at confidence 0.20:

- Bootstrap v2: 9/10 frames detected, 24 total boxes
- 2.1 mm v3: 10/10 frames detected, 10 total boxes

Sixteen reviewed background frames were checked at confidence 0.20. V3 produced three low-score candidates (0.228,
0.254, and 0.391). All are below the runtime new-track threshold of 0.40 and therefore do not create robot targets.

The full 5010-frame video processed in 61.3 seconds using recorded frame timestamps. It produced 35 tracks with at
least eight output records and bridged detector misses for up to three frames with the acceleration model.

## Limits

- The model is scene-specific and should be revalidated after lighting, background, lens focus, or camera changes.
- Camera calibration corrects output angles; the detector still consumes raw distorted frames.
- Fixed 0.5-second prediction can leave the image for fast targets. Robot deployment should predict intersection with
  a calibrated catch plane using camera-to-base extrinsics and measured end-to-end latency.
