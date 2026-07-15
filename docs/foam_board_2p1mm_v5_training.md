# Foam Board 2.1 mm V5 Training

## Static Source

- Session: `datasets/foam_board_2p1mm/raw/20260713_214630`
- Captured images: 200 at 640x480
- Selected images: 140
- Static train/validation split: 125/15
- Annotation review: 182 automatic boxes plus 18 manually corrected boxes

The selection uses farthest-point sampling over target position, scale, aspect ratio, estimated orientation,
target-crop appearance, and full-frame appearance. Immediate neighbors of validation captures are excluded from
training to reduce leakage from near-duplicate consecutive frames.

## Mixed Dataset

V5 copies the complete V4 dataset before adding the selected static images. This keeps the dynamic target frames and
metal-panel hard negatives in the training distribution instead of allowing the white-paper session to dominate it.

- Train: 440 images, including 135 negative images
- Validation: 125 images, including 56 negative images
- Dataset: `datasets/foam_board_2p1mm/v5`
- Builder: `scripts/build_foam_board_2p1mm_v5.py`
- Selection manifest: `datasets/foam_board_2p1mm/v5/manifest.json`

## Model

- Starting weights: `models/foam_board_2p1mm_v4.pt`
- Final weights: `models/foam_board_2p1mm_v5.pt`
- Training run: `runs/foam_board_2p1mm_v5`
- Best epoch: 12
- Device: NVIDIA GeForce RTX 4070 Laptop GPU

V5 validation metrics on the mixed V5 validation set:

- Precision: 0.895
- Recall: 0.928
- mAP50: 0.895
- mAP50-95: 0.454

## V4/V5 Cross-Check

On the original V4 validation set:

| Model | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| V4 | 0.925 | 0.908 | 0.971 | 0.567 |
| V5 | 0.941 | 0.981 | 0.984 | 0.477 |

On the mixed V5 validation set:

| Model | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| V4 | 0.944 | 0.754 | 0.827 | 0.483 |
| V5 | 0.895 | 0.928 | 0.895 | 0.454 |

V5 prioritizes detection recall over very tight box boundaries. This is appropriate for the trajectory pipeline,
which depends primarily on stable target centers and continuous detections.

At confidence 0.25 on the 56 target-free V4 validation images, V4 produced detections in 22 images and V5 in 6.
The maximum background confidence fell from 0.756 to 0.439. Remaining high-confidence V5 errors were primarily dark
rectangles on a laptop display, not white paper or hands.

V4 remains available at `models/foam_board_2p1mm_v4.pt` for rollback.
