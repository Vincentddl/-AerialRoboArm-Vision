# Foam Board 2.1 mm V6 Training

## Dynamic Source

- Video: `datasets/foam_board_2p1mm/raw/20260714_002808/foam_board_2p1mm_trajectory_20260714_002824_016099.mp4`
- Resolution: 640x480
- Frames: 6504
- Encoded duration: 215.36 seconds
- Timestamp records: 6504
- Timestamp duration: 222.30 seconds

The video contains repeated upward throws and descents, partial-frame targets, large close motion blur, hands, white
paper, the metal panel, and idle target-free periods. Corrupt analog-link frames and boxes on the fixed right panel or
hands were rejected during full-resolution contact-sheet review.

## Dataset

V6 copies the complete V5 dataset, then adds reviewed samples from the dynamic video.

- Existing V5 train/validation: 440/125
- New positive train/validation: 60/14
- New negative train/validation: 16/4
- V5-missed targets with manual boxes: 13
- Final train/validation: 516/143
- Final negative train/validation: 151/60
- Dataset: `datasets/foam_board_2p1mm/v6`
- Builder: `scripts/build_foam_board_2p1mm_v6.py`

The new samples are explicit frame-number whitelists. V5 boxes are used only after visual review, and clear V5 misses
are stored as manual blur-envelope boxes. Validation contains four V5-missed targets that are not used for training.

## Training

- Starting weights: `models/foam_board_2p1mm_v5.pt`
- Final weights: `models/foam_board_2p1mm_v6.pt`
- Training run: `runs/foam_board_2p1mm_v6`
- Best epoch: 16
- Device: NVIDIA GeForce RTX 4070 Laptop GPU
- Image size: 640
- Batch: 4
- Optimizer: AdamW, learning rate 0.0003

Batch sizes 16 and 8 exceeded the available 8 GB VRAM while GameViewer held about 1.98 GB of dedicated GPU memory.
Those diagnostic runs are retained as `runs/foam_board_2p1mm_v6_oom_batch16` and
`runs/foam_board_2p1mm_v6_oom_batch8`. Batch 4 completed all 20 epochs on the GPU.

V6 mixed validation metrics:

- Precision: 0.841
- Recall: 0.831
- mAP50: 0.831
- mAP50-95: 0.467

## V5/V6 Cross-Check

On the V5 validation set:

| Model | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| V5 | 0.895 | 0.928 | 0.895 | 0.454 |
| V6 | 0.850 | 0.928 | 0.862 | 0.487 |

On the V6 validation set:

| Model | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| V5 | 0.896 | 0.819 | 0.840 | 0.435 |
| V6 | 0.847 | 0.831 | 0.831 | 0.467 |

On the 14 new dynamic validation positives, at confidence 0.10 V5 detected 10 and V6 detected 12. At confidence
0.40 V5 detected 6 and V6 detected 9. V6 recovered two of the four held-out frames that V5 had completely missed.

At confidence 0.20 on the 56 target-free V5 validation images, false-positive images fell from 6 with V5 to 2 with
V6. V6 favors dynamic continuity and stricter localization at the cost of some aggregate precision. V5 remains
available at `models/foam_board_2p1mm_v5.pt` for rollback.
