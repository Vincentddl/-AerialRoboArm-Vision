# Bearing Estimator V1

## Scope

V1 temporarily disables the 400 ms objective and learns future camera bearing changes at 100 ms and 200 ms. Short trajectories are retained whenever they contain enough history and a matching future observation.

Inputs are the recent relative time, yaw, pitch, V7 confidence, and edge-clipping flag. Targets are future yaw and pitch changes. Samples touching an image edge are retained with lower loss weight.

Data are split by complete detected trajectory, not by neighboring frames. This prevents adjacent observations from the same throw appearing in both training and evaluation.

## Dataset

- Source sessions: 2
- Accepted moving segments: 118
- Total samples: 1015
- 100 ms samples: 608
- 200 ms samples: 407
- Train/validation/test samples: 733/67/215
- Train/validation/test trajectory groups: 71/7/16

All targets are V7 detection centers converted through the 2.1 mm fisheye calibration. They remain pseudo-labels rather than manually verified physical truth.

## Local Smoke Training

The local RTX 4070 run stopped at epoch 72. On the grouped test split:

| Horizon | Method | Median error | P90 error |
|---|---|---:|---:|
| 100 ms | Hold current angle | 8.52 deg | 27.10 deg |
| 100 ms | Learned GRU | 7.03 deg | 15.76 deg |
| 200 ms | Hold current angle | 10.07 deg | 36.55 deg |
| 200 ms | Learned GRU | 11.51 deg | 23.77 deg |

Across both horizons, the learned median is 8.36 deg versus 8.83 deg for hold, an improvement of only 5.3%. V1 reduces large tail errors but has not established reliable median improvement at 200 ms. It must not drive the robot.

## Commands

```powershell
python scripts/build_bearing_estimation_dataset.py
python scripts/train_bearing_estimator.py --device cuda
```

The generated checkpoint is `models/bearing_estimator_v1.pt`; the detailed report is `outputs/bearing_estimator_v1_training.json`.

## Remote Reproduction

The same grouped dataset was trained on host `WhiteHeart` with an NVIDIA RTX 4070 SUPER and PyTorch 2.6.0+cu124. Early stopping selected the same effective model behavior; the remote test median was 8.36 degrees and P90 was 18.47 degrees. The downloaded artifacts are:

- `models/bearing_estimator_v1_remote.pt`
- `outputs/bearing_estimator_v1_remote_training.json`

The local SSH alias is `aerial-vision`. Connect with `ssh aerial-vision`; authentication uses the dedicated local Ed25519 key rather than a password stored in the project.
