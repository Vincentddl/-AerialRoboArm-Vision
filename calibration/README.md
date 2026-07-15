# Camera Calibration And Lens Correction

This folder contains a small OpenCV calibration workflow for the analog/FOV camera used by the robot-arm vision project.

It supports two calibration models:

- `pinhole`: OpenCV ordinary camera model, using `cv2.calibrateCamera`.
- `fisheye`: OpenCV fisheye model, using `cv2.fisheye.calibrate`.

For a 120 degree camera, test both models and keep the one that gives better straight lines, lower reprojection error, and better angle accuracy near the grab window.

## 1. Print And Measure The Checkerboard

The generated PDF in this repo is:

```text
output/pdf/a4_checkerboard_9x6_inner_20mm.pdf
```

Its intended parameters are:

```text
squares: 10 x 7
inner corners: 9 x 6
square size: 20 mm
```

OpenCV uses inner corners, so use:

```text
--cols 9 --rows 6
```

Important: after printing, measure the 100 mm ruler on the page.

If it measures 100 mm:

```text
--square-size 20
```

If it measures 97 mm:

```text
actual square size = 20 * 97 / 100 = 19.4 mm
--square-size 19.4
```

Print settings:

```text
paper: A4
orientation: landscape
scale: actual size / 100%
do not use: fit to page / shrink to printable area
```

Mount the paper on a flat board. Do not use curved paper for calibration.

## 2. Capture Calibration Images

The checkerboard does not need to face the camera directly in every image. In fact, using only front-facing images is bad.

Good calibration images:

```text
checkerboard is complete
all 9 x 6 inner corners are visible
corners are sharp, not blurry
no strong glare or reflection
board is flat
board appears in center, edges, and corners
board has different tilts
board has different distances
```

Recommended count:

```text
ordinary camera: 15-25 usable images
120 degree wide camera: 25-40 usable images
```

You do not need to label the images. OpenCV automatically detects checkerboard corners.

Capture from camera:

```powershell
python camera_calibration\capture_chessboard.py --source 0 --out camera_calibration\images --cols 9 --rows 6
```

Controls:

```text
s: save current frame
q or Esc: quit
```

If using a UVC receiver or capture card, the source index may be `1`, `2`, etc.

## 3. Calibrate

Ordinary pinhole model:

```powershell
python camera_calibration\calibrate.py --images camera_calibration\images --model pinhole --cols 9 --rows 6 --square-size 20 --out camera_calibration\calibration_pinhole.json
```

Fisheye model:

```powershell
python camera_calibration\calibrate.py --images camera_calibration\images --model fisheye --cols 9 --rows 6 --square-size 20 --out camera_calibration\calibration_fisheye.json
```

If your printed square size is 19.4 mm, use:

```powershell
--square-size 19.4
```

The calibration file contains:

```text
model
image_size
camera_matrix
dist_coeffs
reprojection_error
valid_image_count
```

## 4. Preview Correction

Single image:

```powershell
python camera_calibration\undistort.py --calibration camera_calibration\calibration_pinhole.json --source camera_calibration\images\frame_0001.png --output camera_calibration\outputs\undistorted.png
```

Live camera:

```powershell
python camera_calibration\undistort.py --calibration camera_calibration\calibration_pinhole.json --source 0
```

Controls:

```text
q or Esc: quit
```

## 5. Use Only Point Correction For Grab Angle

For the robot-arm project, you often do not need to undistort the whole frame. A faster and cleaner method is:

```text
raw frame -> YOLO detects target -> target center u,v -> undistort only this point -> compute angle
```

Use:

```powershell
python camera_calibration\point_angle.py --calibration camera_calibration\calibration_pinhole.json --u 320 --v 240
```

Formula after correction:

```text
angle_x = atan((u_corrected - cx) / fx)
angle_y = atan((v_corrected - cy) / fy)
```

If the camera is tilted relative to the robot arm, add installation offsets:

```text
final_angle_x = angle_x + yaw_offset
final_angle_y = angle_y + pitch_offset
```

For a one-axis robot arm, a practical approach is often:

```text
corrected target center -> empirical mapping -> arm_pos
```

## 6. Model Selection Notes

Use `pinhole` if:

```text
the lens is wide but not extreme
correction looks stable
checkerboard straight lines look good after correction
reprojection error is acceptable
```

Use `fisheye` if:

```text
the image has strong fish-eye bending
pinhole correction leaves obvious curved edges
wide-edge angle accuracy is poor
```

For 120 degree cameras, do not assume fisheye is always better. Test both.

## 7. Common Mistakes

Avoid:

```text
moving the camera after calibration
changing resolution after calibration
changing lens focus or replacing lens after calibration
using only center/front-facing checkerboard images
using wrinkled A4 paper
using heavily blurred analog video frames
using printed square size without measuring actual square size
using edge detections for final grab trigger without correction
```

If any of these change, recalibrate:

```text
camera position
camera angle
lens
focus
resolution
capture card crop mode
```

## 8. Recommended Robot-Arm Use

For the current one-axis robot arm:

```text
1. Calibrate lens distortion with this folder.
2. Keep the camera fixed.
3. Use YOLO to detect the foam ball.
4. Correct the detected center point.
5. Convert corrected point to angle or empirical arm_pos.
6. Let the gripper-end ToF confirm the object before closing.
```

For a 120 degree lens:

```text
Use edges for early detection.
Use center/grab window for final trigger.
```

