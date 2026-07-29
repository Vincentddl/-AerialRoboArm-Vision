from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "tracking"))

from bearing import BearingMapper  # noqa: E402
from trajectory import Detection, PixelToWorldMapper, TrajectoryEstimator  # noqa: E402


CALIBRATION = PROJECT_DIR / "configs" / "camera_2p1mm_640x480_fisheye.json"


class TrajectoryPredictionTest(unittest.TestCase):
    def test_calibrated_mapper_uses_the_same_undistorted_angles_as_bearing_mapper(self) -> None:
        point = (80.0, 80.0)
        trajectory_mapper = PixelToWorldMapper(str(CALIBRATION))
        bearing_mapper = BearingMapper(CALIBRATION)

        self.assertEqual(trajectory_mapper.unit, "deg")
        actual = trajectory_mapper.to_world(point)
        expected = bearing_mapper.pixel_to_angles(point)
        self.assertAlmostEqual(actual[0], expected[0], places=9)
        self.assertAlmostEqual(actual[1], expected[1], places=9)

    def test_arm_target_contains_a_0p4_second_path_and_endpoint(self) -> None:
        mapper = PixelToWorldMapper(str(CALIBRATION))
        estimator = TrajectoryEstimator(mapper=mapper)
        detection = Detection((300.0, 200.0, 340.0, 240.0), 0.95, 0, "foam_board")
        estimator.update([detection], timestamp=0.0)
        track = estimator.tracks[0]
        track.age = 8
        track.state = np.asarray([320.0, 220.0, 10.0, 100.0])
        track._sync_from_state(shift_bbox=False)

        target = estimator.arm_targets(predict_seconds=0.4, min_age=1, trajectory_steps=4)[0]

        self.assertEqual(target["predict_seconds"], 0.4)
        self.assertEqual(len(target["trajectory_pixel"]), 5)
        self.assertEqual(len(target["trajectory_world"]), 5)
        self.assertEqual(target["trajectory_pixel"][-1], target["predicted_pixel"])
        self.assertEqual(target["trajectory_world"][-1], target["predicted_world"])
        # CV: 320 + 10*0.4 = 324, 220 + 100*0.4 = 260
        self.assertEqual(target["predicted_pixel"], [324.0, 260.0])
        self.assertEqual(target["world_unit"], "deg")
        self.assertEqual(
            target["predicted_angular_displacement_deg"],
            [
                round(target["predicted_angle_deg"][0] - target["angle_deg"][0], 3),
                round(target["predicted_angle_deg"][1] - target["angle_deg"][1], 3),
            ],
        )
        self.assertAlmostEqual(
            target["offset_angle_deg"],
            mapper.optical_axis_offset_deg(mapper.to_world(track.center)),
            places=3,
        )
        self.assertAlmostEqual(
            target["predicted_offset_angle_deg"],
            mapper.optical_axis_offset_deg(
                mapper.to_world(tuple(target["predicted_pixel"]))
            ),
            places=3,
        )
        self.assertEqual(
            target["offset_angle_convention"],
            "unsigned 3-D angle from camera optical axis",
        )

    def test_camera_mount_elevation_does_not_change_optical_axis_offset(self) -> None:
        mapper = PixelToWorldMapper(str(CALIBRATION))
        estimator = TrajectoryEstimator(mapper=mapper)
        detection = Detection((300.0, 200.0, 340.0, 240.0), 0.95, 0, "foam_board")
        estimator.update([detection], timestamp=0.0)

        level_target = estimator.arm_targets(
            predict_seconds=0.4,
            min_age=1,
            camera_elevation_deg=0.0,
        )[0]
        tilted_target = estimator.arm_targets(
            predict_seconds=0.4,
            min_age=1,
            camera_elevation_deg=7.5,
        )[0]

        self.assertAlmostEqual(
            tilted_target["offset_angle_deg"],
            level_target["offset_angle_deg"],
            places=9,
        )

    def test_principal_point_has_zero_optical_axis_offset(self) -> None:
        mapper = PixelToWorldMapper(str(CALIBRATION))
        self.assertIsNotNone(mapper.principal_point)
        angles = mapper.to_world(mapper.principal_point)

        self.assertAlmostEqual(mapper.optical_axis_offset_deg(angles), 0.0, places=7)


if __name__ == "__main__":
    unittest.main()
