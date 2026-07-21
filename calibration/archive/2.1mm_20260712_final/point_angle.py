import argparse

from calibration_utils import load_calibration, point_to_angles, undistort_points


def main():
    parser = argparse.ArgumentParser(description="Undistort one image point and compute camera-relative angles.")
    parser.add_argument("--calibration", required=True, help="Calibration JSON.")
    parser.add_argument("--u", type=float, required=True, help="Raw image x/u coordinate.")
    parser.add_argument("--v", type=float, required=True, help="Raw image y/v coordinate.")
    parser.add_argument("--yaw-offset", type=float, default=0.0, help="Optional installation yaw offset in degrees.")
    parser.add_argument("--pitch-offset", type=float, default=0.0, help="Optional installation pitch offset in degrees.")
    args = parser.parse_args()

    calibration_data, camera_matrix, dist_coeffs, _ = load_calibration(args.calibration)
    corrected = undistort_points([(args.u, args.v)], calibration_data, camera_matrix, dist_coeffs)[0]
    angle_x, angle_y = point_to_angles(corrected, camera_matrix)
    final_x = angle_x + args.yaw_offset
    final_y = angle_y + args.pitch_offset

    print(f"raw_point: [{args.u:.3f}, {args.v:.3f}]")
    print(f"corrected_point: [{corrected[0]:.3f}, {corrected[1]:.3f}]")
    print(f"camera_angle_deg: [{angle_x:.3f}, {angle_y:.3f}]")
    print(f"final_angle_deg: [{final_x:.3f}, {final_y:.3f}]")


if __name__ == "__main__":
    main()
