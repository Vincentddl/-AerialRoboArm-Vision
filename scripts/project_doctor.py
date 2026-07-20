from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ULTRALYTICS = "8.3.218"

CRITICAL_PATHS = {
    "calibration script": ROOT / "calibration" / "calibrate.py",
    "capture script": ROOT / "calibration" / "capture_chessboard.py",
    "runtime calibration": ROOT / "configs" / "camera_2p1mm_640x480_fisheye.json",
    "V7 model": ROOT / "models" / "foam_board_2p1mm_v7.pt",
    "tracking runtime": ROOT / "scripts" / "run_foam_board.py",
    "trajectory baseline": ROOT / "tracking" / "trajectory.py",
    "bearing predictor": ROOT / "tracking" / "bearing.py",
    "bearing evaluator": ROOT / "scripts" / "evaluate_bearing_prediction.py",
    "bearing dataset builder": ROOT / "scripts" / "build_bearing_estimation_dataset.py",
    "bearing trainer": ROOT / "scripts" / "train_bearing_estimator.py",
    "V7 dataset": ROOT / "datasets" / "foam_board_2p1mm" / "v7" / "foam_board_2p1mm.yaml",
    "Z-axis recording": ROOT
    / "datasets"
    / "foam_board_2p1mm_zaxis"
    / "raw"
    / "20260714_223253"
    / "foam_board_2p1mm_zaxis_trajectory_20260714_223351_701282.mp4",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dependency_version(name: str) -> str:
    module = importlib.import_module(name)
    return str(getattr(module, "__version__", "unknown"))


def main() -> int:
    failures = 0
    print(f"project: {ROOT}")
    print(f"python: {sys.version.split()[0]}")

    for name in ("cv2", "numpy", "torch", "ultralytics"):
        try:
            print(f"{name}: {dependency_version(name)}")
        except Exception as exc:  # Environment diagnostics should report every failure.
            failures += 1
            print(f"ERROR dependency {name}: {exc}")

    try:
        ultralytics_version = dependency_version("ultralytics")
        if ultralytics_version != EXPECTED_ULTRALYTICS:
            print(
                f"WARNING ultralytics is {ultralytics_version}; "
                f"the archived project was developed with {EXPECTED_ULTRALYTICS}"
            )
    except Exception:
        pass

    for label, path in CRITICAL_PATHS.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024) if path.is_file() else 0.0
            print(f"OK {label}: {path.relative_to(ROOT)} ({size_mb:.2f} MB)")
        else:
            failures += 1
            print(f"ERROR missing {label}: {path}")

    runtime_calibration = CRITICAL_PATHS["runtime calibration"]
    archived_calibration = ROOT / "calibration" / "calibration_2p1mm_fisheye_final.json"
    if runtime_calibration.exists():
        try:
            data = json.loads(runtime_calibration.read_text(encoding="utf-8"))
            print(
                "calibration: "
                f"model={data.get('model')} size={data.get('image_size')} "
                f"error={data.get('reprojection_error_px')} px"
            )
        except Exception as exc:
            failures += 1
            print(f"ERROR invalid calibration JSON: {exc}")

    if runtime_calibration.exists() and archived_calibration.exists():
        if sha256(runtime_calibration) == sha256(archived_calibration):
            print("OK runtime calibration matches the archived final calibration")
        else:
            print("WARNING runtime calibration differs from the archived final calibration")

    if failures:
        print(f"FAILED: {failures} critical problem(s)")
        return 1
    print("PASS: merged project is structurally complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
