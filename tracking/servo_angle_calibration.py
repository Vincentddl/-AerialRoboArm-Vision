"""Empirical mapping between servo commands and camera-axis target angles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class ServoOpticalAngleCalibration:
    """Linear mapping fitted from labelled servo/camera calibration footage.

    ``optical_offset_deg`` is the signed target bearing in the camera's
    single-axis plane. Positive values point downward in the image. This is
    deliberately separate from the legacy mechanical-horizontal angle.
    """

    name: str
    version: int
    slope: float
    intercept_deg: float
    servo_range_deg: Tuple[float, float]
    boundary_tolerance_deg: float = 0.0
    fit_rmse_deg: float | None = None
    fit_r_squared: float | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "ServoOpticalAngleCalibration":
        calibration_path = Path(path)
        data = json.loads(calibration_path.read_text(encoding="utf-8"))
        if data.get("type") != "servo_optical_angle_linear":
            raise ValueError(
                "servo calibration type must be 'servo_optical_angle_linear'"
            )

        model = data.get("model", {})
        valid_range = data.get("valid_range", {}).get("servo_command_deg")
        if not isinstance(valid_range, list) or len(valid_range) != 2:
            raise ValueError(
                "valid_range.servo_command_deg must contain two numbers"
            )

        slope = float(model["slope"])
        if abs(slope) < 1e-12:
            raise ValueError("servo calibration slope must be non-zero")

        fit = data.get("fit", {})
        limits = sorted(float(value) for value in valid_range)
        return cls(
            name=str(data.get("name", calibration_path.stem)),
            version=int(data.get("version", 1)),
            slope=slope,
            intercept_deg=float(model["intercept_deg"]),
            servo_range_deg=(limits[0], limits[1]),
            boundary_tolerance_deg=max(
                0.0, float(data.get("boundary_tolerance_deg", 0.0))
            ),
            fit_rmse_deg=(
                float(fit["rmse_deg"]) if fit.get("rmse_deg") is not None else None
            ),
            fit_r_squared=(
                float(fit["r_squared"])
                if fit.get("r_squared") is not None
                else None
            ),
        )

    def optical_offset_from_servo(self, servo_command_deg: float) -> float:
        """Return signed optical-axis offset for a labelled servo command."""
        return self.slope * float(servo_command_deg) + self.intercept_deg

    def servo_from_optical_offset(self, optical_offset_deg: float) -> float:
        """Return the servo command that aligns with a measured target bearing."""
        return (float(optical_offset_deg) - self.intercept_deg) / self.slope

    def is_servo_in_range(self, servo_command_deg: float) -> bool:
        minimum, maximum = self.servo_range_deg
        tolerance = self.boundary_tolerance_deg
        return (
            minimum - tolerance
            <= float(servo_command_deg)
            <= maximum + tolerance
        )

    @property
    def optical_range_deg(self) -> Tuple[float, float]:
        values = [
            self.optical_offset_from_servo(limit)
            for limit in self.servo_range_deg
        ]
        return (min(values), max(values))

    @property
    def summary(self) -> str:
        minimum, maximum = self.servo_range_deg
        return (
            f"servo calibration: {self.name} v{self.version}, "
            f"beta={self.slope:.6f}*g{self.intercept_deg:+.6f} deg, "
            f"fitted g=[{minimum:.1f}, {maximum:.1f}] deg "
            f"(boundary tolerance {self.boundary_tolerance_deg:.2f} deg)"
        )
