"""
JSONL Diagnostic Logger for ZFrame Registration

Provides structured logging of per-slice diagnostics, failure codes, and summary statistics.
"""

import json
import os
from datetime import datetime
from pathlib import Path
import numpy as np


class ZFrameDiagnosticLogger:
    """Logger that produces JSONL diagnostic output for ZFrame registration."""

    def __init__(self, output_path=None, enabled=True):
        """
        Initialize the diagnostic logger.

        Args:
            output_path: Path for JSONL output. If None, uses ~/zframe_diagnostics/zframe_reg_YYYYMMDD_HHMMSS.jsonl
            enabled: If False, all logging methods become no-ops.
        """
        self.enabled = enabled
        self.records = []
        self.session_params = {}
        self.session_start_time = None
        self.current_slice_idx = None
        self.current_slice_data = {}

        if output_path:
            self.output_path = Path(output_path)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_dir = Path.home() / "zframe_diagnostics"
            default_dir.mkdir(parents=True, exist_ok=True)
            self.output_path = default_dir / f"zframe_reg_{timestamp}.jsonl"

    def start_session(self, params: dict):
        """
        Start a new registration session.

        Args:
            params: Dictionary containing session parameters like num_fiducials,
                    marker_diameter, orientation, frame_config, slice_range, etc.
        """
        if not self.enabled:
            return

        self.session_start_time = datetime.now()
        self.session_params = params.copy()
        self.records = []

    def start_slice(self, slice_idx: int):
        """
        Start processing a new slice.

        Args:
            slice_idx: The slice index being processed.
        """
        if not self.enabled:
            return

        self.current_slice_idx = slice_idx
        self.current_slice_data = {
            "type": "slice",
            "slice_idx": slice_idx,
            "timestamp": datetime.now().isoformat(),
            "status": None,
            "fail_code": None,
            "locate_fiducials": None,
            "order_fid_points": None,
            "fiducial_count": None,
            "detected_points_ijk": None,
            "template_points_frame": None,
            "transform_4x4": None,
            "position_ras": None,
            "quaternion": None,
            "residual_mm": None
        }

    def log_locate_fiducials(self, peaks, threshold, weak_retries, candidates):
        """
        Log data from LocateFiducials function.

        Args:
            peaks: List of dicts with keys: idx, x, y, correlation_value
            threshold: The threshold value used (e.g., 0.3)
            weak_retries: Number of weak peak retries encountered
            candidates: List of fiducial candidate coordinates [[x,y], ...]
        """
        if not self.enabled:
            return

        self.current_slice_data["locate_fiducials"] = {
            "threshold_used": threshold,
            "peaks": peaks,
            "weak_peak_retries": weak_retries,
            "fiducial_candidates": candidates
        }

    def log_order_fid_points(self, pall_initial, pall_final, pother_final, missing_mid_slots):
        """
        Log data from OrderFidPoints function.

        Args:
            pall_initial: Initial pall array state
            pall_final: Final pall array state
            pother_final: Final pother array state
            missing_mid_slots: List of indices where midpoints were not assigned
        """
        if not self.enabled:
            return

        self.current_slice_data["order_fid_points"] = {
            "pall_initial": list(pall_initial) if pall_initial is not None else None,
            "pall_final": list(pall_final) if pall_final is not None else None,
            "pother_final": list(pother_final) if pother_final is not None else None,
            "missing_mid_slots": list(missing_mid_slots) if missing_mid_slots is not None else None
        }

    def log_slice_success(self, transform, position, quaternion, detected_points, template_points, residual):
        """
        Log successful slice registration.

        Args:
            transform: 4x4 transformation matrix (numpy array or list)
            position: [px, py, pz] position in RAS
            quaternion: [qx, qy, qz, qw] quaternion
            detected_points: List of detected fiducial coordinates
            template_points: List of template frame points
            residual: Residual error in mm
        """
        if not self.enabled:
            return

        self.current_slice_data["status"] = "SUCCESS"
        self.current_slice_data["fail_code"] = None

        # Convert numpy arrays to lists for JSON serialization
        if transform is not None:
            if hasattr(transform, 'tolist'):
                self.current_slice_data["transform_4x4"] = transform.tolist()
            else:
                self.current_slice_data["transform_4x4"] = [list(row) for row in transform]

        if position is not None:
            self.current_slice_data["position_ras"] = list(position) if hasattr(position, '__iter__') else position

        if quaternion is not None:
            self.current_slice_data["quaternion"] = list(quaternion) if hasattr(quaternion, '__iter__') else quaternion

        if detected_points is not None:
            self.current_slice_data["detected_points_ijk"] = [
                list(pt) if hasattr(pt, '__iter__') else pt for pt in detected_points
            ]
            self.current_slice_data["fiducial_count"] = len(detected_points)

        if template_points is not None:
            self.current_slice_data["template_points_frame"] = [
                list(pt) if hasattr(pt, '__iter__') else pt for pt in template_points
            ]

        self.current_slice_data["residual_mm"] = float(residual) if residual is not None else None

        # Add record
        self.records.append(self.current_slice_data.copy())

    def log_slice_failure(self, fail_code: str, details: dict = None):
        """
        Log failed slice registration.

        Args:
            fail_code: One of: FFT_DIVIDE_ZERO, PEAK_TOO_WEAK, PEAK_VALUE_ZERO,
                       GEOMETRY_FAIL, LOCALIZE_FAIL, LOCATE_FIDUCIALS_FAIL
            details: Optional dictionary with additional failure details
        """
        if not self.enabled:
            return

        self.current_slice_data["status"] = "FAIL"
        self.current_slice_data["fail_code"] = fail_code

        if details:
            self.current_slice_data["failure_details"] = details

        # Add record
        self.records.append(self.current_slice_data.copy())

    def end_session(self):
        """
        End the registration session, compute summary, and write JSONL file.
        """
        if not self.enabled:
            return

        summary = self.compute_summary()
        self.records.append(summary)
        self.write_jsonl()

    def compute_summary(self) -> dict:
        """
        Compute summary statistics from all recorded slices.

        Returns:
            Dictionary containing summary statistics.
        """
        slice_records = [r for r in self.records if r.get("type") == "slice"]
        total_slices = len(slice_records)
        success_count = sum(1 for r in slice_records if r.get("status") == "SUCCESS")
        fail_count = total_slices - success_count

        # Fail code distribution
        fail_codes = {}
        for r in slice_records:
            if r.get("fail_code"):
                code = r["fail_code"]
                fail_codes[code] = fail_codes.get(code, 0) + 1

        # Transform statistics (from successful slices)
        translations = []
        rotations_euler = []

        for r in slice_records:
            if r.get("status") == "SUCCESS" and r.get("transform_4x4"):
                decomposed = self.decompose_transform(r["transform_4x4"])
                if decomposed:
                    translations.append(decomposed["translation"])
                    rotations_euler.append(decomposed["euler_angles_deg"])

        transform_stats = {}
        if translations:
            trans_arr = np.array(translations)
            transform_stats["translation"] = {
                "mean": trans_arr.mean(axis=0).tolist(),
                "std": trans_arr.std(axis=0).tolist()
            }

        if rotations_euler:
            rot_arr = np.array(rotations_euler)
            transform_stats["rotation_euler_deg"] = {
                "mean": rot_arr.mean(axis=0).tolist(),
                "std": rot_arr.std(axis=0).tolist()
            }

        # RMS error (from residuals)
        residuals = [r.get("residual_mm") for r in slice_records
                     if r.get("status") == "SUCCESS" and r.get("residual_mm") is not None]
        rms_error = None
        if residuals:
            rms_error = float(np.sqrt(np.mean(np.array(residuals) ** 2)))

        summary = {
            "type": "summary",
            "total_slices": total_slices,
            "success_count": success_count,
            "fail_count": fail_count,
            "success_rate": success_count / total_slices if total_slices > 0 else 0.0,
            "fail_code_distribution": fail_codes,
            "transform_statistics": transform_stats,
            "rms_error_mm": rms_error,
            "parameters": self.session_params
        }

        return summary

    def write_jsonl(self):
        """
        Write all records to the JSONL output file.
        """
        if not self.enabled:
            return

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, 'w') as f:
            for record in self.records:
                f.write(json.dumps(record, default=self._json_serializer) + '\n')

    @staticmethod
    def decompose_transform(matrix_4x4) -> dict:
        """
        Extract translation and euler angles from a 4x4 transformation matrix.

        Args:
            matrix_4x4: 4x4 transformation matrix (list of lists or numpy array)

        Returns:
            Dictionary with 'translation' [tx, ty, tz] and 'euler_angles_deg' [rx, ry, rz]
        """
        if matrix_4x4 is None:
            return None

        m = np.array(matrix_4x4)
        if m.shape != (4, 4):
            return None

        # Extract translation
        translation = m[:3, 3].tolist()

        # Extract rotation matrix
        R = m[:3, :3]

        # Convert to Euler angles (ZYX convention, commonly used)
        # This gives roll, pitch, yaw
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            rx = np.arctan2(R[2, 1], R[2, 2])
            ry = np.arctan2(-R[2, 0], sy)
            rz = np.arctan2(R[1, 0], R[0, 0])
        else:
            rx = np.arctan2(-R[1, 2], R[1, 1])
            ry = np.arctan2(-R[2, 0], sy)
            rz = 0

        euler_deg = [np.degrees(rx), np.degrees(ry), np.degrees(rz)]

        return {
            "translation": translation,
            "euler_angles_deg": euler_deg
        }

    @staticmethod
    def _json_serializer(obj):
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
