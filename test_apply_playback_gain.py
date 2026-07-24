"""Dependency-free checks for apply_playback_gain argument and gain helpers."""

from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).with_name("apply_playback_gain.py")
SPEC = importlib.util.spec_from_file_location("apply_playback_gain", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ApplyPlaybackGainTests(unittest.TestCase):
    def test_fixed_gain_validation_and_scale(self) -> None:
        args = MODULE.parse_args(
            ["--input", "input.wav", "--output", "output.wav", "--gain-db", "6"]
        )
        MODULE.validate_numeric_args(args)
        self.assertAlmostEqual(MODULE.db_to_linear(args.gain_db), 10.0 ** (6.0 / 20.0))

    def test_target_lufs_and_gain_are_mutually_exclusive(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                MODULE.parse_args(
                    [
                        "--input",
                        "input.wav",
                        "--output",
                        "output.wav",
                        "--gain-db",
                        "3",
                        "--target-lufs",
                        "-16",
                    ]
                )

    def test_zero_dbfs_ceiling_is_rejected(self) -> None:
        args = MODULE.parse_args(
            [
                "--input",
                "input.wav",
                "--output",
                "output.wav",
                "--gain-db",
                "3",
                "--true-peak-dbfs",
                "0",
            ]
        )
        with self.assertRaisesRegex(ValueError, "true-peak-dbfs"):
            MODULE.validate_numeric_args(args)

    def test_peak_ceiling_reduces_an_over_limit_signal(self) -> None:
        try:
            import numpy as np
        except ImportError:  # pragma: no cover - NumPy is a runtime requirement
            self.skipTest("NumPy is not installed")
        audio = np.asarray([[0.9], [-0.9]], dtype=np.float64)
        result, reduction_db, final_peak, method = MODULE.apply_peak_ceiling(
            audio,
            ceiling_linear=0.5,
            true_peak_oversample=1,
            np=np,
            scipy_signal=None,
        )
        self.assertLess(reduction_db, 0.0)
        self.assertLessEqual(final_peak, 0.5)
        self.assertLessEqual(float(np.max(np.abs(result))), 0.5)
        self.assertEqual(method, "sample_peak")


if __name__ == "__main__":
    unittest.main()
