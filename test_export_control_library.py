"""Regression coverage for categorized candidate-audio export."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


class ControlLibraryExportTest(unittest.TestCase):
    def test_exports_every_tag_and_review_subsets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "source"
            source_dir.mkdir()
            candidates = root / "candidates.jsonl"
            output = root / "library"
            tags = (
                "speed_slow", "speed_normal", "speed_fast",
                "effort_soft", "effort_normal", "effort_strong",
            )
            with candidates.open("w", encoding="utf-8") as handle:
                for index, tag in enumerate(tags):
                    audio = source_dir / f"sample-{index}.wav"
                    audio.write_bytes(f"audio-{index}".encode("ascii"))
                    record = {
                        "control_candidate_id": f"bb03:sample-{index}::{tag.split('_', 1)[0]}",
                        "control_tag": tag,
                        "control_kind": tag.split("_", 1)[0],
                        "source": "bb03",
                        "audio": str(audio),
                        "text": f"【neutral】测试文本{index}",
                        "control_selection": {
                            "stratum": {
                                "source": "bb03",
                                "emotion_family": "neutral",
                                "duration_bin": "short",
                            }
                        },
                        "control_metrics": {
                            "alignment": {
                                "speed_rate_cps": 2.0 + index,
                                "asr_cer": 0.01,
                                "alignment_coverage": 1.0,
                                "pause_excluded_cps": 2.0 + index,
                            },
                            "active_loudness": {
                                "active_rms_dbfs_p50": -30.0 + index,
                                "active_lufs_p50": -25.0 + index,
                            },
                        },
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "export_control_library.py"),
                    "--input-jsonl", str(candidates),
                    "--output-dir", str(output),
                    "--copy-mode", "copy",
                    "--speed-review-per-tag", "1",
                    "--effort-review-per-tag", "1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            for tag in tags:
                exports = list((output / "audio" / tag / "bb03").glob("*.wav"))
                self.assertEqual(len(exports), 1, tag)
            with (output / "manifests" / "library_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
                manifest_rows = list(csv.DictReader(handle))
            self.assertEqual(len(manifest_rows), 6)
            self.assertEqual({row["control_tag"] for row in manifest_rows}, set(tags))
            with (output / "review" / "speed_review.csv").open(encoding="utf-8-sig", newline="") as handle:
                speed_rows = list(csv.DictReader(handle))
            with (output / "review" / "effort_review.csv").open(encoding="utf-8-sig", newline="") as handle:
                effort_rows = list(csv.DictReader(handle))
            self.assertEqual(len(speed_rows), 3)
            self.assertEqual(len(effort_rows), 3)
            self.assertTrue((output / "review" / "speed_review.html").is_file())
            self.assertTrue((output / "review" / "effort_review_candidates.jsonl").is_file())

    def test_refuses_nonempty_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidates = root / "candidates.jsonl"
            candidates.write_text("", encoding="utf-8")
            output = root / "library"
            output.mkdir()
            (output / "existing.txt").write_text("keep", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "export_control_library.py"),
                    "--input-jsonl", str(candidates),
                    "--output-dir", str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not empty", result.stderr)


if __name__ == "__main__":
    unittest.main()
