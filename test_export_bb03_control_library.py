"""Regression coverage for BB03-only categorized audio export."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


class Bb03ControlLibraryExportTest(unittest.TestCase):
    def test_exports_bb03_tags_and_skips_other_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "source"
            source_dir.mkdir()
            candidates = root / "candidates.jsonl"
            output = root / "bb03-library"
            tags = (
                "speed_slow", "speed_normal", "speed_fast",
                "effort_soft", "effort_normal", "effort_strong",
            )
            with candidates.open("w", encoding="utf-8") as handle:
                for index, tag in enumerate(tags):
                    audio = source_dir / f"bb03-{index}.wav"
                    audio.write_bytes(f"bb03-{index}".encode("ascii"))
                    record = {
                        "control_candidate_id": f"bb03:item-{index}::{tag.split('_', 1)[0]}",
                        "control_tag": tag,
                        "control_kind": tag.split("_", 1)[0],
                        "source": "bb03",
                        "audio": str(audio),
                        "text": f"【neutral】测试文本{index}",
                        "control_metrics": {
                            "alignment": {"pause_excluded_cps": 2.0 + index},
                            "active_loudness": {"active_lufs_p50": -25.0 + index},
                        },
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

                foreign_audio = source_dir / "foreign.wav"
                foreign_audio.write_bytes(b"foreign")
                handle.write(json.dumps({
                    "control_candidate_id": "aopeng:foreign::speed",
                    "control_tag": "speed_fast",
                    "source": "aopeng",
                    "audio": str(foreign_audio),
                    "text": "【neutral】外部数据",
                }, ensure_ascii=False) + "\n")

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "export_bb03_control_library.py"),
                    "--input-jsonl", str(candidates),
                    "--output-dir", str(output),
                    "--copy-mode", "copy",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            for tag in tags:
                exports = list((output / "audio" / tag).glob("*.wav"))
                self.assertEqual(len(exports), 1, tag)
            self.assertFalse(any((output / "audio").rglob("*foreign*")))

            manifest_path = output / "manifests" / "bb03_library_manifest.csv"
            with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
                manifest_rows = list(csv.DictReader(handle))
            self.assertEqual(len(manifest_rows), 6)
            self.assertEqual({row["control_tag"] for row in manifest_rows}, set(tags))
            self.assertTrue(all(row["source"] == "bb03" for row in manifest_rows))

            copied_candidates = [
                json.loads(line)
                for line in (output / "manifests" / "bb03_control_candidates.jsonl")
                .read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(copied_candidates), 6)
            self.assertTrue(all(row["library_audio"].startswith("audio/") for row in copied_candidates))

            report = json.loads(
                (output / "manifests" / "bb03_export_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["skipped_non_bb03_records"], 1)
            self.assertEqual(report["exported_records"], 6)
            self.assertEqual(report["by_control_tag"], {tag: 1 for tag in tags})

    def test_rejects_an_alignment_manifest_instead_of_copying_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            alignment = root / "alignment.jsonl"
            alignment.write_text(
                json.dumps({"source": "bb03", "audio": "/missing.wav", "status": "ok"}) + "\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "export_bb03_control_library.py"),
                    "--input-jsonl", str(alignment),
                    "--output-dir", str(root / "output"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("build_control_candidates.py", result.stderr)


if __name__ == "__main__":
    unittest.main()
