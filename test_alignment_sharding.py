"""Dependency-light coverage for deterministic alignment-manifest sharding."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from build_alignment_manifest import (  # noqa: E402
    _input_ordinal_in_shard,
    _validate_shard_settings,
    parse_args,
)


class AlignmentShardingTest(unittest.TestCase):
    def test_default_language_is_auto_and_explicit_override_is_preserved(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "build_alignment_manifest.py",
                "--input-jsonl",
                "input.jsonl",
                "--output-jsonl",
                "output.jsonl",
            ],
        ):
            defaults = parse_args()
        self.assertEqual(defaults.language, "auto")
        self.assertEqual((defaults.shard_count, defaults.shard_index), (1, 0))

        with patch.object(
            sys,
            "argv",
            [
                "build_alignment_manifest.py",
                "--input-jsonl",
                "input.jsonl",
                "--output-jsonl",
                "output.jsonl",
                "--language",
                "zh",
            ],
        ):
            explicit = parse_args()
        self.assertEqual(explicit.language, "zh")

    def test_stable_ordinal_sharding_and_argument_validation(self) -> None:
        selected = [
            ordinal
            for ordinal in range(10)
            if _input_ordinal_in_shard(ordinal, shard_count=3, shard_index=1)
        ]
        self.assertEqual(selected, [1, 4, 7])
        with self.assertRaises(ValueError):
            _validate_shard_settings(0, 0)
        with self.assertRaises(ValueError):
            _validate_shard_settings(3, 3)
        with self.assertRaises(ValueError):
            _input_ordinal_in_shard(-1, shard_count=3, shard_index=0)

    def test_cli_shards_non_empty_input_prefix_and_preserves_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            output = root / "shard-1.jsonl"
            rows = [
                {"key": f"item-{ordinal}", "text": "", "audio": f"missing-{ordinal}.wav"}
                for ordinal in range(5)
            ]
            # Blank lines intentionally do not affect the stable shard ordinal.
            source.write_text(
                "\n".join(["", *(json.dumps(row) for row in rows), ""]),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "build_alignment_manifest.py"),
                    "--input-jsonl",
                    str(source),
                    "--output-jsonl",
                    str(output),
                    "--max-records",
                    "5",
                    "--shard-count",
                    "3",
                    "--shard-index",
                    "1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["input_ordinal"] for row in manifest], [1, 4])
            self.assertEqual([row["key"] for row in manifest], ["item-1", "item-4"])
            self.assertEqual([row["input_key"] for row in manifest], ["item-1", "item-4"])
            self.assertTrue(all(row["shard_count"] == 3 and row["shard_index"] == 1 for row in manifest))
            self.assertIn("scanned_records=5", result.stdout)
            self.assertIn("selected_records=2", result.stdout)
            self.assertIn("written_records=2", result.stdout)

    def test_cli_excludes_requested_source_before_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            output = root / "filtered.jsonl"
            source.write_text(
                "\n".join(
                    [
                        json.dumps({"source": "bb03", "key": "bb", "text": "", "audio": "missing-bb.wav"}),
                        json.dumps({"source": "aopeng", "text": "", "audio": "missing-aopeng.wav"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "build_alignment_manifest.py"),
                    "--input-jsonl",
                    str(source),
                    "--output-jsonl",
                    str(output),
                    "--exclude-source",
                    "bb03",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["source"] for row in manifest], ["aopeng"])
            self.assertIn("excluded_source_records=1", result.stdout)


if __name__ == "__main__":
    unittest.main()
