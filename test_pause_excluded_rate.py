"""Focused tests for CTC-character/VAD-confirmed pause exclusion."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from build_alignment_manifest import (  # noqa: E402
    _active_audio_for_window,
    _character_pause_rate_metrics,
    _paralinguistic_active_metrics,
)


class CharacterPauseRateTest(unittest.TestCase):
    def setUp(self) -> None:
        # The 1.0 s gap between character 0 and 1 represents an intra-sentence
        # silence; the first and last character still define the 1.6 s span.
        self.rows = [
            {"index": 0, "start_sec": 0.0, "end_sec": 0.2},
            {"index": 1, "start_sec": 1.2, "end_sec": 1.4},
            {"index": 2, "start_sec": 1.4, "end_sec": 1.6},
        ]

    def test_excludes_long_gap_when_vad_is_mostly_inactive(self) -> None:
        metrics = _character_pause_rate_metrics(
            self.rows,
            # Only 0.10 s of the 1.0 s character gap is VAD active.
            [(0.0, 0.25), (1.15, 1.6)],
            0.0,
            1.6,
            spoken_unit_count=6,
            min_gap_sec=0.30,
            max_vad_active_ratio=0.20,
        )
        self.assertEqual(metrics["char_pause_count"], 1)
        self.assertAlmostEqual(metrics["char_pause_duration_sec"], 1.0)
        self.assertAlmostEqual(metrics["char_pause_ratio"], 0.625)
        self.assertAlmostEqual(metrics["pause_excluded_duration_sec"], 0.6)
        self.assertAlmostEqual(metrics["pause_excluded_cps"], 10.0)
        pause = metrics["char_pause_intervals"][0]
        self.assertEqual(pause["after_character_index"], 0)
        self.assertEqual(pause["before_character_index"], 1)
        self.assertAlmostEqual(pause["vad_active_ratio"], 0.10)

    def test_keeps_gap_when_vad_shows_continued_activity(self) -> None:
        metrics = _character_pause_rate_metrics(
            self.rows,
            [(0.2, 1.2)],
            0.0,
            1.6,
            spoken_unit_count=6,
            min_gap_sec=0.30,
            max_vad_active_ratio=0.20,
        )
        self.assertEqual(metrics["char_pause_count"], 0)
        self.assertEqual(metrics["char_pause_intervals"], [])
        self.assertAlmostEqual(metrics["char_pause_duration_sec"], 0.0)
        self.assertAlmostEqual(metrics["pause_excluded_duration_sec"], 1.6)
        self.assertAlmostEqual(metrics["pause_excluded_cps"], 3.75)

    def test_missing_character_timestamp_breaks_pause_adjacency(self) -> None:
        rows = [
            {"index": 0, "start_sec": 0.0, "end_sec": 0.2},
            {"index": 1, "start_sec": None, "end_sec": None},
            {"index": 2, "start_sec": 1.2, "end_sec": 1.4},
        ]
        metrics = _character_pause_rate_metrics(
            rows,
            [],
            0.0,
            1.4,
            spoken_unit_count=3,
            min_gap_sec=0.30,
            max_vad_active_ratio=0.20,
        )
        self.assertEqual(metrics["char_pause_count"], 0)
        self.assertAlmostEqual(metrics["pause_excluded_duration_sec"], 1.4)

    def test_rejects_invalid_thresholds(self) -> None:
        with self.assertRaises(ValueError):
            _character_pause_rate_metrics(
                self.rows, [], 0.0, 1.6, 6, 0.0, 0.20
            )
        with self.assertRaises(ValueError):
            _character_pause_rate_metrics(
                self.rows, [], 0.0, 1.6, 6, 0.30, math.inf
            )

    def test_measures_active_nonlexical_sound_between_character_windows(self) -> None:
        metrics = _paralinguistic_active_metrics(
            [
                {"start_sec": 0.0, "end_sec": 0.2},
                {"start_sec": 0.8, "end_sec": 1.0},
            ],
            # The middle 0.30 s is active but has no nearby aligned character.
            [(0.0, 0.2), (0.3, 0.6), (0.8, 1.0)],
            0.0,
            1.0,
            character_padding_sec=0.05,
        )
        self.assertAlmostEqual(metrics["lexical_vad_active_duration_sec"], 0.4)
        self.assertAlmostEqual(metrics["paralinguistic_active_duration_sec"], 0.3)
        self.assertAlmostEqual(metrics["paralinguistic_ratio"], 3.0 / 7.0)

    def test_active_loudness_input_removes_silence_samples(self) -> None:
        import numpy as np

        audio = np.arange(10, dtype=np.float64)
        active = _active_audio_for_window(
            audio,
            sample_rate=10,
            active_intervals=[(0.0, 0.2), (0.5, 0.7)],
            lower=0.0,
            upper=1.0,
            np=np,
        )
        self.assertListEqual(active.tolist(), [0.0, 1.0, 5.0, 6.0])


if __name__ == "__main__":
    unittest.main()
