"""Dependency-light regression tests for the control-data pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    add_control_tags,
    clean_spoken_text,
    event_tags,
    infer_recording_group,
    infer_source,
    is_speed_excluded,
    normalized_transcript,
    record_identity_variants,
)
from build_control_candidates import _style_rejection  # noqa: E402


class ControlPipelineTest(unittest.TestCase):
    def test_candidate_style_requires_paralinguistic_ratio(self) -> None:
        args = type("Args", (), {"max_paralinguistic_ratio": 0.15})()
        self.assertEqual(
            _style_rejection({"text": "\u4e00\u4e8c\u4e09\u56db"}, args),
            "missing_paralinguistic_ratio",
        )
        self.assertEqual(
            _style_rejection(
                {"text": "\u4e00\u4e8c\u4e09\u56db", "paralinguistic_ratio": 0.16}, args
            ),
            "long_paralinguistic_ratio",
        )
        self.assertEqual(
            _style_rejection(
                {"text": "\u4e00\u4e8c\u4e09\u56db", "paralinguistic_ratio": 1.1}, args
            ),
            "invalid_paralinguistic_ratio",
        )
        self.assertIsNone(
            _style_rejection(
                {"text": "\u4e00\u4e8c\u4e09\u56db", "paralinguistic_ratio": 0.0}, args
            )
        )

    def test_text_normalization_and_grouping(self) -> None:
        text = "【happy2】你[breath]好<stress>世界</stress>！"
        self.assertEqual(clean_spoken_text(text), "你好世界!")
        self.assertEqual(normalized_transcript(text), "你好世界")
        self.assertEqual(event_tags("<cry>我很难过</cry>[breath]"), {"cry", "breath"})
        self.assertTrue(is_speed_excluded({"text": "【sad】<cry>我很难过</cry>"}))
        self.assertFalse(is_speed_excluded({"text": "【neutral】<stress>正常强调</stress>"}))
        record = {
            "audio_path": "0827/1/目标/单句+标注/知识问答-目标音色-旅行-001/000001.wav",
            "key": "demo",
            "speaker_id": "must_not_override_bb03_path_group",
        }
        self.assertEqual(infer_source(record), "bb03")
        self.assertEqual(infer_recording_group(record), "bb03:recording:0827/1")
        male_instruct = {
            "wav_path": "audios/neutral/male/neutral/mmx_00001.wav",
            "item_name": "minimax_mmx_00001",
            "gender": "male",
        }
        female_instruct = {**male_instruct, "gender": "female"}
        self.assertEqual(infer_source(male_instruct), "instruct_tts")
        self.assertEqual(
            infer_recording_group(male_instruct),
            "instruct_tts:item:minimax_mmx:gender:male",
        )
        self.assertNotEqual(
            infer_recording_group(male_instruct), infer_recording_group(female_instruct)
        )
        self.assertEqual(
            infer_source({"source": "aopeng", "audio_path": "external/clip.wav", "key": "foreign-key"}),
            "aopeng",
        )
        self.assertEqual(
            infer_source({"dataset_source": "instruct_tts", "audio_path": "external/clip.wav"}),
            "instruct_tts",
        )
        self.assertEqual(
            infer_source({"audio_path": "0822/目标/单句/demo/000001.wav", "key": "bb03-key"}),
            "bb03",
        )
        self.assertEqual(
            infer_source({"audio_path": "external/clip.wav", "key": "foreign-key"}),
            "unknown",
        )
        self.assertIn(
            "raw-bb03-key",
            record_identity_variants({"control_candidate_id": "bb03:raw-bb03-key::speed"}),
        )
        self.assertEqual(add_control_tags("正文", "speed_fast"), "【speed_fast】正文")

    def test_stage_assembly_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidates = root / "candidates.jsonl"
            replay = root / "replay.jsonl"
            output = root / "stage.jsonl"
            report = root / "stage.json"
            tags = (
                "speed_slow", "speed_normal", "speed_fast",
                "effort_soft", "effort_normal", "effort_strong",
            )
            with candidates.open("w", encoding="utf-8") as handle:
                for index, tag in enumerate(tags):
                    handle.write(json.dumps({
                        "key": f"control-{index}",
                        "audio": f"/tmp/control-{index}.wav",
                        "text": f"【{tag}】【neutral】测试文本{index}",
                        "control_tag": tag,
                        "control_source": "instruct_tts",
                        "control_dataset_role": "control",
                    }, ensure_ascii=False) + "\n")
            with replay.open("w", encoding="utf-8") as handle:
                for index in range(6):
                    handle.write(json.dumps({
                        "key": f"replay-{index}",
                        "audio": f"/tmp/replay-{index}.wav",
                        "text": f"【neutral】回放文本{index}",
                        "source": "instruct_tts",
                    }, ensure_ascii=False) + "\n")

            subprocess.run([
                sys.executable, str(SCRIPT_DIR / "assemble_stage_dataset.py"),
                "--candidates-jsonl", str(candidates),
                "--replay-jsonl", str(replay),
                "--stage", "generic",
                "--output-jsonl", str(output),
                "--report-json", str(report),
                "--per-control", "1",
                "--replay-ratio", "1",
            ], check=True, capture_output=True, text=True)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 12)
            self.assertEqual(sum(row["control_dataset_role"] == "control" for row in rows), 6)
            validation_report = root / "validation.json"
            subprocess.run([
                sys.executable, str(SCRIPT_DIR / "validate_control_dataset.py"),
                "--input-jsonl", str(output),
                "--report-json", str(validation_report),
            ], check=True, capture_output=True, text=True)
            self.assertTrue(json.loads(validation_report.read_text(encoding="utf-8"))["valid"])

    def test_eval_manifest_uses_only_held_out_bb03(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "bb03.jsonl"
            rows = [
                {
                    "audio_path": f"0822/目标/单句/闲聊-目标音色-{index}/000001.wav",
                    "key": f"bb03-{index}",
                    "text": "【neutral】这是用于控制评估的一段完整测试文本。",
                }
                for index in range(3)
            ]
            source.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
            output = root / "eval.jsonl"
            report = root / "eval-report.json"
            training = root / "stage2.jsonl"
            training.write_text(
                json.dumps({"record_id": "bb03:bb03-0", "audio": "/tmp/used.wav"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            subprocess.run([
                sys.executable, str(SCRIPT_DIR / "build_control_eval_manifest.py"),
                "--input-jsonl", str(source),
                "--output-jsonl", str(output),
                "--report-json", str(report),
                "--ref-audio", "/tmp/ref.wav",
                "--exclude-training-jsonl", str(training),
                "--count", "2",
            ], check=True, capture_output=True, text=True)
            cases = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(cases), 14)
            self.assertNotIn("bb03-0", {case["source_key"] for case in cases})
            self.assertEqual({case["variant"] for case in cases}, {
                "untagged", "speed_slow", "speed_normal", "speed_fast",
                "effort_soft", "effort_normal", "effort_strong",
            })

    def test_stage_assembly_excludes_fixed_reference_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidates = root / "candidates.jsonl"
            output = root / "stage.jsonl"
            report = root / "stage-report.json"
            tags = (
                "speed_slow", "speed_normal", "speed_fast",
                "effort_soft", "effort_normal", "effort_strong",
            )
            with candidates.open("w", encoding="utf-8") as handle:
                for index, tag in enumerate(tags):
                    for key in ("fixed-reference", f"usable-{index}"):
                        handle.write(json.dumps({
                            "key": key,
                            "audio": f"/tmp/{key}-{tag}.wav",
                            "text": f"\u3010{tag}\u3011\u3010neutral\u3011\u6d4b\u8bd5\u6587\u672c{index}",
                            "control_tag": tag,
                            "source": "bb03",
                        }, ensure_ascii=False) + "\n")
            subprocess.run([
                sys.executable, str(SCRIPT_DIR / "assemble_stage_dataset.py"),
                "--candidates-jsonl", str(candidates),
                "--stage", "bb03",
                "--output-jsonl", str(output),
                "--report-json", str(report),
                "--exclude-key", "fixed-reference",
                "--per-control", "1",
                "--replay-ratio", "0",
            ], check=True, capture_output=True, text=True)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 6)
            self.assertNotIn("fixed-reference", {row["key"] for row in rows})
            self.assertEqual(
                json.loads(report.read_text(encoding="utf-8"))["rejections"]["explicitly_excluded_key"],
                6,
            )

    def test_speed_filters_and_effort_independence(self) -> None:
        """Keep unreliable rates out while allowing speed samples without effort groups."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "alignment.jsonl"
            candidates = root / "candidates.jsonl"
            report = root / "candidates-report.json"

            def make_record(
                index: int,
                cps: float,
                coverage: float,
                effort_index: int | None = None,
            ) -> dict[str, object]:
                span = 6.0 / cps
                timestamps = [
                    {
                        "start_sec": 0.1 + (span * char_index / 6),
                        "end_sec": 0.1 + (span * (char_index + 1) / 6),
                    }
                    for char_index in range(6)
                ]
                if coverage < 1.0:
                    timestamps.pop()
                level = -48.0 + (2.0 * (index if effort_index is None else effort_index))
                return {
                    "source": "instruct_tts",
                    "record_id": f"record-{index}",
                    "audio": f"/synthetic/item-{index}.wav",
                    "recording_group": "instruct_tts:group:coverage-test",
                    "text": "\u3010neutral\u3011\u4e00\u4e8c\u4e09\u56db\u4e94\u516d",
                    "clean_text": "\u4e00\u4e8c\u4e09\u56db\u4e94\u516d",
                    "status": "ok",
                    "alignment_status": "ok",
                    "asr_cer": 0.01,
                    "alignment_coverage": coverage,
                    "character_timestamps": timestamps,
                    "speech_start_sec": 0.1,
                    "speech_end_sec": 0.1 + span,
                    "speech_span_sec": span,
                    "pause_excluded_duration_sec": span,
                    "pause_excluded_cps": cps,
                    "char_pause_count": 0,
                    "char_pause_duration_sec": 0.0,
                    "char_pause_ratio": 0.0,
                    "paralinguistic_ratio": 0.0,
                    "speech_vad_active_duration_sec": span * 0.8,
                    "pause_ratio": 0.2,
                    "active_rms_dbfs_p25": level,
                    "active_rms_dbfs_p50": level + 1.0,
                    "active_lufs_p25": level - 2.0,
                    "active_lufs_p50": level - 1.0,
                    "clipping_ratio": 0.0,
                    "dynamic_range_db": 10.0,
                    "noise_floor_dbfs": -60.0,
                    "snr_db": 30.0,
                }

            records = [make_record(index, cps, 1.0) for index, cps in enumerate((2, 3, 4, 5, 6, 7))]
            # This high-rate row would alter the fast boundary if its one missing
            # lexical timestamp were allowed into the speed calibration set.
            records.append(make_record(99, 9, 0.95, effort_index=3))
            english = make_record(100, 5, 1.0, effort_index=3)
            english["text"] = "A compact English test phrase"
            english["clean_text"] = "A compact English test phrase"
            records.append(english)
            speed_only = make_record(101, 5, 1.0, effort_index=3)
            speed_only["recording_group"] = "instruct_tts:group:too-small-for-effort"
            records.append(speed_only)
            manifest.write_text(
                "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
                encoding="utf-8",
            )

            subprocess.run([
                sys.executable, str(SCRIPT_DIR / "build_control_candidates.py"),
                "--input-jsonl", str(manifest),
                "--output-jsonl", str(candidates),
                "--report-json", str(report),
                "--min-speed-calibration-records", "6",
                "--min-group-records", "5",
                "--min-effort-rms-span-db", "1",
                "--min-effort-lufs-span-db", "1",
            ], check=True, capture_output=True, text=True)

            stats = json.loads(report.read_text(encoding="utf-8"))
            rows = [json.loads(line) for line in candidates.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(stats["rejections"]["speed_alignment_coverage_below_threshold"], 1)
            self.assertEqual(stats["rejections"]["non_cjk_speed_control"], 1)
            self.assertEqual(stats["speed_calibration"]["records"], 7)
            self.assertEqual(stats["speed_calibration"]["min_alignment_coverage"], 1.0)
            self.assertFalse(stats["speed_calibration"]["include_non_cjk_speed"])
            self.assertFalse(stats["speed_candidate_policy"]["require_normal_effort"])
            self.assertEqual(stats["source_effort_metric_counts"], {"instruct_tts": 9})
            self.assertEqual(stats["selection_rejections"]["effort_requires_complete_speed_alignment"], 1)
            self.assertEqual(stats["selection_rejections"]["effort_requires_cjk_lexical_content"], 1)
            self.assertFalse(any(
                row["control_selection"]["record_id"].endswith(("record-99", "record-100"))
                for row in rows
            ))
            speed_only_row = next(
                row
                for row in rows
                if row["control_candidate_id"].endswith("record-101::speed")
            )
            self.assertIsNone(speed_only_row["control_selection"]["effort_group_threshold"])
            self.assertEqual(
                speed_only_row["control_selection"]["confound_filter"],
                "effort_not_conditioned_for_speed",
            )

            strict_candidates = root / "strict-candidates.jsonl"
            strict_report = root / "strict-candidates-report.json"
            subprocess.run([
                sys.executable, str(SCRIPT_DIR / "build_control_candidates.py"),
                "--input-jsonl", str(manifest),
                "--output-jsonl", str(strict_candidates),
                "--report-json", str(strict_report),
                "--min-speed-calibration-records", "6",
                "--min-group-records", "5",
                "--min-effort-rms-span-db", "1",
                "--min-effort-lufs-span-db", "1",
                "--require-normal-effort-for-speed",
            ], check=True, capture_output=True, text=True)
            strict_rows = [
                json.loads(line)
                for line in strict_candidates.read_text(encoding="utf-8").splitlines()
            ]
            strict_stats = json.loads(strict_report.read_text(encoding="utf-8"))
            self.assertTrue(strict_stats["speed_candidate_policy"]["require_normal_effort"])
            self.assertGreaterEqual(
                strict_stats["selection_rejections"]["speed_requires_normal_effort"], 1
            )
            self.assertFalse(any(
                row["control_candidate_id"].endswith("record-101::speed")
                for row in strict_rows
            ))

    def test_candidates_approval_group_references_and_validation(self) -> None:
        """Exercise the no-audio control-data path, including normal-tag overlap."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "alignment.jsonl"
            candidates = root / "candidates.jsonl"
            candidate_report = root / "candidates-report.json"
            approved = root / "approved-effort.txt"
            stage = root / "stage.jsonl"
            stage_report = root / "stage-report.json"
            paired = root / "paired.jsonl"
            pairing_report = root / "pairing-report.json"
            validation_report = root / "validation-report.json"

            def make_record(
                source: str,
                group: str,
                index: int,
                cps: float,
                effort_index: int,
                cer: float = 0.01,
            ) -> dict[str, object]:
                span = 6.0 / cps
                level = -40.0 + (2.0 * effort_index)
                return {
                    "source": source,
                    "record_id": f"{source}-{group}-{index}",
                    "audio": f"/synthetic/{source}/{group}/item-{index}.wav",
                    "recording_group": f"{source}:group:{group}",
                    "text": "\u3010neutral\u3011\u4e00\u4e8c\u4e09\u56db\u4e94\u516d",
                    "clean_text": "\u4e00\u4e8c\u4e09\u56db\u4e94\u516d",
                    "status": "ok",
                    "alignment_status": "ok",
                    "asr_cer": cer,
                    "alignment_coverage": 1.0,
                    "character_timestamps": [
                        {"start_sec": 0.1 + (span * item / 6), "end_sec": 0.1 + (span * (item + 1) / 6)}
                        for item in range(6)
                    ],
                    "speech_start_sec": 0.1,
                    "speech_end_sec": 0.1 + span,
                    "speech_span_sec": span,
                    "pause_excluded_duration_sec": span,
                    "pause_excluded_cps": cps,
                    "char_pause_count": 0,
                    "char_pause_duration_sec": 0.0,
                    "char_pause_ratio": 0.0,
                    "paralinguistic_ratio": 0.0,
                    "speech_vad_active_duration_sec": span * 0.8,
                    "pause_ratio": 0.2,
                    "active_rms_dbfs_p25": level,
                    "active_rms_dbfs_p50": level + 1.0,
                    "active_lufs_p25": level - 2.0,
                    "active_lufs_p50": level - 1.0,
                    "clipping_ratio": 0.0,
                    "dynamic_range_db": 10.0,
                    "noise_floor_dbfs": -60.0,
                    "snr_db": 30.0,
                }

            manifest_rows: list[dict[str, object]] = []
            for group, cps_values in (
                ("generic-slow", (2.0, 2.0, 2.0, 3.0, 3.0)),
                ("generic-normal", (4.0, 3.0, 4.0, 4.0, 4.0)),
                ("generic-fast", (5.0, 5.0, 6.0, 6.0, 6.0)),
            ):
                for index, cps in enumerate(cps_values):
                    manifest_rows.append(make_record(
                        "instruct_tts",
                        group,
                        index,
                        cps,
                        index,
                    ))
            # This row is deliberately invalid for both candidates and the reference pool.
            manifest_rows.append(make_record("instruct_tts", "generic-slow", 99, 4.0, 2, cer=0.5))
            for index, cps in enumerate((2.0, 2.0, 4.0, 4.0, 6.0, 6.0)):
                manifest_rows.append(make_record("bb03", "calibration", index, cps, index % 5))
            manifest.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in manifest_rows) + "\n",
                encoding="utf-8",
            )

            subprocess.run([
                sys.executable, str(SCRIPT_DIR / "build_control_candidates.py"),
                "--input-jsonl", str(manifest),
                "--output-jsonl", str(candidates),
                "--report-json", str(candidate_report),
                "--min-speed-calibration-records", "6",
                "--min-group-records", "5",
                "--min-effort-rms-span-db", "1",
                "--min-effort-lufs-span-db", "1",
            ], check=True, capture_output=True, text=True)
            candidate_rows = [json.loads(line) for line in candidates.read_text(encoding="utf-8").splitlines()]
            candidate_stats = json.loads(candidate_report.read_text(encoding="utf-8"))
            self.assertEqual(
                candidate_stats["speed_calibration"]["scope"],
                "global_all_high_confidence_speed_items",
            )
            self.assertEqual(candidate_stats["speed_calibration"]["primary_metric"], "pause_excluded_cps")
            self.assertEqual(
                candidate_stats["speed_calibration"]["metric_sources"],
                {"pause_excluded_cps": 21},
            )
            self.assertTrue(all(row["control_candidate_version"] == "natural_speed_effort_candidates_v3" for row in candidate_rows))
            self.assertTrue(all(
                row["control_label_source"] == "natural_pause_excluded_speed_and_within_group_effort_v3"
                for row in candidate_rows
            ))
            self.assertTrue(all(row["speed_rate_metric"] == "pause_excluded_cps" for row in candidate_rows))
            self.assertEqual(
                candidate_stats["speed_calibration"]["source_counts"],
                {"bb03": 6, "instruct_tts": 15},
            )
            # The global lower boundary is 3.0 for all 21 valid rows. BB03
            # alone would yield a different boundary, so this guards against
            # accidentally reverting calibration to the target-voice subset.
            self.assertAlmostEqual(
                candidate_stats["speed_calibration"]["boundaries_cps"]["slow_max"],
                3.0,
            )
            approved.write_text(
                "\n".join(
                    row["control_candidate_id"]
                    for row in candidate_rows
                    if str(row["control_tag"]).startswith("effort_")
                ) + "\n",
                encoding="utf-8",
            )

            subprocess.run([
                sys.executable, str(SCRIPT_DIR / "assemble_stage_dataset.py"),
                "--candidates-jsonl", str(candidates),
                "--stage", "generic",
                "--output-jsonl", str(stage),
                "--report-json", str(stage_report),
                "--approved-keys-file", str(approved),
                "--require-approved-effort",
                "--per-control", "1",
                "--replay-ratio", "0",
            ], check=True, capture_output=True, text=True)
            stage_rows = [json.loads(line) for line in stage.read_text(encoding="utf-8").splitlines()]
            self.assertEqual({row["control_tag"] for row in stage_rows}, {
                "speed_slow", "speed_normal", "speed_fast",
                "effort_soft", "effort_normal", "effort_strong",
            })
            subprocess.run([
                sys.executable, str(SCRIPT_DIR / "attach_group_references.py"),
                "--input-jsonl", str(stage),
                "--reference-manifest-jsonl", str(manifest),
                "--output-jsonl", str(paired),
                "--report-json", str(pairing_report),
                "--reference-pool-size", "5",
            ], check=True, capture_output=True, text=True)
            paired_rows = [json.loads(line) for line in paired.read_text(encoding="utf-8").splitlines()]
            pairing = json.loads(pairing_report.read_text(encoding="utf-8"))
            self.assertEqual(pairing["paired_records"], 6)
            self.assertEqual(pairing["self_reference_records"], 0)
            self.assertEqual(pairing["reference_rejections"]["asr_cer_out_of_range"], 1)
            self.assertTrue(all(row["audio"] != row["ref_audio"] for row in paired_rows))

            subprocess.run([
                sys.executable, str(SCRIPT_DIR / "validate_control_dataset.py"),
                "--input-jsonl", str(paired),
                "--report-json", str(validation_report),
                "--require-ref-audio",
            ], check=True, capture_output=True, text=True)
            self.assertTrue(json.loads(validation_report.read_text(encoding="utf-8"))["valid"])

            self_reference = root / "self-reference.jsonl"
            self_reference_rows = [dict(row) for row in paired_rows]
            self_reference_rows[0]["ref_audio"] = self_reference_rows[0]["audio"]
            self_reference.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in self_reference_rows) + "\n",
                encoding="utf-8",
            )
            self_reference_report = root / "self-reference-report.json"
            self_reference_result = subprocess.run([
                sys.executable, str(SCRIPT_DIR / "validate_control_dataset.py"),
                "--input-jsonl", str(self_reference),
                "--report-json", str(self_reference_report),
                "--require-ref-audio",
            ], capture_output=True, text=True)
            self.assertNotEqual(self_reference_result.returncode, 0)
            self_reference_failures = json.loads(
                self_reference_report.read_text(encoding="utf-8")
            )["failures"]
            self.assertIn("self_reference_audio", {failure["reason"] for failure in self_reference_failures})

            existing_target = root / "existing-target.wav"
            existing_target.touch()
            missing_ref = root / "missing-reference.wav"
            missing_ref_input = root / "missing-reference.jsonl"
            missing_ref_input.write_text(
                json.dumps({"audio": str(existing_target), "ref_audio": str(missing_ref)}) + "\n",
                encoding="utf-8",
            )
            missing_ref_report = root / "missing-reference-report.json"
            missing_ref_result = subprocess.run([
                sys.executable, str(SCRIPT_DIR / "validate_control_dataset.py"),
                "--input-jsonl", str(missing_ref_input),
                "--report-json", str(missing_ref_report),
                "--require-ref-audio", "--require-audio-exists",
            ], capture_output=True, text=True)
            self.assertNotEqual(missing_ref_result.returncode, 0)
            missing_ref_failures = json.loads(missing_ref_report.read_text(encoding="utf-8"))["failures"]
            self.assertIn("ref_audio_missing_on_disk", {failure["reason"] for failure in missing_ref_failures})

            duplicate = root / "duplicate-tag.jsonl"
            duplicate_row = dict(paired_rows[0])
            duplicate_row["audio"] = duplicate_row["audio"].replace("/item-", "/./item-")
            duplicate.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in [*paired_rows, duplicate_row]) + "\n",
                encoding="utf-8",
            )
            duplicate_report = root / "duplicate-tag-report.json"
            duplicate_result = subprocess.run([
                sys.executable, str(SCRIPT_DIR / "validate_control_dataset.py"),
                "--input-jsonl", str(duplicate),
                "--report-json", str(duplicate_report),
            ], capture_output=True, text=True)
            self.assertNotEqual(duplicate_result.returncode, 0)
            failures = json.loads(duplicate_report.read_text(encoding="utf-8"))["failures"]
            self.assertIn("duplicate_control_audio_and_tag", {failure["reason"] for failure in failures})


if __name__ == "__main__":
    unittest.main()
