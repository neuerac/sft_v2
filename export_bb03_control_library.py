"""Export BB03 control candidates into a standalone categorized audio library.

This is deliberately a BB03-only materialization step.  It accepts the
already screened output of :mod:`build_control_candidates`, copies (or links)
the original natural recordings into six tag folders, and writes an auditable
manifest.  It does not run ASR, derive a new label, alter source audio, or
prepare training data.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

if __package__:
    from .common import EFFORT_TAGS, SPEED_TAGS, iter_jsonl, resolve_audio
    from .export_control_library import (
        candidate_id,
        copy_audio,
        destination_name,
        export_row,
        source_name,
        write_csv,
        write_jsonl,
    )
else:  # Direct execution must prefer this directory over site-packages names.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import EFFORT_TAGS, SPEED_TAGS, iter_jsonl, resolve_audio  # type: ignore[no-redef]
    from export_control_library import (  # type: ignore[no-redef]
        candidate_id,
        copy_audio,
        destination_name,
        export_row,
        source_name,
        write_csv,
        write_jsonl,
    )


ALL_TAGS = frozenset((*SPEED_TAGS, *EFFORT_TAGS))
BB03_SOURCE = "bb03"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy or link screened BB03 speed/effort candidates into a standalone "
            "six-folder review library. Input must be a control_candidates JSONL, "
            "not an alignment manifest."
        )
    )
    parser.add_argument(
        "--input-jsonl",
        required=True,
        help="BB03 rows from build_control_candidates.py (mixed-source files are allowed; non-BB03 rows are skipped).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New or empty output directory for the BB03 review library.",
    )
    parser.add_argument(
        "--audio-root",
        default=None,
        help="Prefix for relative audio paths in the candidate JSONL, if required.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help="How to materialize BB03 audio. hardlink falls back to copy across filesystems.",
    )
    return parser.parse_args()


def _error(line_number: int, reason: str, record: dict[str, Any], audio: str = "") -> dict[str, str]:
    return {
        "line": str(line_number),
        "reason": reason,
        "candidate_id": candidate_id(record, line_number),
        "audio": audio,
    }


def _candidate_export_record(
    record: dict[str, Any],
    exported_audio: str,
    source_audio: str,
    materialization: str,
    line_number: int,
) -> dict[str, Any]:
    """Keep candidate provenance while making the review-copy location explicit."""
    result = dict(record)
    result["source"] = BB03_SOURCE
    result["source_audio"] = source_audio
    result["library_audio"] = exported_audio
    result["library_materialization"] = materialization
    result["library_manifest_line"] = line_number
    return result


def _manifest_fields(rows: list[dict[str, Any]]) -> list[str]:
    if rows:
        return list(rows[0])
    return [
        "candidate_id",
        "control_tag",
        "control_kind",
        "source",
        "recording_group",
        "text",
        "source_audio",
        "exported_audio",
        "materialization",
        "speed_rate_cps",
        "speed_rate_metric",
        "asr_cer",
        "alignment_coverage",
        "pause_excluded_cps",
        "char_pause_ratio",
        "active_rms_dbfs_p50",
        "active_lufs_p50",
        "emotion_family",
        "review_status",
        "review_note",
    ]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_root = output_dir / "audio"
    manifest_root = output_dir / "manifests"
    manifest_root.mkdir()

    errors: list[dict[str, str]] = []
    exported_rows: list[dict[str, Any]] = []
    exported_candidates: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    skipped_non_bb03 = 0
    input_records = 0
    candidate_like_records = 0
    seen_ids: set[tuple[str, str]] = set()

    for line_number, record in iter_jsonl(args.input_jsonl):
        input_records += 1
        source = source_name(record)
        if source != BB03_SOURCE:
            skipped_non_bb03 += 1
            continue

        tag = str(record.get("control_tag") or "").strip()
        if tag not in ALL_TAGS:
            errors.append(_error(line_number, "missing_or_unsupported_control_tag", record))
            continue
        candidate_like_records += 1

        identifier = candidate_id(record, line_number)
        dedupe_key = (tag, identifier)
        if dedupe_key in seen_ids:
            errors.append(_error(line_number, "duplicate_candidate_id_and_tag", record))
            continue
        seen_ids.add(dedupe_key)

        source_audio = resolve_audio(record, args.audio_root)
        if not source_audio or not Path(source_audio).is_file():
            errors.append(_error(line_number, "missing_audio", record, str(source_audio or "")))
            continue

        destination_dir = audio_root / tag
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / destination_name(
            record, BB03_SOURCE, identifier, source_audio
        )
        materialization = copy_audio(Path(source_audio), destination, args.copy_mode)
        exported_audio = destination.relative_to(output_dir).as_posix()
        exported_rows.append(
            export_row(record, exported_audio, source_audio, materialization, line_number)
        )
        exported_candidates.append(
            _candidate_export_record(
                record, exported_audio, source_audio, materialization, line_number
            )
        )
        counts[tag] += 1

    if input_records and candidate_like_records == 0:
        raise ValueError(
            "no BB03 control candidates found: input looks like an alignment manifest. "
            "Run build_control_candidates.py first, then pass its candidate JSONL here."
        )

    write_jsonl(manifest_root / "bb03_control_candidates.jsonl", exported_candidates)
    write_csv(
        manifest_root / "bb03_library_manifest.csv",
        exported_rows,
        _manifest_fields(exported_rows),
    )
    (manifest_root / "errors.json").write_text(
        json.dumps(errors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    report = {
        "input_jsonl": args.input_jsonl,
        "output_dir": str(output_dir),
        "input_records": input_records,
        "skipped_non_bb03_records": skipped_non_bb03,
        "bb03_records_with_control_tag": candidate_like_records,
        "exported_records": len(exported_rows),
        "errors": len(errors),
        "copy_mode_requested": args.copy_mode,
        "materialization": dict(sorted(Counter(row["materialization"] for row in exported_rows).items())),
        "by_control_tag": {tag: counts.get(tag, 0) for tag in (*SPEED_TAGS, *EFFORT_TAGS)},
        "library_layout": {
            tag: f"audio/{tag}/" for tag in (*SPEED_TAGS, *EFFORT_TAGS)
        },
        "next_step": (
            "Review manifests/bb03_library_manifest.csv and use "
            "manifests/bb03_control_candidates.jsonl with export_effort_audit.py "
            "for blind effort review."
        ),
    }
    report_path = manifest_root / "bb03_export_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
