"""Validate a control dataset before reference pairing, codec encoding, or SFT."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

if __package__:
    from .common import ALL_CONTROL_TAGS, EFFORT_TAGS, SPEED_TAGS, emotion_family, infer_source, iter_jsonl
else:  # Direct execution must prefer this directory over a site-packages ``common``.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import (  # type: ignore[no-redef]
        ALL_CONTROL_TAGS,
        EFFORT_TAGS,
        SPEED_TAGS,
        emotion_family,
        infer_source,
        iter_jsonl,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate speed/effort tag structure and optional training prerequisites.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--require-ref-audio", action="store_true")
    parser.add_argument("--require-audio-codes", action="store_true")
    parser.add_argument("--require-audio-exists", action="store_true")
    return parser.parse_args()


def valid_audio_codes(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(frame, list) and len(frame) == 16 for frame in value)


def present_tags(text: str) -> set[str]:
    return {tag for tag in ALL_CONTROL_TAGS if f"\u3010{tag}\u3011" in text}


def normalized_audio_key(audio: str) -> str:
    """Normalize separator and dot-segments before control-row de-duplication."""
    return os.path.normpath(audio.replace("\\", "/")).replace("\\", "/")


def same_audio(left: str, right: str) -> bool:
    """Conservatively detect self-reference without requiring the files to exist."""
    return normalized_audio_key(left).lower() == normalized_audio_key(right).lower()


def main() -> None:
    args = parse_args()
    failures: list[dict[str, Any]] = []
    tag_counts = Counter()
    source_counts = Counter()
    emotion_counts = Counter()
    role_counts = Counter()
    audio_seen: dict[tuple[str, str], int] = {}
    total = 0
    for line_number, record in iter_jsonl(args.input_jsonl):
        total += 1
        text = str(record.get("text") or "")
        tags = present_tags(text)
        speed = tags & set(SPEED_TAGS)
        effort = tags & set(EFFORT_TAGS)
        for tag in tags:
            tag_counts[tag] += 1
        source_counts[str(record.get("control_source") or record.get("source") or infer_source(record))] += 1
        emotion_counts[emotion_family(text)] += 1
        role_counts[str(record.get("control_dataset_role") or "unknown")] += 1
        if len(speed) > 1 or len(effort) > 1:
            failures.append({"line": line_number, "reason": "multiple_tags_in_dimension", "tags": sorted(tags)})
        if record.get("control_dataset_role") == "control" and len(tags) != 1:
            failures.append({"line": line_number, "reason": "control_row_requires_one_tag", "tags": sorted(tags)})
        declared_tag = str(record.get("control_tag") or "").strip()
        if record.get("control_dataset_role") == "control" and declared_tag and tags != {declared_tag}:
            failures.append({
                "line": line_number,
                "reason": "declared_control_tag_does_not_match_text",
                "declared_tag": declared_tag,
                "tags": sorted(tags),
            })
        if "【loud】" in text or "【speed_very_slow】" in text or "【speed_very_fast】" in text:
            failures.append({"line": line_number, "reason": "legacy_v3_control_tag"})
        audio = str(record.get("audio") or "").strip()
        if not audio:
            failures.append({"line": line_number, "reason": "missing_audio"})
        else:
            tag_key = declared_tag or (next(iter(tags)) if len(tags) == 1 else "")
            audio_key = (normalized_audio_key(audio), tag_key)
            if record.get("control_dataset_role") == "control" and audio_key in audio_seen:
                failures.append({
                    "line": line_number,
                    "reason": "duplicate_control_audio_and_tag",
                    "control_tag": tag_key,
                    "first_line": audio_seen[audio_key],
                })
            else:
                audio_seen.setdefault(audio_key, line_number)
        if args.require_audio_exists and audio and not os.path.isfile(audio):
            failures.append({"line": line_number, "reason": "audio_missing_on_disk", "audio": audio})
        ref_audio = str(record.get("ref_audio") or "").strip()
        if args.require_ref_audio:
            if not ref_audio:
                failures.append({"line": line_number, "reason": "missing_ref_audio"})
            else:
                if audio and same_audio(audio, ref_audio):
                    failures.append({"line": line_number, "reason": "self_reference_audio"})
                if args.require_audio_exists and not os.path.isfile(ref_audio):
                    failures.append(
                        {"line": line_number, "reason": "ref_audio_missing_on_disk", "ref_audio": ref_audio}
                    )
        if args.require_audio_codes and not valid_audio_codes(record.get("audio_codes")):
            failures.append({"line": line_number, "reason": "invalid_audio_codes"})

    report = {
        "input_jsonl": args.input_jsonl,
        "records": total,
        "valid": not failures,
        "tag_counts": dict(tag_counts),
        "source_counts": dict(source_counts),
        "emotion_counts": dict(emotion_counts),
        "role_counts": dict(role_counts),
        "failure_count": len(failures),
        "failures": failures[:200],
        "failures_truncated": len(failures) > 200,
    }
    destination = Path(args.report_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit("control dataset validation failed")


if __name__ == "__main__":
    main()
