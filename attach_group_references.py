"""Attach deterministic, non-self references from the same recording group."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__:
    from .common import (
        emotion_family,
        event_tags,
        finite_float,
        infer_recording_group,
        iter_jsonl,
        normalized_transcript,
        resolve_audio,
    )
else:  # Direct execution must prefer this directory over a site-packages ``common``.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import (  # type: ignore[no-redef]
        emotion_family,
        event_tags,
        finite_float,
        infer_recording_group,
        iter_jsonl,
        normalized_transcript,
        resolve_audio,
    )


SUCCESS_STATUSES = frozenset({"", "ok", "success", "aligned", "complete", "completed"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pair generic control targets with clean same-recording-group references.")
    parser.add_argument("--input-jsonl", required=True, help="Stage dataset produced by assemble_stage_dataset.py.")
    parser.add_argument("--reference-manifest-jsonl", required=True, help="Full alignment manifest used to create candidates.")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--audio-root", default=None)
    parser.add_argument("--reference-pool-size", type=int, default=3)
    parser.add_argument("--max-asr-cer", type=float, default=0.12)
    parser.add_argument("--min-alignment-coverage", type=float, default=0.90)
    parser.add_argument("--require-audio-exists", action="store_true")
    return parser.parse_args()


def normalized_path(path: str) -> str:
    return str(path).replace("\\", "/").rstrip("/").lower()


def clean_reference_rejection(record: dict[str, Any], max_asr_cer: float, min_alignment_coverage: float) -> str | None:
    if str(record.get("status") or "").strip().lower() not in SUCCESS_STATUSES:
        return "status_not_successful"
    alignment_status = str(record.get("alignment_status") or "").strip().lower()
    if alignment_status not in SUCCESS_STATUSES:
        return "alignment_not_successful"
    text = str(record.get("text") or "")
    if not text or event_tags(text) - {"breath", "hold"}:
        return "not_clean_text"
    cer = finite_float(record.get("asr_cer"))
    if cer is None or cer < 0 or cer > max_asr_cer:
        return "asr_cer_out_of_range"
    coverage = finite_float(record.get("alignment_coverage"))
    if coverage is None or coverage < min_alignment_coverage or coverage > 1.001:
        return "alignment_coverage_out_of_range"
    return None


def reference_score(record: dict[str, Any], audio: str) -> tuple[Any, ...]:
    family = emotion_family(record.get("text"))
    family_rank = 0 if family == "neutral" else 1 if family == "no_emotion" else 2
    cps = record.get("speech_cps")
    try:
        speed_penalty = abs(float(cps) - 4.0)
    except (TypeError, ValueError):
        speed_penalty = 99.0
    return (family_rank, speed_penalty, len(normalized_transcript(record.get("text"))), normalized_path(audio))


def stable_index(value: str, size: int) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big") % size


def target_id(record: dict[str, Any], line_number: int) -> str:
    for field in ("control_candidate_id", "key", "record_id", "audio"):
        value = str(record.get(field) or "").strip()
        if value:
            return value
    return f"line:{line_number}"


def main() -> None:
    args = parse_args()
    if args.reference_pool_size < 1:
        raise ValueError("--reference-pool-size must be >= 1")
    if not 0 <= args.max_asr_cer <= 1:
        raise ValueError("--max-asr-cer must be within [0, 1]")
    if not 0 < args.min_alignment_coverage <= 1:
        raise ValueError("--min-alignment-coverage must be within (0, 1]")
    pools: dict[str, list[dict[str, str]]] = defaultdict(list)
    reference_rejections = Counter()
    for _, record in iter_jsonl(args.reference_manifest_jsonl):
        rejection = clean_reference_rejection(
            record, args.max_asr_cer, args.min_alignment_coverage
        )
        if rejection:
            reference_rejections[rejection] += 1
            continue
        audio = resolve_audio(record, args.audio_root)
        if not audio:
            reference_rejections["missing_audio"] += 1
            continue
        if args.require_audio_exists and not os.path.isfile(audio):
            reference_rejections["audio_missing_on_disk"] += 1
            continue
        group = str(record.get("recording_group") or infer_recording_group(record))
        pools[group].append({"audio": audio, "text": str(record.get("text") or ""), "score": reference_score(record, audio)})

    selected_pools: dict[str, list[dict[str, str]]] = {}
    for group, rows in pools.items():
        rows.sort(key=lambda item: item["score"])
        selected_pools[group] = rows[: args.reference_pool_size]

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    paired = 0
    self_reference = 0
    missing_pool = 0
    policy_counts = Counter()
    records = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for line_number, record in iter_jsonl(args.input_jsonl):
            result = dict(record)
            audio = resolve_audio(result, args.audio_root)
            if not audio:
                raise ValueError(f"line {line_number}: target has no audio path")
            result["audio"] = audio
            group = str(result.get("recording_group") or infer_recording_group(result))
            candidates = [item for item in selected_pools.get(group, []) if normalized_path(item["audio"]) != normalized_path(audio)]
            if not candidates:
                missing_pool += 1
                self_reference += 1
                result["ref_audio"] = audio
                result["ref_policy"] = "self_reference_no_same_group_pool"
                policy_counts[result["ref_policy"]] += 1
            else:
                reference = candidates[stable_index(target_id(result, line_number), len(candidates))]
                result["ref_audio"] = reference["audio"]
                result["ref_policy"] = "same_recording_group_clean_reference"
                paired += 1
                policy_counts[result["ref_policy"]] += 1
            records += 1
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    report = {
        "records": records,
        "recording_groups_with_reference_pools": len(selected_pools),
        "reference_pool_size": args.reference_pool_size,
        "paired_records": paired,
        "self_reference_records": self_reference,
        "missing_reference_pool_records": missing_pool,
        "reference_rejections": dict(reference_rejections),
        "policies": dict(policy_counts),
    }
    destination = Path(args.report_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if self_reference:
        raise SystemExit("some records could not receive a non-self same-group reference")


if __name__ == "__main__":
    main()
