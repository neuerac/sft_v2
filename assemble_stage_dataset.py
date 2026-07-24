"""Assemble balanced generic or BB03 control-training inputs.

This script deliberately does not encode audio or select reference audio.  Run
the repository's ``pair_reference_audio.py`` and ``prepare_data.py`` after this
step so those artifacts stay consistent with the target audio.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__:
    from .common import (
        EFFORT_TAGS,
        SPEED_TAGS,
        emotion_family,
        infer_source,
        iter_jsonl,
        record_identity_variants,
    )
else:  # Direct execution must prefer this directory over a site-packages ``common``.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import (  # type: ignore[no-redef]
        EFFORT_TAGS,
        SPEED_TAGS,
        emotion_family,
        infer_source,
        iter_jsonl,
        record_identity_variants,
    )


LEGACY_CONTROL_MARKERS = (
    "\u3010loud\u3011",
    "\u3010speed_very_slow\u3011",
    "\u3010speed_very_fast\u3011",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one control-SFT stage from reviewed candidates and replay.")
    parser.add_argument("--candidates-jsonl", action="append", required=True, help="May be specified multiple times.")
    parser.add_argument("--replay-jsonl", action="append", default=[], help="Optional original-data replay JSONL(s).")
    parser.add_argument("--stage", choices=("generic", "bb03"), required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--approved-keys-file", default=None, help="Optional human-approved effort candidate keys.")
    parser.add_argument(
        "--exclude-key",
        action="append",
        default=[],
        help="Raw record key to omit from controls and replay; may be repeated.",
    )
    parser.add_argument(
        "--exclude-keys-file",
        default=None,
        help="Optional file of raw record keys to omit from controls and replay.",
    )
    parser.add_argument("--require-approved-effort", action="store_true", help="Drop effort rows absent from --approved-keys-file.")
    parser.add_argument(
        "--per-control",
        type=int,
        default=0,
        help="0 uses the smallest eligible tag count; otherwise cap every tag at this count.",
    )
    parser.add_argument("--replay-ratio", type=float, default=1.0, help="Replay rows per selected control row.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_approved(path: str | None) -> set[str]:
    if not path:
        return set()
    with open(path, "r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")}


def load_excluded(keys: list[str], path: str | None) -> set[str]:
    excluded = {value.strip() for value in keys if value.strip()}
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            excluded.update(
                line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")
            )
    return excluded


def is_explicitly_excluded(record: dict[str, Any], excluded: set[str]) -> bool:
    return bool(record_identity_variants(record) & excluded)


def control_tag(record: dict[str, Any]) -> str | None:
    value = record.get("control_tag")
    if isinstance(value, str) and value in {*SPEED_TAGS, *EFFORT_TAGS}:
        return value
    if record.get("speed_label") in {"slow", "normal", "fast"}:
        return f"speed_{record['speed_label']}"
    if record.get("effort_label") in {"soft", "normal", "strong"}:
        return f"effort_{record['effort_label']}"
    return None


def is_stage_source(record: dict[str, Any], stage: str) -> bool:
    source = str(record.get("control_source") or record.get("source") or infer_source(record)).lower()
    return source == "bb03" if stage == "bb03" else source != "bb03"


def stable_key(record: dict[str, Any], line_number: int, origin: str) -> str:
    for field in ("key", "audio", "audio_path", "wav_path"):
        value = str(record.get(field) or "").strip()
        if value:
            return value.replace("\\", "/").lower()
    return f"{origin}:{line_number}"


def approval_id(record: dict[str, Any]) -> str:
    """Use the generated candidate ID so sources without a BB03-style key work too."""
    for field in ("control_candidate_id", "key", "record_id", "item_name"):
        value = str(record.get(field) or "").strip()
        if value:
            return value
    return ""


def source_name(record: dict[str, Any]) -> str:
    return str(record.get("control_source") or record.get("source") or infer_source(record)).lower()


def source_emotion_stratum(record: dict[str, Any]) -> tuple[str, str]:
    return source_name(record), emotion_family(record.get("text"))


def stratified_sample(
    rows: list[dict[str, Any]],
    count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Sample across source/emotion strata as evenly as availability permits."""
    if count >= len(rows):
        return list(rows)

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[source_emotion_stratum(row)].append(row)
    for values in buckets.values():
        rng.shuffle(values)

    # The score makes source coverage primary, then emotion coverage. A seeded
    # tie breaker prevents fixed path-order bias while retaining reproducibility.
    tie_breaker = {stratum: rng.random() for stratum in buckets}
    source_selected: Counter[str] = Counter()
    emotion_selected: Counter[str] = Counter()
    stratum_selected: Counter[tuple[str, str]] = Counter()
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        available = [stratum for stratum, values in buckets.items() if values]
        if not available:
            break
        stratum = min(
            available,
            key=lambda key: (
                source_selected[key[0]],
                emotion_selected[key[1]],
                stratum_selected[key],
                tie_breaker[key],
            ),
        )
        selected.append(buckets[stratum].pop())
        source_selected[stratum[0]] += 1
        emotion_selected[stratum[1]] += 1
        stratum_selected[stratum] += 1
    if len(selected) != count:
        raise RuntimeError(f"requested {count} rows but selected only {len(selected)}")
    return selected


def select_balanced(groups: dict[str, list[dict[str, Any]]], per_control: int, rng: random.Random) -> list[dict[str, Any]]:
    if not groups:
        return []
    cap = per_control or min(len(rows) for rows in groups.values())
    selected: list[dict[str, Any]] = []
    for tag, rows in sorted(groups.items()):
        if len(rows) < cap:
            raise ValueError(f"control tag {tag!r} has {len(rows)} eligible rows, below requested cap {cap}")
        chosen = stratified_sample(rows, cap, rng)
        selected.extend(chosen)
    rng.shuffle(selected)
    return selected


def main() -> None:
    args = parse_args()
    if args.per_control < 0:
        raise ValueError("--per-control must be >= 0")
    if args.replay_ratio < 0:
        raise ValueError("--replay-ratio must be >= 0")
    if args.require_approved_effort and not args.approved_keys_file:
        raise ValueError("--require-approved-effort requires --approved-keys-file")

    approved = load_approved(args.approved_keys_file)
    excluded = load_excluded(args.exclude_key, args.exclude_keys_file)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejection = Counter()
    seen_controls: set[tuple[str, str]] = set()
    for input_path in args.candidates_jsonl:
        for line_number, record in iter_jsonl(input_path):
            tag = control_tag(record)
            if tag is None:
                rejection["missing_control_tag"] += 1
                continue
            if not is_stage_source(record, args.stage):
                rejection["outside_stage_source"] += 1
                continue
            if is_explicitly_excluded(record, excluded):
                rejection["explicitly_excluded_key"] += 1
                continue
            candidate_approval_id = approval_id(record)
            if tag.startswith("effort_") and args.require_approved_effort and candidate_approval_id not in approved:
                rejection["effort_not_approved"] += 1
                continue
            dedupe_key = (tag, stable_key(record, line_number, input_path))
            if dedupe_key in seen_controls:
                rejection["duplicate_control"] += 1
                continue
            seen_controls.add(dedupe_key)
            result = dict(record)
            result["control_stage"] = args.stage
            result["control_tag"] = tag
            result["control_dataset_role"] = "control"
            groups[tag].append(result)

    required_tags = set(SPEED_TAGS) | set(EFFORT_TAGS)
    missing = sorted(required_tags - set(groups))
    if missing:
        raise ValueError(f"missing eligible control tags for stage {args.stage}: {missing}")
    rng = random.Random(args.seed)
    controls = select_balanced(groups, args.per_control, rng)
    selected_control_targets = {
        stable_key(record, -1, "selected_control") for record in controls
    }

    replay_pool: list[dict[str, Any]] = []
    seen_replay: set[str] = set()
    for input_path in args.replay_jsonl:
        for line_number, record in iter_jsonl(input_path):
            if not is_stage_source(record, args.stage):
                continue
            if is_explicitly_excluded(record, excluded):
                rejection["replay_explicitly_excluded_key"] += 1
                continue
            text = str(record.get("text") or "")
            if any(f"\u3010{tag}\u3011" in text for tag in required_tags) or any(
                marker in text for marker in LEGACY_CONTROL_MARKERS
            ):
                rejection["replay_existing_control_tag"] += 1
                continue
            dedupe_key = stable_key(record, line_number, input_path)
            if dedupe_key in selected_control_targets:
                rejection["replay_matches_selected_control_target"] += 1
                continue
            if dedupe_key in seen_replay:
                continue
            seen_replay.add(dedupe_key)
            result = dict(record)
            result["control_stage"] = args.stage
            result["control_dataset_role"] = "replay"
            replay_pool.append(result)

    replay_target = round(len(controls) * args.replay_ratio)
    if replay_target > len(replay_pool):
        raise ValueError(f"requested {replay_target} replay rows but only {len(replay_pool)} are available")
    replay = stratified_sample(replay_pool, replay_target, rng) if replay_target else []
    records = controls + replay
    rng.shuffle(records)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = {
        "stage": args.stage,
        "candidate_inputs": args.candidates_jsonl,
        "replay_inputs": args.replay_jsonl,
        "approved_keys_file": args.approved_keys_file,
        "excluded_keys_file": args.exclude_keys_file,
        "explicit_excluded_key_count": len(excluded),
        "require_approved_effort": args.require_approved_effort,
        "per_control": args.per_control,
        "replay_ratio": args.replay_ratio,
        "eligible_by_control_tag": {tag: len(rows) for tag, rows in sorted(groups.items())},
        "selected_by_control_tag": dict(Counter(record["control_tag"] for record in controls)),
        "selected_by_source": dict(Counter(source_name(record) for record in controls)),
        "selected_by_emotion": dict(Counter(emotion_family(record.get("text")) for record in controls)),
        "selected_by_source_emotion": dict(
            Counter(f"{source}|{emotion}" for source, emotion in map(source_emotion_stratum, controls))
        ),
        "replay_selected": len(replay),
        "output_records": len(records),
        "rejections": dict(rejection),
    }
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
