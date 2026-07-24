"""Export natural control candidates into a categorized review library.

The exporter is intentionally downstream of ``build_control_candidates.py``.
It never modifies source audio or candidate JSONL files.  It creates a review
copy (or hard-link/symlink) of every usable candidate and deterministic,
stratified speed/effort review subsets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

try:
    from .common import EFFORT_TAGS, SPEED_TAGS, emotion_family, infer_source, iter_jsonl, resolve_audio
except ImportError:  # pragma: no cover - direct CLI invocation.
    from common import EFFORT_TAGS, SPEED_TAGS, emotion_family, infer_source, iter_jsonl, resolve_audio  # type: ignore


ALL_TAGS = frozenset((*SPEED_TAGS, *EFFORT_TAGS))
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy or link every natural speed/effort candidate into a categorized "
            "human-review library. Source audio and input JSONL are read only."
        )
    )
    parser.add_argument("--input-jsonl", required=True, help="Output from build_control_candidates.py.")
    parser.add_argument("--output-dir", required=True, help="Must not already contain files.")
    parser.add_argument("--audio-root", default=None, help="Prefix for relative audio paths, when needed.")
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help=(
            "How to materialize candidate audio. 'copy' is safest; 'hardlink' "
            "avoids duplicate disk use on the same filesystem and falls back to copy."
        ),
    )
    parser.add_argument(
        "--speed-review-per-tag",
        type=int,
        default=50,
        help="Stratified speed-review rows per speed tag; 0 disables speed subset export.",
    )
    parser.add_argument(
        "--effort-review-per-tag",
        type=int,
        default=50,
        help="Stratified effort-review rows per effort tag; 0 disables effort subset export.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def value_text(value: Any) -> str:
    return "" if value is None else str(value)


def candidate_id(record: dict[str, Any], line_number: int) -> str:
    for field in ("control_candidate_id", "record_id", "key", "id", "item_name"):
        value = value_text(record.get(field)).strip()
        if value:
            return value
    return f"line:{line_number}"


def source_name(record: dict[str, Any]) -> str:
    value = value_text(record.get("source") or record.get("control_source")).strip().lower()
    return value or infer_source(record) or "unknown"


def safe_component(value: str, fallback: str) -> str:
    cleaned = SAFE_NAME_RE.sub("_", value).strip("._")
    return cleaned[:80] or fallback


def destination_name(record: dict[str, Any], source: str, identifier: str, source_audio: str) -> str:
    suffix = Path(source_audio).suffix.lower()
    if not suffix or len(suffix) > 12:
        suffix = ".wav"
    digest = hashlib.sha256(
        f"{record.get('control_tag')}|{source}|{identifier}|{source_audio}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{safe_component(source, 'source')}__{safe_component(identifier, 'item')}__{digest}{suffix}"


def copy_audio(source: Path, destination: Path, mode: str) -> str:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing export: {destination}")
    if mode == "copy":
        shutil.copy2(source, destination)
        return "copy"
    if mode == "symlink":
        os.symlink(source, destination)
        return "symlink"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy_fallback"


def selection_stratum(record: dict[str, Any]) -> tuple[str, str, str]:
    selection = record.get("control_selection")
    nested = selection.get("stratum") if isinstance(selection, dict) else None
    if not isinstance(nested, dict):
        nested = {}
    source = value_text(nested.get("source") or record.get("source") or infer_source(record)).lower()
    emotion = value_text(nested.get("emotion_family") or emotion_family(record.get("text"))).lower()
    duration = value_text(nested.get("duration_bin") or "unknown").lower()
    return source or "unknown", emotion or "unknown", duration or "unknown"


def stratified_sample(records: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    if count >= len(records):
        return list(records)
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[selection_stratum(record)].append(record)
    for values in buckets.values():
        rng.shuffle(values)

    tie_breaker = {key: rng.random() for key in buckets}
    source_counts: Counter[str] = Counter()
    emotion_counts: Counter[str] = Counter()
    duration_counts: Counter[str] = Counter()
    stratum_counts: Counter[tuple[str, str, str]] = Counter()
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        available = [key for key, values in buckets.items() if values]
        if not available:
            break
        key = min(
            available,
            key=lambda item: (
                source_counts[item[0]],
                emotion_counts[item[1]],
                duration_counts[item[2]],
                stratum_counts[item],
                tie_breaker[item],
            ),
        )
        selected.append(buckets[key].pop())
        source_counts[key[0]] += 1
        emotion_counts[key[1]] += 1
        duration_counts[key[2]] += 1
        stratum_counts[key] += 1
    return selected


def control_metric(record: dict[str, Any], name: str) -> Any:
    metrics = record.get("control_metrics")
    if isinstance(metrics, dict):
        for group in ("alignment", "active_loudness"):
            nested = metrics.get(group)
            if isinstance(nested, dict) and name in nested:
                return nested[name]
    return record.get(name)


def export_row(record: dict[str, Any], exported_audio: str, source_audio: str, method: str, line_number: int) -> dict[str, Any]:
    tag = value_text(record.get("control_tag"))
    return {
        "candidate_id": candidate_id(record, line_number),
        "control_tag": tag,
        "control_kind": value_text(record.get("control_kind")),
        "source": source_name(record),
        "recording_group": value_text(record.get("recording_group")),
        "text": value_text(record.get("text")),
        "source_audio": source_audio,
        "exported_audio": exported_audio,
        "materialization": method,
        "speed_rate_cps": control_metric(record, "speed_rate_cps"),
        "speed_rate_metric": value_text(control_metric(record, "speed_rate_metric")),
        "asr_cer": control_metric(record, "asr_cer"),
        "alignment_coverage": control_metric(record, "alignment_coverage"),
        "pause_excluded_cps": control_metric(record, "pause_excluded_cps"),
        "char_pause_ratio": control_metric(record, "char_pause_ratio"),
        "active_rms_dbfs_p50": control_metric(record, "active_rms_dbfs_p50"),
        "active_lufs_p50": control_metric(record, "active_lufs_p50"),
        "emotion_family": emotion_family(record.get("text")),
        "review_status": "",
        "review_note": "",
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def relative_audio_for_review(exported_audio: str) -> str:
    return "../" + exported_audio.replace("\\", "/")


def write_speed_html(path: Path, rows: list[dict[str, Any]]) -> None:
    body = "".join(
        "<tr>"
        f"<td>{html.escape(value_text(row['review_id']))}</td>"
        f"<td>{html.escape(value_text(row['control_tag']))}</td>"
        f"<td>{html.escape(value_text(row['source']))}</td>"
        f"<td>{html.escape(value_text(row['speed_rate_cps']))}</td>"
        f"<td><audio controls preload='none' src='{quote(value_text(row['review_audio']), safe='/')}'></audio></td>"
        f"<td>{html.escape(value_text(row['text']))}</td>"
        "</tr>"
        for row in rows
    )
    document = """<!doctype html><html><head><meta charset='utf-8'>
<title>Speed review</title><style>
body{font-family:Arial,sans-serif;margin:20px}table{border-collapse:collapse;width:100%}
td,th{border:1px solid #ddd;padding:6px;vertical-align:top}audio{width:220px}
</style></head><body><h1>Speed review</h1>
<p>Listen for genuinely slow/normal/fast articulation. Check the CSV for pause and alignment diagnostics before rejecting a row.</p>
<table><thead><tr><th>ID</th><th>Label</th><th>Source</th><th>Pause-excluded CPS</th><th>Audio</th><th>Text</th></tr></thead><tbody>"""
    path.write_text(document + body + "</tbody></table></body></html>\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.speed_review_per_tag < 0 or args.effort_review_per_tag < 0:
        raise ValueError("review counts must be >= 0")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_root = output_dir / "audio"
    manifest_root = output_dir / "manifests"
    review_root = output_dir / "review"
    manifest_root.mkdir()
    review_root.mkdir()

    errors: list[dict[str, str]] = []
    exported_records: list[dict[str, Any]] = []
    candidates_by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_counts: Counter[str] = Counter()
    materialization_counts: Counter[str] = Counter()
    seen_candidate_ids: set[tuple[str, str]] = set()

    for line_number, record in iter_jsonl(args.input_jsonl):
        tag = value_text(record.get("control_tag"))
        if tag not in ALL_TAGS:
            errors.append({"line": str(line_number), "reason": "unsupported_or_missing_control_tag", "audio": ""})
            continue
        identifier = candidate_id(record, line_number)
        dedupe_key = (tag, identifier)
        if dedupe_key in seen_candidate_ids:
            errors.append({"line": str(line_number), "reason": "duplicate_candidate_id_and_tag", "audio": ""})
            continue
        seen_candidate_ids.add(dedupe_key)
        source_audio = resolve_audio(record, args.audio_root)
        if not source_audio or not Path(source_audio).is_file():
            errors.append({"line": str(line_number), "reason": "missing_audio", "audio": value_text(source_audio)})
            continue
        source = source_name(record)
        destination_dir = audio_root / tag / safe_component(source, "unknown")
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / destination_name(record, source, identifier, source_audio)
        method = copy_audio(Path(source_audio), destination, args.copy_mode)
        relative_path = destination.relative_to(output_dir).as_posix()
        row = export_row(record, relative_path, source_audio, method, line_number)
        exported_records.append(row)
        candidates_by_tag[tag].append(record)
        source_counts[source] += 1
        materialization_counts[method] += 1

    manifest_fields = list(exported_records[0]) if exported_records else [
        "candidate_id", "control_tag", "control_kind", "source", "recording_group", "text",
        "source_audio", "exported_audio", "materialization", "speed_rate_cps", "speed_rate_metric",
        "asr_cer", "alignment_coverage", "pause_excluded_cps", "char_pause_ratio",
        "active_rms_dbfs_p50", "active_lufs_p50", "emotion_family", "review_status", "review_note",
    ]
    write_csv(manifest_root / "library_manifest.csv", exported_records, manifest_fields)
    with (manifest_root / "errors.json").open("w", encoding="utf-8") as handle:
        json.dump(errors, handle, ensure_ascii=False, indent=2)

    rng = random.Random(args.seed)
    review_records: dict[str, list[dict[str, Any]]] = {"speed": [], "effort": []}
    lookup = {row["candidate_id"]: row for row in exported_records}
    for tag in SPEED_TAGS:
        for record in stratified_sample(candidates_by_tag.get(tag, []), args.speed_review_per_tag, rng):
            review_records["speed"].append(record)
    for tag in EFFORT_TAGS:
        for record in stratified_sample(candidates_by_tag.get(tag, []), args.effort_review_per_tag, rng):
            review_records["effort"].append(record)

    for kind, records in review_records.items():
        write_jsonl(review_root / f"{kind}_review_candidates.jsonl", records)
        rows: list[dict[str, Any]] = []
        for index, record in enumerate(records, start=1):
            identifier = candidate_id(record, -1)
            base = dict(lookup[identifier])
            base["review_id"] = f"{'S' if kind == 'speed' else 'E'}{index:04d}"
            base["review_audio"] = relative_audio_for_review(value_text(base["exported_audio"]))
            rows.append(base)
        write_csv(review_root / f"{kind}_review.csv", rows, ["review_id", *manifest_fields, "review_audio"])
        if kind == "speed":
            write_speed_html(review_root / "speed_review.html", rows)

    report = {
        "input_jsonl": args.input_jsonl,
        "output_dir": str(output_dir),
        "copy_mode_requested": args.copy_mode,
        "exported_records": len(exported_records),
        "errors": len(errors),
        "by_control_tag": dict(sorted(Counter(row["control_tag"] for row in exported_records).items())),
        "by_source": dict(sorted(source_counts.items())),
        "materialization": dict(sorted(materialization_counts.items())),
        "speed_review_records": len(review_records["speed"]),
        "effort_review_records": len(review_records["effort"]),
        "next_step": {
            "speed": "Open review/speed_review.html and fill review/speed_review.csv.",
            "effort": (
                "Use review/effort_review_candidates.jsonl as input to export_effort_audit.py "
                "for blind raw-versus-equalized listening."
            ),
        },
    }
    (manifest_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
