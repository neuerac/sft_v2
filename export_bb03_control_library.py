"""Export a sorted, limited BB03-only control-audio listening library.

The input is the screened candidate JSONL from build_control_candidates.py.
This script deliberately does not derive labels or alter source recordings. It
selects at most a fixed number of BB03 examples per control tag, materializes
them in six folders, and creates an HTML listener plus auditable manifests.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

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


TAG_ORDER = (*SPEED_TAGS, *EFFORT_TAGS)
ALL_TAGS = frozenset(TAG_ORDER)
BB03_SOURCE = "bb03"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select, sort, copy, and HTML-review up to N screened BB03 "
            "speed/effort candidates per control tag."
        )
    )
    parser.add_argument(
        "--input-jsonl",
        required=True,
        help=(
            "Rows from build_control_candidates.py. Mixed-source files are "
            "allowed; non-BB03 rows are skipped."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New or empty output directory for the BB03 listening library.",
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
    parser.add_argument(
        "--per-tag",
        type=int,
        default=100,
        help="Maximum exported and listed examples per control tag (default: 100).",
    )
    return parser.parse_args()


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metric(record: dict[str, Any], name: str) -> Any:
    metrics = record.get("control_metrics")
    if isinstance(metrics, dict):
        for group in ("alignment", "active_loudness"):
            nested = metrics.get(group)
            if isinstance(nested, dict) and name in nested:
                return nested[name]
    return record.get(name)


def _effort_relative_lufs(record: dict[str, Any]) -> tuple[float | None, str]:
    raw_lufs = _finite(_metric(record, "active_lufs_p50"))
    if raw_lufs is None:
        return None, "missing_active_lufs_p50"

    selection = record.get("control_selection")
    threshold = selection.get("effort_group_threshold") if isinstance(selection, dict) else None
    metrics = threshold.get("metrics") if isinstance(threshold, dict) else None
    values = metrics.get("active_lufs_p50") if isinstance(metrics, dict) else None
    if isinstance(values, list) and len(values) >= 4:
        lower = _finite(values[0])
        upper = _finite(values[3])
        if lower is not None and upper is not None and upper > lower:
            return (raw_lufs - lower) / (upper - lower), "relative_active_lufs_p50"
    return raw_lufs, "active_lufs_p50_fallback"


def _sort_value(record: dict[str, Any], tag: str) -> tuple[float | None, str]:
    if tag.startswith("speed_"):
        # Candidates may have been classified with full-audio VAD CPS rather
        # than the legacy pause-excluded rate. Preserve the classifier's
        # selected metric when choosing the listening subset and its order.
        value = _finite(_metric(record, "speed_rate_cps"))
        if value is not None:
            metric = str(_metric(record, "speed_rate_metric") or "speed_rate_cps")
            return value, metric
        value = _finite(_metric(record, "pause_excluded_cps"))
        return value, "pause_excluded_cps"
    return _effort_relative_lufs(record)


def _selection_policy(tag: str) -> str:
    if tag in ("speed_slow", "effort_soft"):
        return "lowest_values_first"
    if tag in ("speed_fast", "effort_strong"):
        return "highest_values_first"
    return "closest_to_category_median_then_ascending"


def _select_and_sort(
    tag: str, items: list[dict[str, Any]], per_tag: int
) -> list[dict[str, Any]]:
    if not items:
        return []
    if tag in ("speed_slow", "effort_soft"):
        return sorted(
            items,
            key=lambda item: (float(item["sort_value"]), str(item["candidate_id"])),
        )[:per_tag]
    if tag in ("speed_fast", "effort_strong"):
        return sorted(
            items,
            key=lambda item: (-float(item["sort_value"]), str(item["candidate_id"])),
        )[:per_tag]

    center = statistics.median(float(item["sort_value"]) for item in items)
    closest = sorted(
        items,
        key=lambda item: (
            abs(float(item["sort_value"]) - center),
            str(item["candidate_id"]),
        ),
    )[:per_tag]
    return sorted(
        closest,
        key=lambda item: (float(item["sort_value"]), str(item["candidate_id"])),
    )


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
    rank: int,
    sort_metric: str,
    sort_value: float,
    selection_policy: str,
) -> dict[str, Any]:
    result = dict(record)
    result["source"] = BB03_SOURCE
    result["source_audio"] = source_audio
    result["library_audio"] = exported_audio
    result["library_materialization"] = materialization
    result["library_manifest_line"] = line_number
    result["library_rank"] = rank
    result["library_sort_metric"] = sort_metric
    result["library_sort_value"] = sort_value
    result["library_selection_policy"] = selection_policy
    return result


def _manifest_fields(rows: list[dict[str, Any]]) -> list[str]:
    if rows:
        return list(rows[0])
    return [
        "rank",
        "control_tag",
        "sort_metric",
        "sort_value",
        "selection_policy",
        "candidate_id",
        "source",
        "text",
        "exported_audio",
    ]


def _html_number(value: Any) -> str:
    number = _finite(value)
    return "" if number is None else f"{number:.4f}"


def _write_listener_html(path: Path, rows: list[dict[str, Any]]) -> None:
    by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_tag[str(row["control_tag"])].append(row)

    sections: list[str] = []
    for tag in TAG_ORDER:
        tag_rows = by_tag.get(tag, [])
        entries = []
        for row in tag_rows:
            audio_path = quote(str(row["exported_audio"]), safe="/._-")
            entries.append(
                "<tr>"
                f"<td>{html.escape(str(row['rank']))}</td>"
                f"<td>{html.escape(str(row['sort_metric']))}</td>"
                f"<td>{html.escape(_html_number(row['sort_value']))}</td>"
                f"<td>{html.escape(_html_number(row.get('speed_rate_cps')))}</td>"
                f"<td>{html.escape(_html_number(row.get('active_lufs_p50')))}</td>"
                f"<td><audio controls preload='none' src='{audio_path}'></audio></td>"
                f"<td>{html.escape(str(row.get('text') or ''))}</td>"
                "</tr>"
            )
        empty = "<tr><td colspan='7'>No eligible BB03 candidate exported.</td></tr>" if not entries else ""
        sections.append(
            f"<section id='{html.escape(tag)}'><h2>{html.escape(tag)} ({len(tag_rows)})</h2>"
            "<table><thead><tr><th>Rank</th><th>Sort metric</th><th>Sort value</th>"
            "<th>Selected-rate CPS</th><th>Active LUFS P50</th><th>Audio</th>"
            "<th>Text</th></tr></thead><tbody>"
            + "".join(entries)
            + empty
            + "</tbody></table></section>"
        )

    document = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>BB03 control listening library</title>"
        "<style>"
        "body{font-family:Arial,sans-serif;margin:24px;line-height:1.4}"
        "nav{position:sticky;top:0;background:white;padding:8px 0;border-bottom:1px solid #ddd}"
        "nav a{margin-right:12px}section{margin:30px 0}"
        "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:6px;vertical-align:top}"
        "th{background:#f4f4f4}audio{width:220px}"
        "</style></head><body><h1>BB03 Control Listening Library</h1><nav>"
        + "".join(f"<a href='#{html.escape(tag)}'>{html.escape(tag)}</a>" for tag in TAG_ORDER)
        + "</nav>"
        + "".join(sections)
        + "</body></html>\n"
    )
    path.write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.per_tag < 1:
        raise ValueError("--per-tag must be >= 1")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_root = output_dir / "audio"
    manifest_root = output_dir / "manifests"
    manifest_root.mkdir()
    for tag in TAG_ORDER:
        (audio_root / tag).mkdir(parents=True, exist_ok=True)

    errors: list[dict[str, str]] = []
    eligible_by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    input_records = 0
    candidate_like_records = 0
    skipped_non_bb03 = 0
    seen_ids: set[tuple[str, str]] = set()

    for line_number, record in iter_jsonl(args.input_jsonl):
        input_records += 1
        if source_name(record) != BB03_SOURCE:
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

        sort_value, sort_metric = _sort_value(record, tag)
        if sort_value is None:
            errors.append(_error(line_number, f"missing_sort_metric:{sort_metric}", record, source_audio))
            continue
        eligible_by_tag[tag].append(
            {
                "record": record,
                "line_number": line_number,
                "candidate_id": identifier,
                "source_audio": source_audio,
                "sort_value": sort_value,
                "sort_metric": sort_metric,
            }
        )

    if input_records and candidate_like_records == 0 and not skipped_non_bb03:
        raise ValueError(
            "no BB03 control candidates found: input looks like an alignment manifest. "
            "Run build_control_candidates.py first, then pass its candidate JSONL here."
        )

    exported_rows: list[dict[str, Any]] = []
    exported_candidates: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    available_counts = {tag: len(eligible_by_tag[tag]) for tag in TAG_ORDER}
    width = max(3, len(str(args.per_tag)))

    for tag in TAG_ORDER:
        selected = _select_and_sort(tag, eligible_by_tag[tag], args.per_tag)
        policy = _selection_policy(tag)
        for rank, item in enumerate(selected, start=1):
            record = item["record"]
            destination = audio_root / tag / (
                f"{rank:0{width}d}__"
                + destination_name(record, BB03_SOURCE, str(item["candidate_id"]), str(item["source_audio"]))
            )
            materialization = copy_audio(Path(str(item["source_audio"])), destination, args.copy_mode)
            exported_audio = destination.relative_to(output_dir).as_posix()
            row = export_row(
                record,
                exported_audio,
                str(item["source_audio"]),
                materialization,
                int(item["line_number"]),
            )
            row["rank"] = rank
            row["sort_metric"] = item["sort_metric"]
            row["sort_value"] = float(item["sort_value"])
            row["selection_policy"] = policy
            exported_rows.append(row)
            exported_candidates.append(
                _candidate_export_record(
                    record,
                    exported_audio,
                    str(item["source_audio"]),
                    materialization,
                    int(item["line_number"]),
                    rank,
                    str(item["sort_metric"]),
                    float(item["sort_value"]),
                    policy,
                )
            )
            selected_counts[tag] += 1

    write_jsonl(manifest_root / "bb03_control_candidates.jsonl", exported_candidates)
    write_csv(
        manifest_root / "bb03_library_manifest.csv",
        exported_rows,
        _manifest_fields(exported_rows),
    )
    (manifest_root / "errors.json").write_text(
        json.dumps(errors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_listener_html(output_dir / "index.html", exported_rows)

    report = {
        "input_jsonl": args.input_jsonl,
        "output_dir": str(output_dir),
        "input_records": input_records,
        "skipped_non_bb03_records": skipped_non_bb03,
        "bb03_records_with_control_tag": candidate_like_records,
        "per_tag_requested": args.per_tag,
        "eligible_by_control_tag": available_counts,
        "exported_by_control_tag": {tag: selected_counts.get(tag, 0) for tag in TAG_ORDER},
        "by_control_tag": {tag: selected_counts.get(tag, 0) for tag in TAG_ORDER},
        "exported_records": len(exported_rows),
        "errors": len(errors),
        "copy_mode_requested": args.copy_mode,
        "materialization": dict(sorted(Counter(row["materialization"] for row in exported_rows).items())),
        "selection_policy": {
            "speed_slow": "lowest selected speed-rate CPS first",
            "speed_normal": "closest to selected-category speed-rate CPS median, then ascending",
            "speed_fast": "highest selected speed-rate CPS first",
            "effort_soft": "lowest within-group relative active_lufs_p50 first",
            "effort_normal": "closest to selected-category relative active_lufs_p50 median, then ascending",
            "effort_strong": "highest within-group relative active_lufs_p50 first",
        },
        "listener_html": "index.html",
        "library_layout": {tag: f"audio/{tag}/" for tag in TAG_ORDER},
    }
    report_path = manifest_root / "bb03_export_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
