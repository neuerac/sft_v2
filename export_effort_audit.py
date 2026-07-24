"""Export effort candidates with equalized active-speech previews for human review."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

if __package__:
    from .common import iter_jsonl, resolve_audio
else:  # Direct execution must prefer this directory over a site-packages ``common``.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import iter_jsonl, resolve_audio  # type: ignore[no-redef]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create raw/equalized listening previews for effort labels.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audio-root", default=None)
    parser.add_argument("--target-active-rms-dbfs", type=float, default=-24.0)
    parser.add_argument("--vad-top-db", type=float, default=35.0)
    parser.add_argument("--per-tag", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-labels", action="store_true", help="Show automatic labels in the HTML reviewer page.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def dbfs(value: float) -> float:
    return 20.0 * math.log10(max(value, 1e-12))


def active_rms_dbfs(audio: np.ndarray, sample_rate: int, top_db: float) -> float:
    frame = max(1, round(sample_rate * 0.03))
    hop = max(1, round(sample_rate * 0.01))
    if len(audio) <= frame:
        return dbfs(float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))))
    starts = np.arange(0, len(audio) - frame + 1, hop, dtype=np.int64)
    squares = np.square(audio, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(squares)))
    rms = np.sqrt((cumulative[starts + frame] - cumulative[starts]) / frame)
    values = np.asarray([dbfs(float(value)) for value in rms], dtype=np.float64)
    threshold = max(-50.0, float(np.percentile(values, 95)) - top_db)
    active = values[values >= threshold]
    if not len(active):
        raise ValueError("no active speech frames")
    return float(np.percentile(active, 50))


def clipped_gain(audio: np.ndarray, requested_db: float, ceiling_dbfs: float = -1.0) -> tuple[np.ndarray, float]:
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    peak_dbfs = dbfs(peak)
    allowed_gain = ceiling_dbfs - peak_dbfs
    actual_gain = min(requested_db, allowed_gain)
    return (audio * (10.0 ** (actual_gain / 20.0))).astype(np.float32), actual_gain


def review_stratum(record: dict[str, Any]) -> tuple[str, str, str]:
    """Read the candidate's audit strata without exposing its effort label."""
    selection = record.get("control_selection")
    stratum = selection.get("stratum") if isinstance(selection, dict) else None
    if not isinstance(stratum, dict):
        stratum = {}
    emotion = str(stratum.get("emotion_family") or "unknown")
    speed = str(stratum.get("speed_label") or record.get("speed_label") or "unknown")
    duration = str(stratum.get("duration_bin") or "unknown")
    return emotion, speed, duration


def stratified_review_sample(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    """Cover emotion, speed, and duration strata as evenly as possible."""
    if count >= len(rows):
        return list(rows)
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in rows:
        buckets[review_stratum(record)].append(record)
    for values in buckets.values():
        rng.shuffle(values)

    tie_breaker = {stratum: rng.random() for stratum in buckets}
    emotion_counts: Counter[str] = Counter()
    speed_counts: Counter[str] = Counter()
    duration_counts: Counter[str] = Counter()
    stratum_counts: Counter[tuple[str, str, str]] = Counter()
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        available = [stratum for stratum, values in buckets.items() if values]
        if not available:
            break
        stratum = min(
            available,
            key=lambda key: (
                emotion_counts[key[0]],
                speed_counts[key[1]],
                duration_counts[key[2]],
                stratum_counts[key],
                tie_breaker[key],
            ),
        )
        selected.append(buckets[stratum].pop())
        emotion_counts[stratum[0]] += 1
        speed_counts[stratum[1]] += 1
        duration_counts[stratum[2]] += 1
        stratum_counts[stratum] += 1
    if len(selected) != count:
        raise RuntimeError(f"requested {count} review records but selected only {len(selected)}")
    return selected


def make_html(rows: list[dict[str, str]], show_labels: bool) -> str:
    label_header = "<th>自动标签</th>" if show_labels else ""
    body = "".join(
        "<tr>"
        + (f"<td>{html.escape(row['tag'])}</td>" if show_labels else f"<td>{html.escape(row['review_id'])}</td>")
        + f"<td><audio controls preload='none' src='{quote(row['raw_file'])}'></audio></td>"
        + f"<td><audio controls preload='none' src='{quote(row['equalized_file'])}'></audio></td>"
        + f"<td>{html.escape(row['text'])}</td>"
        + f"<td>{html.escape(row['source_audio'])}</td>"
        + "</tr>"
        for row in rows
    )
    first_header = label_header or "<th>试听编号</th>"
    return """<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<title>Effort blind audit</title><style>body{font-family:Arial,'Microsoft YaHei';margin:20px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:7px;vertical-align:top}audio{width:230px}</style>
</head><body><h1>发声力度试听</h1><p>比较原始音频与活动语音响度统一后的版本。统一后仍显著有力或轻柔，才可保留 effort 标签。</p>
<table><thead><tr>""" + first_header + "<th>原始</th><th>统一响度</th><th>文本</th><th>原路径</th></tr></thead><tbody>" + body + "</tbody></table></body></html>"


def main() -> None:
    args = parse_args()
    if args.per_tag < 1:
        raise ValueError("--per-tag must be >= 1")
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("missing dependency 'soundfile'; install soundfile numpy") from exc

    grouped: dict[str, list[dict]] = defaultdict(list)
    for _, record in iter_jsonl(args.input_jsonl):
        tag = str(record.get("control_tag") or "")
        if tag.startswith("effort_"):
            grouped[tag].append(record)
    if not grouped:
        raise ValueError("input contains no effort candidates")
    rng = random.Random(args.seed)
    selected = []
    for tag, rows in sorted(grouped.items()):
        selected.extend(stratified_review_sample(rows, min(len(rows), args.per_tag), rng))

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    equalized_dir = output_dir / "equalized"
    raw_dir.mkdir(exist_ok=True)
    equalized_dir.mkdir(exist_ok=True)

    rows: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    for index, record in enumerate(selected, start=1):
        source = resolve_audio(record, args.audio_root)
        tag = str(record.get("control_tag"))
        if not source or not Path(source).is_file():
            errors.append({"key": str(record.get("key") or ""), "error": "missing_audio", "audio": str(source or "")})
            continue
        try:
            audio, sr = sf.read(source, dtype="float32", always_2d=False)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            source_rms = active_rms_dbfs(audio, int(sr), args.vad_top_db)
            normalized, gain = clipped_gain(audio, args.target_active_rms_dbfs - source_rms)
            # File names are deliberately label-free so opening index.html is
            # a real blind audit unless --show-labels was explicitly requested.
            stem = f"R{index:04d}"
            raw_path = raw_dir / f"{stem}.wav"
            normalized_path = equalized_dir / f"{stem}.wav"
            sf.write(raw_path, audio, int(sr))
            sf.write(normalized_path, normalized, int(sr))
            approval_id = str(record.get("control_candidate_id") or record.get("key") or record.get("record_id") or "")
            rows.append({
                "review_id": f"R{len(rows) + 1:04d}",
                "tag": tag,
                "raw_file": raw_path.relative_to(output_dir).as_posix(),
                "equalized_file": normalized_path.relative_to(output_dir).as_posix(),
                "text": str(record.get("text") or ""),
                "source_audio": str(source),
                "approval_id": approval_id,
                "source_active_rms_dbfs": f"{source_rms:.3f}",
                "applied_gain_db": f"{gain:.3f}",
                "review_status": "",
                "review_note": "",
            })
        except Exception as exc:  # noqa: BLE001
            errors.append({"key": str(record.get("key") or ""), "error": f"{type(exc).__name__}: {exc}", "audio": str(source)})

    fields = ["review_id", "tag", "raw_file", "equalized_file", "text", "source_audio", "approval_id", "source_active_rms_dbfs", "applied_gain_db", "review_status", "review_note"]
    with (output_dir / "labels_private.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    reviewer_fields = ["review_id", "raw_file", "equalized_file", "text", "source_audio", "approval_id", "source_active_rms_dbfs", "applied_gain_db", "review_status", "review_note"]
    with (output_dir / "review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=reviewer_fields)
        writer.writeheader()
        writer.writerows([{field: row[field] for field in reviewer_fields} for row in rows])
    with (output_dir / "errors.json").open("w", encoding="utf-8") as handle:
        json.dump(errors, handle, ensure_ascii=False, indent=2)
    (output_dir / "index.html").write_text(make_html(rows, args.show_labels), encoding="utf-8")
    report = {
        "selected": len(selected),
        "exported": len(rows),
        "errors": len(errors),
        "by_tag": dict(Counter(row["tag"] for row in rows)),
        "selected_by_emotion_speed_duration": dict(
            Counter("|".join(review_stratum(record)) for record in selected)
        ),
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
