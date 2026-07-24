"""Build high-confidence natural speed and vocal-effort control candidates.

The input is the canonical JSONL emitted by ``build_alignment_manifest.py``.
It deliberately does not synthesize tempo or gain variants: every output row
points at the original recording and carries enough provenance to audit the
pseudo-label later. The legacy speed path uses the alignment manifest's
``pause_excluded_cps``. A strict speed-only profile is also available for
pilot training: it uses full-audio VAD articulation CPS and rejects CTC
alignments, pauses, and non-speech annotations that make a rate label unsafe.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:  # Supports both ``python -m Script.control_pipeline...`` and direct use.
    from .common import (
        CJK_RE,
        add_control_tags,
        clean_spoken_text,
        count_spoken_units,
        emotion_family,
        emotion_tags,
        event_tags,
        finite_float,
        infer_recording_group,
        infer_source,
        is_speed_excluded,
        iter_jsonl,
        resolve_audio,
        strip_emotion_tags,
        write_jsonl,
    )
except ImportError:  # pragma: no cover - exercised when run as a script.
    from common import (  # type: ignore[no-redef]
        CJK_RE,
        add_control_tags,
        clean_spoken_text,
        count_spoken_units,
        emotion_family,
        emotion_tags,
        event_tags,
        finite_float,
        infer_recording_group,
        infer_source,
        is_speed_excluded,
        iter_jsonl,
        resolve_audio,
        strip_emotion_tags,
        write_jsonl,
    )


VERSION = "natural_speed_effort_candidates_v4"
SUCCESS_STATUSES = frozenset({"ok", "success", "aligned", "complete", "completed"})
CONTROL_TAG_RE = re.compile(
    r"【(?:speed|effort|volume)(?:_[^】]*)?】|【loud】", re.IGNORECASE
)

# The canonical manifest is flat, but these aliases keep the builder useful for
# manifests produced during earlier pipeline iterations.
NESTED_MAPPINGS = (
    "alignment",
    "alignment_metrics",
    "audio_metrics",
    "active_speech_metrics",
    "loudness_metrics",
    "audio_control_metrics",
    "loud_scan_metrics",
    "metrics",
)

FIELD_ALIASES = {
    "status": ("status",),
    "alignment_status": ("alignment_status", "align_status"),
    "asr_cer": ("asr_cer", "cer", "character_error_rate"),
    "coverage": ("alignment_coverage", "coverage", "aligned_coverage"),
    "speech_start": ("speech_start_sec", "aligned_start_sec", "first_speech_sec"),
    "speech_end": ("speech_end_sec", "aligned_end_sec", "last_speech_sec"),
    "speech_span": ("speech_span_sec", "aligned_span_sec", "speech_duration_sec"),
    "audio_duration": ("audio_duration_sec", "duration_sec", "duration"),
    "global_vad_active_duration": (
        "vad_active_duration_sec",
        "active_speech_duration_sec",
        "active_duration_sec",
        "active_duration_sec_est",
    ),
    "speech_vad_active_duration": (
        "speech_vad_active_duration_sec",
        "aligned_vad_active_duration_sec",
    ),
    "active_duration": (
        "speech_vad_active_duration_sec",
        "vad_active_duration_sec",
        "active_speech_duration_sec",
        "active_duration_sec",
        "active_duration_sec_est",
    ),
    "active_ratio": (
        "speech_vad_active_ratio",
        "vad_active_ratio",
        "speech_ratio",
        "speech_ratio_est",
        "active_speech_ratio",
    ),
    "pause_ratio": ("pause_ratio", "internal_pause_ratio"),
    "pause_excluded_cps": ("pause_excluded_cps",),
    "pause_excluded_duration": ("pause_excluded_duration_sec",),
    "char_pause_count": ("char_pause_count",),
    "char_pause_duration": ("char_pause_duration_sec",),
    "char_pause_ratio": ("char_pause_ratio",),
    # This is intentionally only used with --allow-legacy-speed-fallback.
    # It is the old first-to-last aligned span rate, never VAD articulation CPS.
    "legacy_speech_cps": ("speech_cps", "speed_cps", "units_per_sec"),
    "rms_p25": (
        "active_rms_dbfs_p25",
        "active_rms_p25_dbfs",
        "active_frame_rms_dbfs_p25",
        "rms_dbfs_p25",
    ),
    "rms_p50": (
        "active_rms_dbfs_p50",
        "active_rms_p50_dbfs",
        "active_frame_rms_dbfs_p50",
        "rms_dbfs_p50",
    ),
    "lufs_p25": (
        "active_lufs_p25",
        "short_term_lufs_p25",
        "short_term_lufs_db_p25",
        "lufs_short_term_p25",
        "lufs_p25",
    ),
    "lufs_p50": (
        "active_lufs_p50",
        "short_term_lufs_p50",
        "short_term_lufs_db_p50",
        "lufs_short_term_p50",
        "lufs_p50",
    ),
    "integrated_lufs": ("lufs_i", "integrated_lufs", "lufs"),
    "clipping_ratio": ("clipping_ratio", "clip_ratio", "clipped_sample_ratio"),
    "dynamic_range": (
        "dynamic_range_db",
        "active_rms_span_p90_p10_db",
        "active_dynamic_range_db",
        "loudness_range_lu",
        "crest_db",
    ),
    "noise_floor": ("noise_floor_dbfs", "vad_noise_floor_dbfs", "noise_gate_dbfs"),
    "snr": ("snr_db", "signal_to_noise_db"),
    "paralinguistic_ratio": (
        "paralinguistic_ratio",
        "non_speech_ratio",
        "long_non_speech_ratio",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create natural three-tier speed and vocal-effort training candidates "
            "from canonical alignment manifests."
        )
    )
    parser.add_argument(
        "--input-jsonl",
        "--input_jsonl",
        dest="input_jsonls",
        nargs="+",
        required=True,
        help="One or more canonical alignment JSONL manifests.",
    )
    parser.add_argument("--output-jsonl", "--output_jsonl", required=True)
    parser.add_argument(
        "--report-json",
        "--report_json",
        default=None,
        help="Defaults to <output-jsonl>.report.json.",
    )
    parser.add_argument(
        "--audio-root",
        "--audio_root",
        default=None,
        help="Optional root for relative audio values in the input manifest.",
    )
    parser.add_argument(
        "--emit-controls",
        choices=("both", "speed", "effort"),
        default="both",
        help="Write both control families by default; use speed for a speed-only pilot.",
    )

    parser.add_argument(
        "--speed-tier-strategy",
        choices=("contiguous", "extreme-middle"),
        default="contiguous",
        help=(
            "contiguous keeps the legacy slow/normal/fast split. "
            "extreme-middle keeps only the slow tail, the middle band, and "
            "the fast tail; records in the two transition bands are omitted."
        ),
    )
    parser.add_argument(
        "--speed-slow-quantile",
        type=float,
        default=1.0 / 3.0,
        help=(
            "Legacy contiguous slow boundary. With --speed-tier-strategy "
            "extreme-middle, this is the slow-tail upper quantile (for "
            "example 0.20)."
        ),
    )
    parser.add_argument(
        "--speed-normal-low-quantile",
        type=float,
        default=0.40,
        help=(
            "Lower quantile of the retained normal band for "
            "--speed-tier-strategy extreme-middle."
        ),
    )
    parser.add_argument(
        "--speed-normal-high-quantile",
        type=float,
        default=0.60,
        help=(
            "Upper quantile of the retained normal band for "
            "--speed-tier-strategy extreme-middle."
        ),
    )
    parser.add_argument(
        "--speed-fast-quantile",
        type=float,
        default=2.0 / 3.0,
        help=(
            "Legacy contiguous fast boundary. With --speed-tier-strategy "
            "extreme-middle, this is the fast-tail lower quantile (for "
            "example 0.80)."
        ),
    )
    parser.add_argument("--min-speed-calibration-records", type=int, default=100)
    parser.add_argument(
        "--min-bb03-calibration-records",
        dest="min_speed_calibration_records",
        type=int,
        default=argparse.SUPPRESS,
        help="Deprecated alias for --min-speed-calibration-records; calibration is global.",
    )
    parser.add_argument(
        "--allow-legacy-speed-fallback",
        action="store_true",
        help=(
            "Permit old manifests missing pause_excluded_cps to use full-span speech_cps. "
            "Never falls back to articulation_cps."
        ),
    )
    parser.add_argument(
        "--speed-clean-profile",
        action="store_true",
        help=(
            "Enable strict speed-only quality gates and default to full-audio "
            "VAD articulation CPS. This is recommended for a first speed pilot."
        ),
    )
    parser.add_argument(
        "--speed-rate-metric",
        choices=("pause_excluded_cps", "speech_span_cps", "full_audio_vad_cps"),
        default=None,
        help=(
            "Metric used to calibrate and label speed. The clean profile defaults "
            "to full_audio_vad_cps, which excludes silent gaps over the complete audio."
        ),
    )
    parser.add_argument(
        "--min-speed-spoken-units",
        type=int,
        default=None,
        help="Optional speed-only minimum lexical-unit count; clean profile defaults to 20.",
    )
    parser.add_argument(
        "--max-speed-spoken-units",
        type=int,
        default=None,
        help="Optional speed-only maximum lexical-unit count; clean profile defaults to 80.",
    )
    parser.add_argument(
        "--max-speed-breath-events",
        type=int,
        default=None,
        help="Optional maximum [breath] markers; clean profile defaults to 2.",
    )
    parser.add_argument(
        "--max-speed-hold-events",
        type=int,
        default=None,
        help="Optional maximum [hold] markers; clean profile defaults to 0.",
    )
    parser.add_argument(
        "--min-speed-ctc-score",
        type=float,
        default=None,
        help="Optional per-character CTC confidence cutoff; clean profile defaults to 0.50.",
    )
    parser.add_argument(
        "--max-speed-low-ctc-score-ratio",
        type=float,
        default=None,
        help="Optional maximum fraction below --min-speed-ctc-score; clean profile defaults to 0.10.",
    )
    parser.add_argument(
        "--min-speed-character-duration-sec",
        type=float,
        default=None,
        help="Optional short-character cutoff; clean profile defaults to 0.035 seconds.",
    )
    parser.add_argument(
        "--max-speed-short-character-ratio",
        type=float,
        default=None,
        help="Optional maximum short-character fraction; clean profile defaults to 0.10.",
    )
    parser.add_argument(
        "--min-speed-vad-alignment-coverage",
        type=float,
        default=None,
        help=(
            "Optional minimum fraction of full-audio VAD speech covered by the CTC "
            "speech interval; clean profile defaults to 0.85."
        ),
    )
    parser.add_argument(
        "--max-speed-long-gap-ratio",
        type=float,
        default=None,
        help=(
            "Optional maximum CTC inter-character-gap share; clean profile defaults "
            "to 0.15 for gaps of at least 0.30 seconds."
        ),
    )
    parser.add_argument(
        "--speed-long-gap-min-sec",
        type=float,
        default=0.30,
        help="Gap duration used by --max-speed-long-gap-ratio (default: 0.30).",
    )
    parser.add_argument(
        "--speed-slow-proxy-emotions",
        default="",
        help=(
            "Comma-separated single-emotion prefixes to deliberately proxy as "
            "speed_slow, for example reserved1,reserved2. Proxy rows are excluded "
            "from rate calibration and their matching emotion tags are removed from "
            "the model text."
        ),
    )
    parser.add_argument(
        "--speed-slow-proxy-allow-extra-events",
        action="store_true",
        help=(
            "Let slow emotion-proxy rows bypass the breath/hold count gates. This is "
            "an explicitly confounded diagnostic option, not a clean speed dataset."
        ),
    )
    parser.add_argument(
        "--allow-speed-paralinguistic",
        action="store_true",
        help=(
            "Do not reject speed candidates for non-speech event annotations or "
            "paralinguistic VAD ratio. Whisper, recitation, and panting remain "
            "excluded because they are separate speaking styles."
        ),
    )
    parser.add_argument("--max-asr-cer", type=float, default=0.12)
    parser.add_argument("--min-alignment-coverage", type=float, default=0.90)
    parser.add_argument(
        "--min-speed-alignment-coverage",
        type=float,
        default=1.0,
        help=(
            "Require this lexical alignment coverage for speed calibration and "
            "speed/effort candidate output. The lower --min-alignment-coverage "
            "still admits rows for basic effort-metric auditing."
        ),
    )
    parser.add_argument(
        "--include-non-cjk-speed",
        action="store_true",
        help=(
            "Allow non-CJK text in global speed calibration. Disabled by default "
            "because English words/sec and Chinese characters/sec are not comparable."
        ),
    )
    parser.add_argument(
        "--require-normal-effort-for-speed",
        action="store_true",
        help=(
            "Only emit speed candidates whose within-group effort label is normal. "
            "Disabled by default so speed controls can use all high-confidence "
            "speed-aligned recordings."
        ),
    )
    parser.add_argument("--min-spoken-units", type=int, default=4)
    parser.add_argument("--min-speech-span-sec", type=float, default=0.60)
    parser.add_argument("--max-speech-span-sec", type=float, default=30.0)
    parser.add_argument(
        "--min-vad-active-ratio",
        type=float,
        default=0.35,
        help="Reject extreme pause-heavy rows even after confirmed gaps are removed.",
    )
    parser.add_argument("--max-pause-ratio", type=float, default=0.65)
    parser.add_argument("--max-paralinguistic-ratio", type=float, default=0.15)

    parser.add_argument("--min-group-records", type=int, default=20)
    parser.add_argument("--effort-soft-quantile", type=float, default=0.20)
    parser.add_argument("--effort-normal-low-quantile", type=float, default=0.40)
    parser.add_argument("--effort-normal-high-quantile", type=float, default=0.60)
    parser.add_argument("--effort-strong-quantile", type=float, default=0.80)
    parser.add_argument("--min-effort-rms-span-db", type=float, default=3.0)
    parser.add_argument("--min-effort-lufs-span-db", type=float, default=3.0)
    parser.add_argument("--max-clipping-ratio", type=float, default=1e-4)
    parser.add_argument("--min-dynamic-range-db", type=float, default=4.0)
    parser.add_argument(
        "--max-noise-floor-dbfs",
        type=float,
        default=-25.0,
        help="Apply only when a noise-floor metric is present; set 0 to disable.",
    )
    parser.add_argument(
        "--min-snr-db",
        type=float,
        default=12.0,
        help="Apply only when an SNR metric is present; set 0 to disable.",
    )
    parser.add_argument(
        "--allow-integrated-lufs-fallback",
        action="store_true",
        help="Use integrated LUFS twice only when active short-term LUFS is unavailable.",
    )
    return parser.parse_args()


def _mappings(record: dict[str, Any]) -> Iterable[dict[str, Any]]:
    yield record
    for name in NESTED_MAPPINGS:
        value = record.get(name)
        if isinstance(value, dict):
            yield value


def _lookup(record: dict[str, Any], names: Iterable[str]) -> Any:
    for mapping in _mappings(record):
        for name in names:
            if name in mapping and mapping[name] is not None:
                return mapping[name]
    return None


def _number(record: dict[str, Any], name: str) -> float | None:
    return finite_float(_lookup(record, FIELD_ALIASES[name]))


def _quantile(values: Iterable[float], level: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a quantile of no values")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * level
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _normalize_source(record: dict[str, Any]) -> str:
    declared = str(record.get("source") or record.get("dataset_source") or "").strip().lower()
    inferred = infer_source(record).strip().lower()
    value = declared or inferred
    if "bb03" in value:
        return "bb03"
    if "instruct" in value:
        return "instruct_tts"
    if "aopeng" in value or "obs_0" in value:
        return "aopeng"
    return value or "unknown"


def _recording_group(record: dict[str, Any], source: str) -> str:
    explicit = str(record.get("recording_group") or "").strip()
    group = explicit or infer_recording_group(record)
    if group.startswith(f"{source}:"):
        return group
    return f"{source}:{group}"


def _record_id(record: dict[str, Any], source: str, manifest: str, line_number: int) -> str:
    for field in ("record_id", "key", "id", "item_id"):
        value = str(record.get(field) or "").strip()
        if value:
            return f"{source}:{value}"
    raw_audio = str(
        record.get("audio_path_raw")
        or record.get("audio_path")
        or record.get("audio")
        or record.get("wav_path")
        or ""
    ).strip()
    if raw_audio:
        normalized_audio = raw_audio.replace("\\", "/")
        return f"{source}:audio:{normalized_audio}"
    return f"{source}:{manifest}:{line_number}"


def _timestamp_entries(record: dict[str, Any]) -> list[tuple[float, float]]:
    values = _lookup(
        record,
        (
            "character_timestamps",
            "word_timestamps",
            "alignment_timestamps",
            "aligned_timestamps",
        ),
    )
    if not isinstance(values, list):
        return []
    output: list[tuple[float, float]] = []
    for value in values:
        if isinstance(value, dict):
            start = finite_float(
                value.get("start_sec", value.get("start", value.get("begin")))
            )
            end = finite_float(value.get("end_sec", value.get("end", value.get("stop"))))
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            start, end = finite_float(value[0]), finite_float(value[1])
        else:
            continue
        if start is not None and end is not None and 0 <= start <= end:
            output.append((start, end))
    return output


def _timestamp_quality_metrics(
    record: dict[str, Any],
    timestamps: list[tuple[float, float]],
    score_cutoff: float,
    short_duration_sec: float,
    long_gap_min_sec: float,
) -> dict[str, Any]:
    """Summarise CTC timestamp health without trusting count coverage alone."""
    values = _lookup(
        record,
        (
            "character_timestamps",
            "word_timestamps",
            "alignment_timestamps",
            "aligned_timestamps",
        ),
    )
    scores: list[float] = []
    short_count = 0
    if isinstance(values, list):
        for value in values:
            if not isinstance(value, dict):
                continue
            start = finite_float(value.get("start_sec", value.get("start", value.get("begin"))))
            end = finite_float(value.get("end_sec", value.get("end", value.get("stop"))))
            if start is None or end is None or end < start:
                continue
            if end - start < short_duration_sec:
                short_count += 1
            score = finite_float(value.get("score"))
            if score is not None:
                scores.append(score)

    long_gaps: list[float] = []
    for (_, previous_end), (next_start, _) in zip(timestamps, timestamps[1:]):
        gap = next_start - previous_end
        if gap >= long_gap_min_sec:
            long_gaps.append(gap)

    count = len(timestamps)
    span = timestamps[-1][1] - timestamps[0][0] if count >= 2 else 0.0
    return {
        "ctc_scores_available": len(scores) == count and count > 0,
        "ctc_score_p50": _quantile(scores, 0.50) if scores else None,
        "ctc_low_score_count": sum(score < score_cutoff for score in scores),
        "ctc_low_score_ratio": (sum(score < score_cutoff for score in scores) / len(scores)) if scores else None,
        "short_character_count": short_count,
        "short_character_ratio": short_count / count if count else None,
        "long_gap_count": len(long_gaps),
        "long_gap_duration_sec": sum(long_gaps),
        "long_gap_ratio": sum(long_gaps) / span if span > 0 else None,
    }


def _event_count(text: Any, event: str) -> int:
    return sum(
        value.strip().lower() == event
        for value in re.findall(r"\[([^\]]+)\]", str(text or ""))
    )


def _parse_tag_prefixes(value: str) -> tuple[str, ...]:
    return tuple(part.strip().lower() for part in str(value or "").split(",") if part.strip())


def _matches_slow_proxy(text: Any, prefixes: tuple[str, ...]) -> tuple[bool, tuple[str, ...]]:
    tags = tuple(emotion_tags(text))
    if not prefixes or len(tags) != 1:
        return False, tags
    return any(tags[0] == prefix or tags[0].startswith(prefix) for prefix in prefixes), tags


def _resolve_speed_profile(args: argparse.Namespace) -> None:
    """Fill strict speed-pilot defaults without changing legacy invocations."""
    if args.speed_rate_metric is None:
        args.speed_rate_metric = "full_audio_vad_cps" if args.speed_clean_profile else "pause_excluded_cps"
    args.speed_slow_proxy_prefixes = _parse_tag_prefixes(args.speed_slow_proxy_emotions)
    if not args.speed_clean_profile:
        return
    defaults: dict[str, Any] = {
        "min_speed_spoken_units": 20,
        "max_speed_spoken_units": 80,
        "max_speed_breath_events": 2,
        "max_speed_hold_events": 0,
        "min_speed_ctc_score": 0.50,
        "max_speed_low_ctc_score_ratio": 0.10,
        "min_speed_character_duration_sec": 0.035,
        "max_speed_short_character_ratio": 0.10,
        "min_speed_vad_alignment_coverage": 0.85,
        "max_speed_long_gap_ratio": 0.15,
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)


def _status_is_successful(record: dict[str, Any]) -> bool:
    status = _lookup(record, FIELD_ALIASES["status"])
    alignment_status = _lookup(record, FIELD_ALIASES["alignment_status"])
    if status is not None and str(status).strip().lower() not in SUCCESS_STATUSES:
        return False
    if alignment_status is not None and str(alignment_status).strip().lower() not in SUCCESS_STATUSES:
        return False
    # Older manifests do not carry a status field. Their CER, coverage, and
    # timestamp requirements are still checked by ``_alignment_metrics``.
    return True


def _contains_existing_control_tag(text: Any) -> bool:
    return CONTROL_TAG_RE.search(str(text or "")) is not None


def _synthetic_control_reason(record: dict[str, Any]) -> str | None:
    # Explicit prior augmentation metadata is more reliable than path-name guesses.
    for field in ("speed_tempo_factor", "speed_original_audio", "volume_output_lufs", "actual_gain_db"):
        if record.get(field) is not None:
            return "generated_control_provenance"
    for field in (
        "speed_source",
        "volume_source",
        "volume_label_source",
        "augmentation",
        "generated_by",
        "control_source",
    ):
        value = str(record.get(field) or "").strip().lower()
        if any(token in value for token in ("atempo", "ffmpeg", "gain", "augment", "synthetic", "loud_pair")):
            return "generated_control_provenance"
    return None


def _is_whisper(record: dict[str, Any]) -> bool:
    text = str(record.get("text") or "").lower()
    values = " ".join(
        str(record.get(field) or "")
        for field in ("audio", "audio_path", "audio_path_raw", "style", "voice_style")
    ).lower()
    return "whisper" in event_tags(text) or "whisper" in values or "悄悄话" in values


def _style_rejection(record: dict[str, Any], args: argparse.Namespace) -> str | None:
    if _is_whisper(record):
        return "whisper_style"
    path = " ".join(
        str(record.get(field) or "") for field in ("audio", "audio_path", "audio_path_raw")
    ).lower()
    if "recitation" in path or "panting" in path or "朗诵" in path or "气喘" in path:
        return "speed_style_excluded"
    allow_paralinguistic = bool(getattr(args, "allow_speed_paralinguistic", False))
    if is_speed_excluded(record) and not allow_paralinguistic:
        return "speed_style_excluded"
    if allow_paralinguistic:
        return None
    ratio = _number(record, "paralinguistic_ratio")
    if ratio is None:
        return "missing_paralinguistic_ratio"
    if not 0.0 <= ratio <= 1.0:
        return "invalid_paralinguistic_ratio"
    if ratio > args.max_paralinguistic_ratio:
        return "long_paralinguistic_ratio"
    return None


def _speed_eligibility_rejection(
    record: dict[str, Any],
    alignment: dict[str, Any],
    args: argparse.Namespace,
    is_slow_proxy: bool = False,
) -> str | None:
    """Return the speed-specific gate that does not apply to effort metrics."""
    clean_text = clean_spoken_text(record.get("clean_text") or record.get("text"))
    if not args.include_non_cjk_speed and not CJK_RE.search(clean_text):
        return "non_cjk_speed_control"
    if alignment["alignment_coverage"] < args.min_speed_alignment_coverage:
        return "speed_alignment_coverage_below_threshold"

    units = int(alignment["spoken_units"])
    if args.min_speed_spoken_units is not None and units < args.min_speed_spoken_units:
        return "speed_too_few_spoken_units"
    if args.max_speed_spoken_units is not None and units > args.max_speed_spoken_units:
        return "speed_too_many_spoken_units"

    quality = alignment["timestamp_quality"]
    if args.min_speed_ctc_score is not None:
        if not quality["ctc_scores_available"]:
            return "speed_missing_ctc_scores"
        if quality["ctc_low_score_ratio"] > args.max_speed_low_ctc_score_ratio:
            return "speed_ctc_low_score_ratio_too_high"
    if (
        args.max_speed_short_character_ratio is not None
        and quality["short_character_ratio"] is not None
        and quality["short_character_ratio"] > args.max_speed_short_character_ratio
    ):
        return "speed_short_character_ratio_too_high"
    if (
        args.max_speed_long_gap_ratio is not None
        and quality["long_gap_ratio"] is not None
        and quality["long_gap_ratio"] > args.max_speed_long_gap_ratio
    ):
        return "speed_long_gap_ratio_too_high"
    if args.min_speed_vad_alignment_coverage is not None:
        coverage = alignment["alignment_vad_coverage"]
        if coverage is None:
            return "speed_missing_full_audio_vad_coverage"
        if coverage < args.min_speed_vad_alignment_coverage:
            return "speed_vad_alignment_coverage_too_low"

    if not (is_slow_proxy and args.speed_slow_proxy_allow_extra_events):
        text = str(record.get("text") or "")
        breath_count = _event_count(text, "breath")
        hold_count = _event_count(text, "hold")
        if (
            args.max_speed_breath_events is not None
            and breath_count > args.max_speed_breath_events
        ):
            return "speed_too_many_breath_events"
        if args.max_speed_hold_events is not None and hold_count > args.max_speed_hold_events:
            return "speed_too_many_hold_events"
    return None


def _alignment_metrics(record: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any] | None, str | None]:
    if not _status_is_successful(record):
        return None, "alignment_not_successful"

    cer = _number(record, "asr_cer")
    if cer is None:
        return None, "missing_asr_cer"
    if cer < 0 or cer > args.max_asr_cer:
        return None, "asr_cer_out_of_range"

    coverage = _number(record, "coverage")
    if coverage is None:
        return None, "missing_alignment_coverage"
    if coverage < args.min_alignment_coverage or coverage > 1.001:
        return None, "alignment_coverage_out_of_range"

    text = str(record.get("text") or "")
    clean_text = str(record.get("clean_text") or clean_spoken_text(text))
    units = count_spoken_units(clean_text)
    if units < args.min_spoken_units:
        return None, "too_few_spoken_units"

    timestamps = _timestamp_entries(record)
    if not timestamps:
        return None, "missing_character_timestamps"
    timestamp_quality = _timestamp_quality_metrics(
        record,
        timestamps,
        args.min_speed_ctc_score if args.min_speed_ctc_score is not None else 0.50,
        (
            args.min_speed_character_duration_sec
            if args.min_speed_character_duration_sec is not None
            else 0.035
        ),
        args.speed_long_gap_min_sec,
    )
    start = _number(record, "speech_start")
    end = _number(record, "speech_end")
    if start is None:
        start = timestamps[0][0]
    if end is None:
        end = timestamps[-1][1]
    if start < 0 or end <= start:
        return None, "invalid_speech_boundaries"
    span = end - start
    declared_span = _number(record, "speech_span")
    if declared_span is not None and abs(declared_span - span) > max(0.05, span * 0.05):
        return None, "inconsistent_speech_span"
    if not args.min_speech_span_sec <= span <= args.max_speech_span_sec:
        return None, "speech_span_out_of_range"

    active_duration = _number(record, "speech_vad_active_duration")
    if active_duration is None:
        active_duration = _number(record, "active_duration")
    if active_duration is None or active_duration <= 0:
        return None, "missing_vad_active_duration"
    if active_duration > span * 1.05:
        return None, "invalid_vad_active_duration"
    active_ratio = _number(record, "active_ratio")
    derived_ratio = active_duration / span
    if active_ratio is None:
        active_ratio = derived_ratio
    if not 0 < active_ratio <= 1.001:
        return None, "invalid_vad_active_ratio"
    # Prefer the duration-derived ratio when a manifest's global VAD ratio was
    # supplied alongside a within-speech active duration.
    active_ratio = min(1.0, derived_ratio)
    if active_ratio < args.min_vad_active_ratio:
        return None, "vad_active_ratio_too_low"

    pause_ratio = _number(record, "pause_ratio")
    if pause_ratio is None:
        pause_ratio = max(0.0, 1.0 - active_ratio)
    if not 0 <= pause_ratio <= 1.001:
        return None, "invalid_pause_ratio"
    if pause_ratio > args.max_pause_ratio:
        return None, "pause_ratio_too_high"

    pause_excluded_cps = _number(record, "pause_excluded_cps")
    pause_metric_name = "pause_excluded_cps"
    if pause_excluded_cps is None:
        if args.speed_rate_metric == "pause_excluded_cps" and not args.allow_legacy_speed_fallback:
            return None, "missing_pause_excluded_cps"
        if args.allow_legacy_speed_fallback:
            # Old manifests may only contain a full aligned-span rate. This is
            # an explicit compatibility path for legacy speed mode only.
            pause_excluded_cps = _number(record, "legacy_speech_cps")
            if pause_excluded_cps is None:
                pause_excluded_cps = units / span
            pause_metric_name = "speech_cps_legacy_fallback"
    if pause_excluded_cps is not None and pause_excluded_cps <= 0:
        return None, "invalid_pause_excluded_cps"

    global_vad_active_duration = _number(record, "global_vad_active_duration")
    if global_vad_active_duration is not None and global_vad_active_duration <= 0:
        return None, "invalid_full_audio_vad_active_duration"
    audio_duration = _number(record, "audio_duration")
    if (
        global_vad_active_duration is not None
        and audio_duration is not None
        and global_vad_active_duration > audio_duration * 1.05
    ):
        return None, "full_audio_vad_duration_exceeds_audio"

    speech_span_cps = units / span
    full_audio_vad_cps = (
        units / global_vad_active_duration if global_vad_active_duration is not None else None
    )
    speed_rates = {
        "pause_excluded_cps": pause_excluded_cps,
        "speech_span_cps": speech_span_cps,
        "full_audio_vad_cps": full_audio_vad_cps,
    }
    speed_rate_cps = speed_rates[args.speed_rate_metric]
    if speed_rate_cps is None:
        return None, f"missing_{args.speed_rate_metric}"
    speed_rate_metric = pause_metric_name if args.speed_rate_metric == "pause_excluded_cps" else args.speed_rate_metric
    alignment_vad_coverage = (
        active_duration / global_vad_active_duration
        if global_vad_active_duration is not None
        else None
    )

    return {
        "asr_cer": cer,
        "alignment_coverage": coverage,
        "spoken_units": units,
        "timestamp_count": len(timestamps),
        "speech_start_sec": start,
        "speech_end_sec": end,
        "speech_span_sec": span,
        "vad_active_duration_sec": active_duration,
        "full_audio_vad_active_duration_sec": global_vad_active_duration,
        "audio_duration_sec": audio_duration,
        "alignment_vad_coverage": alignment_vad_coverage,
        "vad_active_ratio": active_ratio,
        "pause_ratio": pause_ratio,
        "speech_cps": speech_span_cps,
        "articulation_cps": units / active_duration,
        "full_audio_vad_cps": full_audio_vad_cps,
        "pause_excluded_cps": pause_excluded_cps,
        "pause_excluded_duration_sec": _number(record, "pause_excluded_duration"),
        "char_pause_count": _number(record, "char_pause_count"),
        "char_pause_duration_sec": _number(record, "char_pause_duration"),
        "char_pause_ratio": _number(record, "char_pause_ratio"),
        "timestamp_quality": timestamp_quality,
        "speed_rate_cps": speed_rate_cps,
        "speed_rate_metric": speed_rate_metric,
    }, None


def _effort_metrics(record: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any] | None, str | None]:
    values = {
        "active_rms_dbfs_p25": _number(record, "rms_p25"),
        "active_rms_dbfs_p50": _number(record, "rms_p50"),
        "active_lufs_p25": _number(record, "lufs_p25"),
        "active_lufs_p50": _number(record, "lufs_p50"),
        "clipping_ratio": _number(record, "clipping_ratio"),
        "dynamic_range_db": _number(record, "dynamic_range"),
        "noise_floor_dbfs": _number(record, "noise_floor"),
        "snr_db": _number(record, "snr"),
    }
    if values["active_lufs_p25"] is None or values["active_lufs_p50"] is None:
        integrated = _number(record, "integrated_lufs")
        if not args.allow_integrated_lufs_fallback or integrated is None:
            return None, "missing_active_lufs_metrics"
        values["active_lufs_p25"] = integrated
        values["active_lufs_p50"] = integrated
        values["lufs_metric_source"] = "integrated_lufs_fallback"
    else:
        values["lufs_metric_source"] = "active_short_term"

    required = (
        "active_rms_dbfs_p25",
        "active_rms_dbfs_p50",
        "active_lufs_p25",
        "active_lufs_p50",
        "clipping_ratio",
        "dynamic_range_db",
    )
    if any(values[name] is None for name in required):
        return None, "missing_active_effort_metrics"
    if not 0 <= float(values["clipping_ratio"]) <= args.max_clipping_ratio:
        return None, "clipping_ratio_out_of_range"
    if float(values["dynamic_range_db"]) < args.min_dynamic_range_db:
        return None, "dynamic_range_too_low"
    noise_floor = values["noise_floor_dbfs"]
    if args.max_noise_floor_dbfs < 0 and noise_floor is not None and noise_floor > args.max_noise_floor_dbfs:
        return None, "noise_floor_too_high"
    snr = values["snr_db"]
    if args.min_snr_db > 0 and snr is not None and snr < args.min_snr_db:
        return None, "snr_too_low"
    return values, None


def _effort_thresholds(
    items: list[dict[str, Any]], args: argparse.Namespace
) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[item["recording_group"]].append(item)

    levels = (
        args.effort_soft_quantile,
        args.effort_normal_low_quantile,
        args.effort_normal_high_quantile,
        args.effort_strong_quantile,
    )
    field_names = (
        "active_rms_dbfs_p25",
        "active_rms_dbfs_p50",
        "active_lufs_p25",
        "active_lufs_p50",
    )
    thresholds: dict[str, dict[str, Any]] = {}
    rejections: Counter[str] = Counter()
    for group, group_items in sorted(groups.items()):
        if len(group_items) < args.min_group_records:
            rejections["group_too_small"] += 1
            continue
        metric_thresholds = {
            field: [_quantile((float(item["effort"][field]) for item in group_items), level) for level in levels]
            for field in field_names
        }
        rms_span = min(
            metric_thresholds["active_rms_dbfs_p25"][3] - metric_thresholds["active_rms_dbfs_p25"][0],
            metric_thresholds["active_rms_dbfs_p50"][3] - metric_thresholds["active_rms_dbfs_p50"][0],
        )
        lufs_span = min(
            metric_thresholds["active_lufs_p25"][3] - metric_thresholds["active_lufs_p25"][0],
            metric_thresholds["active_lufs_p50"][3] - metric_thresholds["active_lufs_p50"][0],
        )
        if rms_span < args.min_effort_rms_span_db:
            rejections["group_rms_span_too_small"] += 1
            continue
        if lufs_span < args.min_effort_lufs_span_db:
            rejections["group_lufs_span_too_small"] += 1
            continue
        thresholds[group] = {
            "eligible_records": len(group_items),
            "tier_strategy": "extreme-middle",
            "quantile_levels": list(levels),
            "metrics": {name: [float(value) for value in values] for name, values in metric_thresholds.items()},
            "rms_soft_to_strong_span_db": rms_span,
            "lufs_soft_to_strong_span_db": lufs_span,
        }
    return thresholds, rejections


def _classify_effort(item: dict[str, Any], threshold: dict[str, Any]) -> str | None:
    metrics = item["effort"]
    fields = tuple(threshold["metrics"])
    if all(float(metrics[field]) <= threshold["metrics"][field][0] for field in fields):
        return "soft"
    if all(float(metrics[field]) >= threshold["metrics"][field][3] for field in fields):
        return "strong"
    if all(
        threshold["metrics"][field][1] <= float(metrics[field]) <= threshold["metrics"][field][2]
        for field in fields
    ):
        return "normal"
    return None


def _classify_speed(
    cps: float,
    slow_boundary: float,
    fast_boundary: float,
    strategy: str,
    normal_low_boundary: float | None = None,
    normal_high_boundary: float | None = None,
) -> str | None:
    if cps <= slow_boundary:
        return "slow"
    if cps >= fast_boundary:
        return "fast"
    if strategy == "extreme-middle":
        if normal_low_boundary is None or normal_high_boundary is None:
            raise ValueError("extreme-middle speed classification requires normal-band boundaries")
        if normal_low_boundary <= cps <= normal_high_boundary:
            return "normal"
        return None
    return "normal"


def _unavailable_speed_selection_reason(item: dict[str, Any], control_kind: str) -> str:
    if item.get("speed_eligibility_rejection") == "non_cjk_speed_control":
        return f"{control_kind}_requires_cjk_lexical_content"
    return f"{control_kind}_requires_complete_speed_alignment"


def _duration_bin(seconds: float) -> str:
    if seconds < 3.0:
        return "short"
    if seconds < 8.0:
        return "medium"
    return "long"


def _candidate_record(
    item: dict[str, Any],
    control_kind: str,
    tier: str,
    boundaries: dict[str, float],
    group_threshold: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    tag = f"{control_kind}_{tier}"
    result = dict(item["record"])
    # Audio codes are tied to an old codec/configuration and must be regenerated
    # downstream for the selected audio, especially during BB03 adaptation.
    result.pop("audio_codes", None)
    if item["audio"]:
        result["audio"] = item["audio"]
    model_text = str(item.get("model_text") or item["original_text"])
    result["text"] = add_control_tags(model_text, tag)
    result["source"] = item["source"]
    result["recording_group"] = item["recording_group"]
    result["control_candidate_id"] = f"{item['record_id']}::{control_kind}"
    result["control_candidate_version"] = VERSION
    result["control_data_role"] = "control"
    result["control_kind"] = control_kind
    result["control_tag"] = tag
    result["control_label_source"] = (
        "reserved_emotion_proxy_speed_v1"
        if control_kind == "speed" and item.get("slow_proxy")
        else "natural_speed_and_within_group_effort_v4"
    )
    result["requires_human_review"] = control_kind == "effort"
    result["speed_label"] = item["speed_label"]
    result["effort_label"] = item["effort_label"]
    result["speed_rate_cps"] = item["alignment"]["speed_rate_cps"]
    result["speed_rate_metric"] = item["alignment"]["speed_rate_metric"]
    result["control_metrics"] = {
        "alignment": item["alignment"],
        "active_loudness": item["effort"],
    }
    result["control_selection"] = {
        "source": item["source"],
        "record_id": item["record_id"],
        "recording_group": item["recording_group"],
        "original_text": item["original_text"],
        "model_text_before_control_tag": model_text,
        "speed_label_source": item.get("speed_label_source", "rate_quantile"),
        "slow_emotion_proxy": bool(item.get("slow_proxy")),
        "removed_model_emotion_tags": list(item.get("removed_model_emotion_tags", ())),
        "emotion_tags": emotion_tags(item["original_text"]),
        "emotion_family": emotion_family(item["original_text"]),
        "stratum": {
            "source": item["source"],
            "emotion_family": emotion_family(item["original_text"]),
            "speed_label": item["speed_label"],
            "effort_label": item["effort_label"],
            "duration_bin": _duration_bin(item["alignment"]["speech_span_sec"]),
        },
        "speed_boundaries_cps": boundaries,
        "speed_metric": item["alignment"]["speed_rate_metric"],
        "speed_tier_strategy": args.speed_tier_strategy,
        "effort_group_threshold": group_threshold,
        "confound_filter": (
            (
                "effort_normal_required_for_speed"
                if args.require_normal_effort_for_speed
                else "effort_not_conditioned_for_speed"
            )
            if control_kind == "speed"
            else "speed_normal_required_for_effort"
        ),
        "effort_metrics_available": item["effort"] is not None,
        "natural_audio_only": True,
        "requires_human_review": control_kind == "effort",
        "requires_audio_codes_regeneration": True,
        "input_manifest": item["manifest"],
        "input_line": item["line_number"],
    }
    # Keep any original instruct_tts labels visible for a later correlation audit,
    # but never use them to choose this candidate's control label.
    audit: dict[str, Any] = {}
    for output_name, source_fields in {
        "speed_tag": ("input_speed_tag", "speed_tag"),
        "volume_tag": ("input_volume_tag", "volume_tag"),
        "emotion": ("input_emotion", "emotion"),
    }.items():
        for field in source_fields:
            if item["record"].get(field) is not None:
                audit[output_name] = item["record"][field]
                break
    if audit:
        result["control_source_tag_audit"] = audit
    return result


def _validate_args(args: argparse.Namespace) -> None:
    if args.speed_tier_strategy == "contiguous":
        if not 0 < args.speed_slow_quantile < args.speed_fast_quantile < 1:
            raise ValueError("speed quantiles must satisfy 0 < slow < fast < 1")
    elif not (
        0
        < args.speed_slow_quantile
        < args.speed_normal_low_quantile
        < args.speed_normal_high_quantile
        < args.speed_fast_quantile
        < 1
    ):
        raise ValueError(
            "extreme-middle speed quantiles must satisfy "
            "0 < slow-tail < normal-low < normal-high < fast-tail < 1"
        )
    effort_levels = (
        args.effort_soft_quantile,
        args.effort_normal_low_quantile,
        args.effort_normal_high_quantile,
        args.effort_strong_quantile,
    )
    if not 0 < effort_levels[0] < effort_levels[1] < effort_levels[2] < effort_levels[3] < 1:
        raise ValueError("effort quantiles must satisfy 0 < soft < normal-low < normal-high < strong < 1")
    if args.min_speed_calibration_records < 1:
        raise ValueError("--min-speed-calibration-records must be >= 1")
    if args.min_group_records < 5:
        raise ValueError("--min-group-records must be >= 5")
    if not 0 <= args.max_asr_cer <= 1:
        raise ValueError("--max-asr-cer must be within [0, 1]")
    if not 0 < args.min_alignment_coverage <= 1:
        raise ValueError("--min-alignment-coverage must be within (0, 1]")
    if not args.min_alignment_coverage <= args.min_speed_alignment_coverage <= 1:
        raise ValueError(
            "--min-speed-alignment-coverage must be within "
            "[--min-alignment-coverage, 1]"
        )
    if not 0 < args.min_vad_active_ratio <= 1:
        raise ValueError("--min-vad-active-ratio must be within (0, 1]")
    if not 0 <= args.max_pause_ratio <= 1:
        raise ValueError("--max-pause-ratio must be within [0, 1]")
    if not 0 <= args.max_paralinguistic_ratio <= 1:
        raise ValueError("--max-paralinguistic-ratio must be within [0, 1]")
    if args.min_speech_span_sec <= 0 or args.max_speech_span_sec <= args.min_speech_span_sec:
        raise ValueError("speech span limits are invalid")
    if args.min_speed_spoken_units is not None and args.min_speed_spoken_units < 1:
        raise ValueError("--min-speed-spoken-units must be >= 1")
    if (
        args.max_speed_spoken_units is not None
        and args.max_speed_spoken_units < 1
    ):
        raise ValueError("--max-speed-spoken-units must be >= 1")
    if (
        args.min_speed_spoken_units is not None
        and args.max_speed_spoken_units is not None
        and args.max_speed_spoken_units < args.min_speed_spoken_units
    ):
        raise ValueError("speed spoken-unit limits are invalid")
    for name in ("max_speed_breath_events", "max_speed_hold_events"):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")
    paired = (args.min_speed_ctc_score, args.max_speed_low_ctc_score_ratio)
    if (paired[0] is None) != (paired[1] is None):
        raise ValueError(
            "--min-speed-ctc-score and --max-speed-low-ctc-score-ratio must be set together"
        )
    if paired[0] is not None and not 0 <= paired[0] <= 1:
        raise ValueError("--min-speed-ctc-score must be within [0, 1]")
    if paired[1] is not None and not 0 <= paired[1] <= 1:
        raise ValueError("--max-speed-low-ctc-score-ratio must be within [0, 1]")
    if (
        args.min_speed_character_duration_sec is not None
        and args.min_speed_character_duration_sec <= 0
    ):
        raise ValueError("--min-speed-character-duration-sec must be positive")
    if (
        args.max_speed_short_character_ratio is not None
        and not 0 <= args.max_speed_short_character_ratio <= 1
    ):
        raise ValueError("--max-speed-short-character-ratio must be within [0, 1]")
    if (
        args.min_speed_vad_alignment_coverage is not None
        and not 0 <= args.min_speed_vad_alignment_coverage <= 1
    ):
        raise ValueError("--min-speed-vad-alignment-coverage must be within [0, 1]")
    if (
        args.max_speed_long_gap_ratio is not None
        and not 0 <= args.max_speed_long_gap_ratio <= 1
    ):
        raise ValueError("--max-speed-long-gap-ratio must be within [0, 1]")
    if args.speed_long_gap_min_sec <= 0:
        raise ValueError("--speed-long-gap-min-sec must be positive")
    if args.speed_slow_proxy_allow_extra_events and not args.speed_slow_proxy_prefixes:
        raise ValueError(
            "--speed-slow-proxy-allow-extra-events requires --speed-slow-proxy-emotions"
        )
    if args.emit_controls == "speed" and args.require_normal_effort_for_speed:
        raise ValueError("--require-normal-effort-for-speed cannot be used with --emit-controls speed")


def main() -> None:
    args = parse_args()
    _resolve_speed_profile(args)
    _validate_args(args)

    total_input = 0
    duplicate_records = 0
    rejections: Counter[str] = Counter()
    source_input_counts: Counter[str] = Counter()
    source_alignment_counts: Counter[str] = Counter()
    source_effort_metric_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    speed_items: list[dict[str, Any]] = []
    effort_items: list[dict[str, Any]] = []

    for manifest in args.input_jsonls:
        for line_number, record in iter_jsonl(manifest):
            total_input += 1
            source = _normalize_source(record)
            source_input_counts[source] += 1
            record_id = _record_id(record, source, manifest, line_number)
            if record_id in seen_ids:
                duplicate_records += 1
                rejections["duplicate_record_id"] += 1
                continue
            seen_ids.add(record_id)

            synthetic_reason = _synthetic_control_reason(record)
            if synthetic_reason:
                rejections[synthetic_reason] += 1
                continue
            if _contains_existing_control_tag(record.get("text")):
                rejections["prelabeled_control_text"] += 1
                continue
            style_reason = _style_rejection(record, args)
            if style_reason:
                rejections[style_reason] += 1
                continue
            alignment, alignment_reason = _alignment_metrics(record, args)
            if alignment_reason:
                rejections[alignment_reason] += 1
                continue
            audio = resolve_audio(record, args.audio_root)
            if not audio:
                rejections["missing_audio"] += 1
                continue
            source_alignment_counts[source] += 1

            item = {
                "record": record,
                "manifest": manifest,
                "line_number": line_number,
                "source": source,
                "record_id": record_id,
                "recording_group": _recording_group(record, source),
                "original_text": str(record.get("text") or ""),
                "audio": audio,
                "alignment": alignment,
                "effort": None,
                "speed_label": None,
                "speed_label_source": "rate_quantile",
                "effort_label": None,
                "speed_eligibility_rejection": None,
            }
            slow_proxy, source_emotions = _matches_slow_proxy(
                item["original_text"], args.speed_slow_proxy_prefixes
            )
            item["slow_proxy"] = slow_proxy
            item["source_emotions"] = source_emotions
            item["removed_model_emotion_tags"] = source_emotions if slow_proxy else ()
            item["model_text"] = (
                strip_emotion_tags(item["original_text"], source_emotions)
                if slow_proxy
                else item["original_text"]
            )
            # A partial CTC alignment can still provide useful acoustic data
            # for effort auditing, but its pause-excluded rate is not reliable
            # enough to calibrate or deconfound a speed label. Keep the two
            # gates separate so the general coverage threshold remains useful
            # for effort metric diagnostics.
            speed_eligibility_rejection = _speed_eligibility_rejection(
                record, alignment, args, slow_proxy
            )
            if speed_eligibility_rejection is None:
                speed_items.append(item)
            else:
                item["speed_eligibility_rejection"] = speed_eligibility_rejection
                rejections[speed_eligibility_rejection] += 1
            if args.emit_controls != "speed":
                effort, effort_reason = _effort_metrics(record, args)
                if effort_reason:
                    rejections[effort_reason] += 1
                    continue
                item["effort"] = effort
                effort_items.append(item)
                source_effort_metric_counts[source] += 1

    speed_calibration = [item for item in speed_items if not item["slow_proxy"]]
    if len(speed_calibration) < args.min_speed_calibration_records:
        raise ValueError(
            "not enough high-confidence rows for global speed calibration: "
            f"need {args.min_speed_calibration_records}, found {len(speed_calibration)}"
        )
    slow_boundary = _quantile(
        (item["alignment"]["speed_rate_cps"] for item in speed_calibration), args.speed_slow_quantile
    )
    fast_boundary = _quantile(
        (item["alignment"]["speed_rate_cps"] for item in speed_calibration), args.speed_fast_quantile
    )
    if not slow_boundary < fast_boundary:
        raise ValueError("global speed calibration quantiles did not yield distinct boundaries")
    speed_boundaries = {"slow_max": slow_boundary, "fast_min": fast_boundary}
    normal_low_boundary: float | None = None
    normal_high_boundary: float | None = None
    if args.speed_tier_strategy == "extreme-middle":
        normal_low_boundary = _quantile(
            (item["alignment"]["speed_rate_cps"] for item in speed_calibration),
            args.speed_normal_low_quantile,
        )
        normal_high_boundary = _quantile(
            (item["alignment"]["speed_rate_cps"] for item in speed_calibration),
            args.speed_normal_high_quantile,
        )
        if not slow_boundary < normal_low_boundary < normal_high_boundary < fast_boundary:
            raise ValueError(
                "extreme-middle speed calibration quantiles did not yield four distinct boundaries; "
                "use more diverse data or adjust the quantiles"
            )
        speed_boundaries.update(
            {
                "normal_min": normal_low_boundary,
                "normal_max": normal_high_boundary,
            }
        )
    for item in speed_items:
        if item["slow_proxy"]:
            item["speed_label"] = "slow"
            item["speed_label_source"] = "emotion_proxy"
            continue

        rate_label = _classify_speed(
            item["alignment"]["speed_rate_cps"],
            slow_boundary,
            fast_boundary,
            args.speed_tier_strategy,
            normal_low_boundary,
            normal_high_boundary,
        )
        has_proxy_emotion = any(
            tag == prefix or tag.startswith(prefix)
            for tag in item["source_emotions"]
            for prefix in args.speed_slow_proxy_prefixes
        )
        if args.speed_slow_proxy_prefixes and (has_proxy_emotion or rate_label == "slow"):
            # In proxy mode every emitted slow example must originate from the
            # requested emotion, and mixed-emotion proxy rows are not silently
            # repurposed as normal/fast controls.
            item["speed_label"] = None
            item["speed_label_source"] = "omitted_by_slow_proxy_policy"
        else:
            item["speed_label"] = rate_label

    group_thresholds: dict[str, dict[str, Any]] = {}
    group_rejections: Counter[str] = Counter()
    if args.emit_controls != "speed":
        group_thresholds, group_rejections = _effort_thresholds(effort_items, args)
        rejections.update({f"effort_{name}": count for name, count in group_rejections.items()})
        for item in effort_items:
            threshold = group_thresholds.get(item["recording_group"])
            if threshold is None:
                rejections["effort_group_without_threshold"] += 1
                continue
            label = _classify_effort(item, threshold)
            if label is None:
                rejections["effort_metric_disagreement_or_ambiguous"] += 1
                continue
            item["effort_label"] = label

    output_records: list[dict[str, Any]] = []
    output_counts: Counter[str] = Counter()
    source_output_counts: Counter[str] = Counter()
    selection_rejections: Counter[str] = Counter()
    ordered_speed_items = sorted(
        speed_items,
        key=lambda item: (
            item["source"],
            item["recording_group"],
            item["record_id"],
            item["line_number"],
        ),
    )
    if args.emit_controls != "effort":
        for item in ordered_speed_items:
            if item["speed_label"] is None:
                selection_rejections["speed_transition_band_excluded"] += 1
                continue
            if args.require_normal_effort_for_speed and item["effort_label"] != "normal":
                selection_rejections["speed_requires_normal_effort"] += 1
                continue
            candidate = _candidate_record(
                item,
                "speed",
                str(item["speed_label"]),
                speed_boundaries,
                group_thresholds.get(item["recording_group"]),
                args,
            )
            output_records.append(candidate)
            output_counts[f"speed_{item['speed_label']}"] += 1
            source_output_counts[f"{item['source']}:speed_{item['speed_label']}"] += 1

    ordered_effort_items = sorted(
        effort_items,
        key=lambda item: (
            item["source"],
            item["recording_group"],
            item["record_id"],
            item["line_number"],
        ),
    )
    if args.emit_controls != "speed":
        for item in ordered_effort_items:
            if item["effort_label"] is None:
                continue
            threshold = group_thresholds[item["recording_group"]]
            if item["speed_label"] == "normal":
                candidate = _candidate_record(
                    item, "effort", str(item["effort_label"]), speed_boundaries, threshold, args
                )
                output_records.append(candidate)
                output_counts[f"effort_{item['effort_label']}"] += 1
                source_output_counts[f"{item['source']}:effort_{item['effort_label']}"] += 1
            elif item["speed_label"] is None:
                if item.get("speed_eligibility_rejection") is None:
                    selection_rejections["effort_speed_transition_band_excluded"] += 1
                else:
                    selection_rejections[_unavailable_speed_selection_reason(item, "effort")] += 1
            else:
                selection_rejections["effort_requires_normal_speed"] += 1

    output_path = Path(args.output_jsonl)
    write_jsonl(output_path, output_records)
    report_path = Path(args.report_json or f"{args.output_jsonl}.report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    speed_counts = Counter(
        f"{item['source']}:{item['speed_label']}" for item in speed_items if item["speed_label"]
    )
    speed_metric_counts = Counter(
        str(item["alignment"]["speed_rate_metric"]) for item in speed_calibration
    )
    speed_calibration_sources = Counter(item["source"] for item in speed_calibration)
    speed_proxy_items = [item for item in speed_items if item["slow_proxy"]]
    speed_quantiles = {
        "slow": args.speed_slow_quantile,
        "fast": args.speed_fast_quantile,
    }
    if args.speed_tier_strategy == "extreme-middle":
        speed_quantiles.update(
            {
                "normal_low": args.speed_normal_low_quantile,
                "normal_high": args.speed_normal_high_quantile,
            }
        )
    effort_counts = Counter(
        f"{item['source']}:{item['effort_label']}"
        for item in effort_items
        if item["effort_label"]
    )
    report = {
        "version": VERSION,
        "input_manifests": args.input_jsonls,
        "output_jsonl": str(output_path),
        "input_records": total_input,
        "unique_records": len(seen_ids),
        "duplicate_records": duplicate_records,
        "high_confidence_alignment_records": len(speed_items),
        "effort_metric_eligible_records": len(effort_items),
        "speed_candidate_policy": {
            "require_normal_effort": args.require_normal_effort_for_speed,
            "speed_candidates_before_optional_effort_filter": len(speed_items),
            "clean_profile": args.speed_clean_profile,
            "rate_metric": args.speed_rate_metric,
            "slow_emotion_proxy_prefixes": list(args.speed_slow_proxy_prefixes),
            "slow_emotion_proxy_candidates": len(speed_proxy_items),
            "slow_emotion_proxy_allow_extra_events": args.speed_slow_proxy_allow_extra_events,
        },
        "output_records": len(output_records),
        "source_input_counts": dict(sorted(source_input_counts.items())),
        "source_alignment_counts": dict(sorted(source_alignment_counts.items())),
        "source_effort_metric_counts": dict(sorted(source_effort_metric_counts.items())),
        "rejections": dict(sorted(rejections.items())),
        "selection_rejections": dict(sorted(selection_rejections.items())),
        "speed_calibration": {
            "scope": (
                "global_high_confidence_speed_items_excluding_slow_emotion_proxy"
                if args.speed_slow_proxy_prefixes
                else "global_all_high_confidence_speed_items"
            ),
            "primary_metric": args.speed_rate_metric,
            "tier_strategy": args.speed_tier_strategy,
            "min_alignment_coverage": args.min_speed_alignment_coverage,
            "include_non_cjk_speed": args.include_non_cjk_speed,
            "metric_sources": dict(sorted(speed_metric_counts.items())),
            "source_counts": dict(sorted(speed_calibration_sources.items())),
            "records": len(speed_calibration),
            "quantiles": speed_quantiles,
            "boundaries_cps": speed_boundaries,
            "speed_rate_cps": {
                "min": min(item["alignment"]["speed_rate_cps"] for item in speed_calibration),
                "median": _quantile(
                    (item["alignment"]["speed_rate_cps"] for item in speed_calibration), 0.5
                ),
                "max": max(item["alignment"]["speed_rate_cps"] for item in speed_calibration),
            },
        },
        "speed_labels_by_source": dict(sorted(speed_counts.items())),
        "effort_labels_by_source": dict(sorted(effort_counts.items())),
        "output_labels": dict(sorted(output_counts.items())),
        "output_labels_by_source": dict(sorted(source_output_counts.items())),
        "effort_groups_with_thresholds": len(group_thresholds),
        "effort_tier_strategy": "extreme-middle",
        "effort_group_rejections": dict(sorted(group_rejections.items())),
        "effort_thresholds_by_group": group_thresholds,
        "parameters": vars(args),
        "invariants": {
            "speed_boundaries_shared_across_sources": True,
            "speed_boundaries_calibrated_from_all_high_confidence_sources": True,
            "primary_speed_metric": args.speed_rate_metric,
            "speed_controls_require_min_alignment_coverage": args.min_speed_alignment_coverage,
            "non_cjk_speed_controls_enabled": args.include_non_cjk_speed,
            "legacy_speed_fallback_enabled": args.allow_legacy_speed_fallback,
            "speed_controls_require_effort_normal": args.require_normal_effort_for_speed,
            "speed_controls_are_independent_of_effort_by_default": not args.require_normal_effort_for_speed,
            "speed_transition_bands_excluded": args.speed_tier_strategy == "extreme-middle",
            "speed_slow_proxy_is_emotion_confounded": bool(args.speed_slow_proxy_prefixes),
            "speed_paralinguistic_filter_disabled": args.allow_speed_paralinguistic,
            "effort_controls_require_speed_normal": True,
            "natural_audio_only": True,
            "atempo_or_gain_records_excluded": True,
        },
    }
    with report_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"input_records={total_input}")
    print(f"high_confidence_alignment_records={len(speed_items)}")
    print(f"effort_metric_eligible_records={len(effort_items)}")
    print(f"global_speed_calibration_records={len(speed_calibration)}")
    print(f"global_speed_metric_sources={dict(sorted(speed_metric_counts.items()))}")
    print(f"global_speed_boundaries={speed_boundaries}")
    print(f"effort_groups_with_thresholds={len(group_thresholds)}")
    print(f"output_labels={dict(sorted(output_counts.items()))}")
    print(f"output_records={len(output_records)}")
    print(f"output={output_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
