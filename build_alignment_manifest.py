"""Build a canonical ASR/CTC-alignment manifest for control-data selection.

The input is an existing JSONL audio manifest.  Each output row keeps enough
identity and text information to join it back to that source manifest, then
adds ASR quality-control results, forced character alignment, timing-derived
speaking-rate measurements, VAD activity, and passive loudness measurements.

Known cleaned text is the alignment target.  Faster-Whisper is used only to
transcribe the audio for CER quality control; it is never used as the text
target.  WhisperX's CTC aligner receives the known text over the full audio
span and returns character timestamps.  The script never changes, rescales,
or writes audio files.

Optional ML and audio dependencies are imported only after a usable input row
is encountered.  Missing models, packages, or audio therefore produce
diagnostic manifest rows instead of losing the rest of a long JSONL run.  Use
--drop-failed when only fully processed rows should be written.

Example (run on the Linux GPU machine)::

    python Script/control_pipeline/build_alignment_manifest.py \
        --input-jsonl data/balanced_data_pretrain.jsonl \
        --output-jsonl work/alignment_manifest.jsonl \
        --audio-root /mnt/datasets --device cuda --compute-type float16 \
        --language auto --model large-v3
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

try:  # Allow both ``python -m`` and direct script execution.
    from .common import (
        ALNUM_RE,
        CJK_RE,
        audio_value,
        character_error_rate,
        clean_spoken_text,
        count_spoken_units,
        emotion_family,
        emotion_tags,
        finite_float,
        infer_recording_group,
        infer_source,
        iter_jsonl,
        normalized_transcript,
        resolve_audio,
    )
except ImportError:  # pragma: no cover - exercised by direct CLI invocation.
    from common import (  # type: ignore
        ALNUM_RE,
        CJK_RE,
        audio_value,
        character_error_rate,
        clean_spoken_text,
        count_spoken_units,
        emotion_family,
        emotion_tags,
        finite_float,
        infer_recording_group,
        infer_source,
        iter_jsonl,
        normalized_transcript,
        resolve_audio,
    )


MANIFEST_VERSION = 4
_UNSET = object()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Force-align known cleaned JSONL text with WhisperX and write a "
            "canonical timing/loudness manifest. Audio is read only."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "faster-whisper and whisperx are loaded lazily. When either model "
            "or an audio path is unavailable, the corresponding row is marked "
            "failed unless --drop-failed is set."
        ),
    )
    parser.add_argument(
        "--input-jsonl",
        "--input_jsonl",
        dest="input_jsonl",
        required=True,
        help="Input JSONL with text plus audio/audio_path/wav_path.",
    )
    parser.add_argument(
        "--output-jsonl",
        "--output_jsonl",
        dest="output_jsonl",
        required=True,
        help="Destination JSONL manifest. This is the only file written.",
    )
    parser.add_argument(
        "--audio-root",
        "--audio_root",
        dest="audio_root",
        default=None,
        help="Prefix for relative audio paths in the input JSONL.",
    )
    parser.add_argument(
        "--exclude-source",
        action="append",
        default=[],
        metavar="SOURCE",
        help=(
            "Skip records whose inferred source matches SOURCE; may be repeated. "
            "Useful for excluding BB03 from a mixed manifest before aligning the "
            "complete raw BB03 manifest separately."
        ),
    )
    parser.add_argument(
        "--max-records",
        "--max_records",
        dest="max_records",
        type=int,
        default=None,
        help=(
            "Scan at most this many non-empty input rows before shard selection; useful "
            "for a deterministic smoke-test prefix, not a per-shard limit."
        ),
    )
    parser.add_argument(
        "--shard-count",
        "--shard_count",
        dest="shard_count",
        type=int,
        default=1,
        help="Number of deterministic non-empty-record shards (default: 1).",
    )
    parser.add_argument(
        "--shard-index",
        "--shard_index",
        dest="shard_index",
        type=int,
        default=0,
        help="Zero-based shard index to process, in [0, shard-count) (default: 0).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Whisper/Faster-Whisper device: cuda, cuda:N, cpu, or auto (default: cuda).",
    )
    parser.add_argument(
        "--compute-type",
        "--compute_type",
        dest="compute_type",
        default="auto",
        help="faster-whisper compute type, or auto (cuda=float16, cpu=int8).",
    )
    parser.add_argument(
        "--language",
        default="auto",
        help=(
            "ASR/alignment language code, or auto to detect each recording with "
            "Faster-Whisper (default: auto)."
        ),
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="faster-whisper model name or local model path (default: large-v3).",
    )
    parser.add_argument(
        "--align-model",
        "--align_model",
        dest="align_model",
        default=None,
        help="Optional WhisperX CTC alignment model name/path; default selects by language.",
    )
    parser.add_argument(
        "--drop-failed",
        "--drop_failed",
        dest="drop_failed",
        action="store_true",
        help="Write only rows whose ASR, alignment, and audio metrics all succeeded.",
    )
    parser.add_argument(
        "--vad-top-db",
        "--vad_top_db",
        dest="vad_top_db",
        type=float,
        default=35.0,
        help="Energy-VAD threshold below the 95th-percentile frame level (default: 35 dB).",
    )
    parser.add_argument(
        "--char-pause-min-sec",
        "--char_pause_min_sec",
        dest="char_pause_min_sec",
        type=float,
        default=0.30,
        help=(
            "Only examine aligned-character gaps at least this long for pause exclusion "
            "(default: 0.30 s)."
        ),
    )
    parser.add_argument(
        "--char-pause-vad-active-ratio-max",
        "--char_pause_vad_active_ratio_max",
        dest="char_pause_vad_active_ratio_max",
        type=float,
        default=0.20,
        help=(
            "A character gap is a confirmed pause only when energy-VAD is active for at "
            "most this fraction of the gap (default: 0.20)."
        ),
    )
    return parser.parse_args()


def _exception_text(exc: BaseException) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _normalise_requested_language(language: str | None) -> str | None:
    value = str(language or "").strip().lower()
    return None if value in {"", "auto", "none"} else value


def _validate_shard_settings(shard_count: int, shard_index: int) -> None:
    if shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--shard-index must be in [0, --shard-count)")


def _input_ordinal_in_shard(input_ordinal: int, shard_count: int, shard_index: int) -> bool:
    """Return whether a zero-based non-empty input ordinal belongs to one shard.

    The ordinal deliberately does not depend on physical JSONL line numbers, so
    added or removed blank lines do not move records between shards.  Callers
    validate shard settings once before using this hot-path helper.
    """
    if input_ordinal < 0:
        raise ValueError("input ordinal must be >= 0")
    return input_ordinal % shard_count == shard_index


def _record_identifier(record: dict[str, Any], line_number: int) -> str:
    for field in ("key", "id", "utt_id", "item_name"):
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return str(value)
    value, _ = audio_value(record)
    return value or f"line:{line_number}"


def _lexical_characters(clean_text: str) -> list[dict[str, str]]:
    """Return the lexical characters preserved by common.normalized_transcript.

    Chinese characters remain separate units; ASCII letters and digits are kept
    per character for timestamp coverage.  Speaking-rate counts intentionally
    use common.count_spoken_units instead, which treats a Latin word as one
    spoken unit.
    """
    items: list[dict[str, str]] = []
    for character in clean_text:
        if CJK_RE.fullmatch(character) or ALNUM_RE.fullmatch(character):
            normalized = normalized_transcript(character)
            if normalized:
                items.append({"character": character, "normalized": normalized})
    return items


def _base_row(
    record: dict[str, Any],
    line_number: int,
    input_ordinal: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    raw_audio, audio_field = audio_value(record)
    clean_text = clean_spoken_text(record.get("text", ""))
    lexical = _lexical_characters(clean_text)
    row: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "input_line": line_number,
        "input_ordinal": input_ordinal,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        # Keep the source key verbatim: downstream BB03 reference/exclusion
        # operations are intentionally keyed by this original identifier.
        "key": record.get("key"),
        "input_key": record.get("key"),
        "record_id": _record_identifier(record, line_number),
        "source": infer_source(record),
        "recording_group": infer_recording_group(record),
        "audio_field": audio_field,
        "audio_path_raw": raw_audio,
        "audio_path": resolve_audio(record, args.audio_root),
        "text": str(record.get("text") or ""),
        "clean_text": clean_text,
        "normalized_text": normalized_transcript(clean_text),
        "spoken_unit_count": count_spoken_units(clean_text),
        "emotion_tags": emotion_tags(record.get("text", "")),
        "emotion_family": emotion_family(record.get("text", "")),
        "input_speed_tag": record.get("speed_tag"),
        "input_volume_tag": record.get("volume_tag"),
        "input_emotion": record.get("emotion"),
        "asr_text": None,
        "asr_normalized_text": None,
        "asr_cer": None,
        "asr_language": None,
        "asr_status": "not_run",
        "asr_error": None,
        "alignment_language": None,
        "character_timestamps": [],
        "alignment_expected_characters": len(lexical),
        "alignment_observed_characters": 0,
        "alignment_timed_characters": 0,
        "alignment_coverage": None,
        "speech_start_sec": None,
        "speech_end_sec": None,
        "speech_span_sec": None,
        "audio_duration_sec": None,
        "vad_method": "energy_gate_30ms_10ms",
        "vad_threshold_dbfs": None,
        "vad_active_duration_sec": None,
        "vad_active_ratio": None,
        "speech_ratio": None,
        "speech_vad_active_duration_sec": None,
        "speech_vad_active_ratio": None,
        "speech_cps": None,
        "articulation_cps": None,
        "pause_ratio": None,
        # ``speech_cps`` deliberately remains the full aligned speech-span
        # rate.  The following fields are the pause-excluded alternative used
        # for speed-label selection, with enough detail to audit every removal.
        "char_pause_min_sec": args.char_pause_min_sec,
        "char_pause_vad_active_ratio_max": args.char_pause_vad_active_ratio_max,
        "char_pause_intervals": [],
        "char_pause_count": None,
        "char_pause_duration_sec": None,
        "char_pause_ratio": None,
        "pause_excluded_duration_sec": None,
        "pause_excluded_cps": None,
        "lexical_vad_active_duration_sec": None,
        "paralinguistic_active_duration_sec": None,
        "paralinguistic_ratio": None,
        "paralinguistic_method": "vad_active_outside_padded_character_windows_v1",
        "rms_dbfs": None,
        "peak_dbfs": None,
        "lufs_i": None,
        "active_rms_dbfs_p25": None,
        "active_rms_dbfs_p50": None,
        "active_lufs_p25": None,
        "active_lufs_p50": None,
        "active_lufs_method": None,
        "active_rms_span_p90_p10_db": None,
        "dynamic_range_db": None,
        "noise_floor_dbfs": None,
        "snr_db": None,
        "clipping_ratio": None,
        "audio_metrics_status": "not_run",
        "audio_metrics_error": None,
        "alignment_status": "not_run",
        "alignment_error": None,
        "status": "pending",
        "error": None,
    }
    return row


class LazyModels:
    """Cache optional models while keeping imports out of module import time."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._asr: Any = _UNSET
        self._asr_error: str | None = None
        self._whisperx: Any = _UNSET
        self._whisperx_error: str | None = None
        self._align: dict[str, tuple[Any, Any]] = {}
        self._align_error: dict[str, str] = {}
        self._device: str | None = None
        self._compute_type: str | None = None

    @property
    def device(self) -> str:
        if self._device is not None:
            return self._device
        requested = str(self.args.device or "cuda").strip().lower()
        if requested != "auto":
            self._device = requested
            return self._device
        try:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            self._device = "cpu"
        return self._device

    @property
    def compute_type(self) -> str:
        if self._compute_type is not None:
            return self._compute_type
        requested = str(self.args.compute_type or "auto").strip().lower()
        if requested != "auto":
            self._compute_type = requested
        else:
            self._compute_type = "float16" if self.device.startswith("cuda") else "int8"
        return self._compute_type

    def _faster_whisper_kwargs(self) -> dict[str, Any]:
        device = self.device
        if device.startswith("cuda:"):
            suffix = device.split(":", 1)[1]
            try:
                index = int(suffix)
            except ValueError as exc:
                raise ValueError(f"invalid CUDA device {device!r}; expected cuda or cuda:N") from exc
            return {"device": "cuda", "device_index": index, "compute_type": self.compute_type}
        return {"device": device, "compute_type": self.compute_type}

    def asr_model(self) -> Any:
        if self._asr is not _UNSET:
            return self._asr
        if self._asr_error is not None:
            raise RuntimeError(self._asr_error)
        try:
            from faster_whisper import WhisperModel

            self._asr = WhisperModel(self.args.model, **self._faster_whisper_kwargs())
            return self._asr
        except Exception as exc:
            self._asr_error = _exception_text(exc)
            raise RuntimeError(self._asr_error) from exc

    def whisperx_module(self) -> Any:
        if self._whisperx is not _UNSET:
            return self._whisperx
        if self._whisperx_error is not None:
            raise RuntimeError(self._whisperx_error)
        try:
            import whisperx

            self._whisperx = whisperx
            return whisperx
        except Exception as exc:
            self._whisperx_error = _exception_text(exc)
            raise RuntimeError(self._whisperx_error) from exc

    def align_model(self, language: str) -> tuple[Any, Any]:
        language = language or "zh"
        if language in self._align:
            return self._align[language]
        if language in self._align_error:
            raise RuntimeError(self._align_error[language])
        try:
            whisperx = self.whisperx_module()
            kwargs: dict[str, Any] = {"language_code": language, "device": self.device}
            if self.args.align_model:
                kwargs["model_name"] = self.args.align_model
            result = whisperx.load_align_model(**kwargs)
            self._align[language] = result
            return result
        except Exception as exc:
            self._align_error[language] = _exception_text(exc)
            raise RuntimeError(self._align_error[language]) from exc


def _asr_text_and_language(model: Any, audio_path: str, language: str | None) -> tuple[str, str | None]:
    kwargs: dict[str, Any] = {
        "beam_size": 5,
        "vad_filter": False,
        "condition_on_previous_text": False,
    }
    if language:
        kwargs["language"] = language
    segments, info = model.transcribe(audio_path, **kwargs)
    text_parts = []
    for segment in segments:  # Faster-Whisper returns a lazy generator.
        value = getattr(segment, "text", None)
        if value is None and isinstance(segment, dict):
            value = segment.get("text")
        if value:
            text_parts.append(str(value))
    detected = getattr(info, "language", None)
    return "".join(text_parts).strip(), str(detected).strip().lower() if detected else None


def _alignment_character_entries(alignment: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    segments = alignment.get("segments") if isinstance(alignment, dict) else None
    if not isinstance(segments, list):
        return entries
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        characters = segment.get("chars") or segment.get("characters")
        if not isinstance(characters, list):
            continue
        for item in characters:
            if not isinstance(item, dict):
                continue
            raw = item.get("char", item.get("character", ""))
            for character in str(raw or ""):
                normalized = normalized_transcript(character)
                if normalized:
                    entries.append(
                        {
                            "normalized": normalized,
                            "start_sec": finite_float(item.get("start")),
                            "end_sec": finite_float(item.get("end")),
                            "score": finite_float(item.get("score")),
                        }
                    )
    return entries


def _character_timestamp_rows(
    lexical: list[dict[str, str]],
    observed: list[dict[str, Any]],
    duration_sec: float,
) -> tuple[list[dict[str, Any]], int]:
    """Map CTC character output back to source text with exact sequence blocks.

    WhisperX may omit punctuation and may leave a few characters untimed.  A
    SequenceMatcher map avoids assuming that its returned character indexes
    always equal the indexes in cleaned source text.
    """
    expected_chars = [item["normalized"] for item in lexical]
    observed_chars = [str(item["normalized"]) for item in observed]
    matched: dict[int, int] = {}
    matcher = difflib.SequenceMatcher(a=expected_chars, b=observed_chars, autojunk=False)
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            matched[block.a + offset] = block.b + offset

    rows: list[dict[str, Any]] = []
    timed = 0
    for index, item in enumerate(lexical):
        source = observed[matched[index]] if index in matched else None
        start = source.get("start_sec") if source else None
        end = source.get("end_sec") if source else None
        start = finite_float(start)
        end = finite_float(end)
        if start is not None and end is not None:
            start = min(max(start, 0.0), duration_sec)
            end = min(max(end, 0.0), duration_sec)
            if end <= start:
                start = None
                end = None
        else:
            start = None
            end = None
        if start is not None:
            timed += 1
        rows.append(
            {
                "index": index,
                "character": item["character"],
                "normalized_character": item["normalized"],
                "start_sec": start,
                "end_sec": end,
                "score": source.get("score") if source else None,
            }
        )
    return rows, timed


def _speech_bounds(character_rows: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    starts = [row["start_sec"] for row in character_rows if row.get("start_sec") is not None]
    ends = [row["end_sec"] for row in character_rows if row.get("end_sec") is not None]
    if not starts or not ends:
        return None, None
    start = min(float(value) for value in starts)
    end = max(float(value) for value in ends)
    return (start, end) if end > start else (None, None)


def _dbfs(value: float) -> float | None:
    return 20.0 * math.log10(value) if value > 0.0 and math.isfinite(value) else None


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _interval_duration(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def _interval_overlap_duration(
    intervals: list[tuple[float, float]], lower: float | None, upper: float | None
) -> float:
    if lower is None or upper is None or upper <= lower:
        return 0.0
    return sum(max(0.0, min(end, upper) - max(start, lower)) for start, end in intervals)


def _character_pause_rate_metrics(
    character_rows: list[dict[str, Any]],
    active_intervals: list[tuple[float, float]],
    speech_start: float | None,
    speech_end: float | None,
    spoken_unit_count: int,
    min_gap_sec: float,
    max_vad_active_ratio: float,
) -> dict[str, Any]:
    """Measure long CTC character gaps that energy VAD confirms as inactive.

    The full first-to-last-character duration remains available as
    ``speech_span_sec`` / ``speech_cps``.  This helper derives a separate
    duration for speed control by subtracting only gaps that satisfy both
    criteria below:

    * their duration is at least ``min_gap_sec``;
    * active VAD overlap is no more than ``max_vad_active_ratio`` of the gap.

    A missing or non-monotonic character timestamp breaks adjacency.  That is
    intentionally conservative: a partially aligned word must not turn a
    missing alignment interval into an artificial long pause.
    """
    if not math.isfinite(min_gap_sec) or min_gap_sec <= 0.0:
        raise ValueError("min_gap_sec must be a positive finite number")
    if (
        not math.isfinite(max_vad_active_ratio)
        or max_vad_active_ratio < 0.0
        or max_vad_active_ratio > 1.0
    ):
        raise ValueError("max_vad_active_ratio must be a finite number in [0, 1]")

    unavailable: dict[str, Any] = {
        "char_pause_intervals": [],
        "char_pause_count": None,
        "char_pause_duration_sec": None,
        "char_pause_ratio": None,
        "pause_excluded_duration_sec": None,
        "pause_excluded_cps": None,
    }
    lower = finite_float(speech_start)
    upper = finite_float(speech_end)
    if lower is None or upper is None or upper <= lower:
        return unavailable
    if not isinstance(spoken_unit_count, int) or spoken_unit_count < 0:
        return unavailable

    speech_span = upper - lower
    # The production VAD intervals are already merged. Normalize here too so
    # direct callers and future VAD implementations cannot double-count an
    # overlap when calculating the active fraction of a character gap.
    normalized_active: list[tuple[float, float]] = []
    for interval in active_intervals:
        if not isinstance(interval, (tuple, list)) or len(interval) != 2:
            continue
        start = finite_float(interval[0])
        end = finite_float(interval[1])
        if start is None or end is None:
            continue
        start = max(lower, start)
        end = min(upper, end)
        if end > start:
            normalized_active.append((start, end))
    normalized_active = _merge_intervals(normalized_active)

    confirmed: list[dict[str, Any]] = []
    previous: tuple[int, float, float] | None = None
    for position, row in enumerate(character_rows):
        if not isinstance(row, dict):
            previous = None
            continue
        start = finite_float(row.get("start_sec"))
        end = finite_float(row.get("end_sec"))
        if start is None or end is None:
            previous = None
            continue
        start = max(lower, start)
        end = min(upper, end)
        if end <= start:
            previous = None
            continue

        if previous is not None:
            previous_position, previous_start, previous_end = previous
            # Out-of-order CTC output is not reliable enough to form a pause.
            if start < previous_start:
                previous = (position, start, end)
                continue
            gap_start = previous_end
            gap_end = start
            gap_duration = gap_end - gap_start
            if gap_duration >= min_gap_sec:
                active_duration = _interval_overlap_duration(
                    normalized_active, gap_start, gap_end
                )
                active_ratio = min(1.0, max(0.0, active_duration / gap_duration))
                if active_ratio <= max_vad_active_ratio:
                    confirmed.append(
                        {
                            "after_character_index": character_rows[previous_position].get("index"),
                            "before_character_index": row.get("index"),
                            "start_sec": gap_start,
                            "end_sec": gap_end,
                            "duration_sec": gap_duration,
                            "vad_active_duration_sec": active_duration,
                            "vad_active_ratio": active_ratio,
                        }
                    )

            # CTC character windows occasionally overlap. Carry the farther
            # end forward so a later gap is never measured from inside an
            # earlier character's duration.
            if start <= previous_end:
                previous = (position, previous_start, max(previous_end, end))
                continue
        previous = (position, start, end)

    merged_pauses = _merge_intervals(
        [(float(item["start_sec"]), float(item["end_sec"])) for item in confirmed]
    )
    pause_duration = min(speech_span, _interval_duration(merged_pauses))
    pause_excluded_duration = max(0.0, speech_span - pause_duration)
    return {
        "char_pause_intervals": confirmed,
        "char_pause_count": len(confirmed),
        "char_pause_duration_sec": pause_duration,
        "char_pause_ratio": pause_duration / speech_span,
        "pause_excluded_duration_sec": pause_excluded_duration,
        "pause_excluded_cps": (
            spoken_unit_count / pause_excluded_duration
            if pause_excluded_duration > 0.0
            else None
        ),
    }


def _paralinguistic_active_metrics(
    character_rows: list[dict[str, Any]],
    active_intervals: list[tuple[float, float]],
    speech_start: float | None,
    speech_end: float | None,
    character_padding_sec: float = 0.05,
) -> dict[str, Any]:
    """Measure active sound not explained by aligned lexical characters.

    The text target intentionally omits markup such as ``[breath]`` and
    ``<cry>``.  Long non-lexical sounds can therefore remain VAD-active while
    falling between CTC character windows.  This metric restricts both signal
    and character windows to the first/last aligned span, pads each character
    window slightly for CTC boundary jitter, then reports the remaining active
    fraction.  It is a rejection signal, not a claim that every residual frame
    is a discrete non-speech event.
    """
    unavailable: dict[str, Any] = {
        "lexical_vad_active_duration_sec": None,
        "paralinguistic_active_duration_sec": None,
        "paralinguistic_ratio": None,
        "paralinguistic_method": "vad_active_outside_padded_character_windows_v1",
    }
    if not math.isfinite(character_padding_sec) or character_padding_sec < 0.0:
        raise ValueError("character_padding_sec must be a finite number >= 0")
    lower = finite_float(speech_start)
    upper = finite_float(speech_end)
    if lower is None or upper is None or upper <= lower:
        return unavailable

    scoped_active: list[tuple[float, float]] = []
    for interval in active_intervals:
        if not isinstance(interval, (tuple, list)) or len(interval) != 2:
            continue
        start = finite_float(interval[0])
        end = finite_float(interval[1])
        if start is None or end is None:
            continue
        start = max(lower, start)
        end = min(upper, end)
        if end > start:
            scoped_active.append((start, end))
    scoped_active = _merge_intervals(scoped_active)
    active_duration = _interval_duration(scoped_active)
    if active_duration <= 0.0:
        return unavailable

    character_windows: list[tuple[float, float]] = []
    for row in character_rows:
        if not isinstance(row, dict):
            continue
        start = finite_float(row.get("start_sec"))
        end = finite_float(row.get("end_sec"))
        if start is None or end is None or end <= start:
            continue
        start = max(lower, start - character_padding_sec)
        end = min(upper, end + character_padding_sec)
        if end > start:
            character_windows.append((start, end))
    character_windows = _merge_intervals(character_windows)
    lexical_active = sum(
        _interval_overlap_duration(character_windows, start, end)
        for start, end in scoped_active
    )
    lexical_active = min(active_duration, max(0.0, lexical_active))
    paralinguistic_active = max(0.0, active_duration - lexical_active)
    return {
        "lexical_vad_active_duration_sec": lexical_active,
        "paralinguistic_active_duration_sec": paralinguistic_active,
        "paralinguistic_ratio": paralinguistic_active / active_duration,
        "paralinguistic_method": "vad_active_outside_padded_character_windows_v1",
    }


def _quantile(values: Any, level: float, np: Any) -> float | None:
    if len(values) == 0:
        return None
    value = float(np.quantile(values, level))
    return value if math.isfinite(value) else None


def _active_audio_for_window(
    audio: Any,
    sample_rate: int,
    active_intervals: list[tuple[float, float]],
    lower: float,
    upper: float,
    np: Any,
) -> Any:
    """Concatenate VAD-active samples in a time window without modifying audio."""
    pieces: list[Any] = []
    for start, end in active_intervals:
        clipped_start = max(lower, start)
        clipped_end = min(upper, end)
        if clipped_end <= clipped_start:
            continue
        start_index = max(0, int(math.floor(clipped_start * sample_rate)))
        end_index = min(len(audio), int(math.ceil(clipped_end * sample_rate)))
        if end_index > start_index:
            pieces.append(audio[start_index:end_index])
    if not pieces:
        return np.asarray([], dtype=audio.dtype)
    return np.concatenate(pieces)


def _windowed_lufs(
    audio: Any,
    sample_rate: int,
    active_intervals: list[tuple[float, float]],
    lower: float,
    upper: float,
    np: Any,
) -> tuple[float | None, float | None, str | None]:
    """Return P25/P50 loudness from VAD-active samples in each time window."""
    try:
        import pyloudnorm as pyln
    except Exception:
        return None, None, None
    if upper <= lower:
        return None, None, "pyloudnorm_vad_active_concatenated_windows"

    minimum = max(1, int(round(0.4 * sample_rate)))
    window = max(minimum, int(round(3.0 * sample_rate)))
    hop = max(1, int(round(1.0 * sample_rate)))
    begin = max(0, int(math.floor(lower * sample_rate)))
    finish = min(len(audio), int(math.ceil(upper * sample_rate)))
    if finish - begin < minimum:
        return None, None, "pyloudnorm_vad_active_concatenated_windows"
    starts = list(range(begin, max(begin + 1, finish - window + 1), hop))
    final_start = max(begin, finish - window)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)

    meter = pyln.Meter(sample_rate)
    values: list[float] = []
    for start in starts:
        end = min(finish, start + window)
        if end - start < minimum:
            continue
        active_audio = _active_audio_for_window(
            audio,
            sample_rate,
            active_intervals,
            start / sample_rate,
            end / sample_rate,
            np,
        )
        if len(active_audio) < minimum:
            continue
        try:
            value = float(meter.integrated_loudness(active_audio.astype("float64")))
        except (ValueError, FloatingPointError):
            continue
        if math.isfinite(value):
            values.append(value)
    array = np.asarray(values, dtype=np.float64)
    return (
        _quantile(array, 0.25, np),
        _quantile(array, 0.50, np),
        "pyloudnorm_vad_active_concatenated_windows",
    )


def _audio_metrics(
    audio: Any,
    sample_rate: int,
    vad_top_db: float,
    speech_start: float | None,
    speech_end: float | None,
) -> tuple[dict[str, Any], list[tuple[float, float]]]:
    """Compute read-only VAD and active-speech loudness metrics.

    The VAD is a conservative energy gate.  It is deliberately not reused for
    forced alignment: CTC timestamps remain the source of first/last speech
    boundaries, while the VAD measures active duration and pauses inside them.
    """
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - WhisperX normally requires numpy.
        raise RuntimeError(f"numpy is required for audio metrics: {_exception_text(exc)}") from exc

    samples = np.asarray(audio, dtype=np.float64)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if samples.ndim != 1 or not samples.size:
        raise ValueError("decoded audio is empty")
    if not np.all(np.isfinite(samples)):
        raise ValueError("decoded audio contains NaN or infinity")
    if sample_rate <= 0:
        raise ValueError(f"invalid sample rate: {sample_rate}")

    duration = float(len(samples) / sample_rate)
    frame = max(1, int(round(sample_rate * 0.030)))
    hop = max(1, int(round(sample_rate * 0.010)))
    if len(samples) <= frame:
        starts = np.asarray([0], dtype=np.int64)
        ends = np.asarray([len(samples)], dtype=np.int64)
    else:
        starts = np.arange(0, len(samples) - frame + 1, hop, dtype=np.int64)
        ends = starts + frame
        if int(ends[-1]) < len(samples):
            starts = np.append(starts, len(samples) - frame)
            ends = np.append(ends, len(samples))
    squares = np.square(samples, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(squares)))
    rms = np.sqrt((cumulative[ends] - cumulative[starts]) / np.maximum(ends - starts, 1))
    frame_dbfs = 20.0 * np.log10(np.maximum(rms, 1e-12))
    reference_dbfs = float(np.percentile(frame_dbfs, 95))
    threshold_dbfs = max(-50.0, reference_dbfs - vad_top_db)
    active_mask = frame_dbfs >= threshold_dbfs
    frame_starts_sec = starts.astype(np.float64) / sample_rate
    frame_ends_sec = ends.astype(np.float64) / sample_rate
    active_intervals = _merge_intervals(
        [
            (float(start), float(end))
            for start, end, active in zip(frame_starts_sec, frame_ends_sec, active_mask)
            if bool(active)
        ]
    )
    active_duration = _interval_duration(active_intervals)
    active_ratio = active_duration / duration if duration > 0 else None

    scope_start = 0.0 if speech_start is None else max(0.0, speech_start)
    scope_end = duration if speech_end is None else min(duration, speech_end)
    if scope_end <= scope_start:
        scope_start, scope_end = 0.0, duration
    speech_span = scope_end - scope_start
    speech_active = _interval_overlap_duration(active_intervals, scope_start, scope_end)
    speech_active_ratio = speech_active / speech_span if speech_span > 0 else None
    center = (frame_starts_sec + frame_ends_sec) / 2.0
    active_in_scope = active_mask & (center >= scope_start) & (center <= scope_end)
    active_dbfs = frame_dbfs[active_in_scope]
    if not len(active_dbfs):
        active_dbfs = frame_dbfs[active_mask]
    inactive_dbfs = frame_dbfs[~active_mask]
    if not len(inactive_dbfs):
        inactive_dbfs = frame_dbfs

    peak = float(np.max(np.abs(samples)))
    global_rms = float(np.sqrt(np.mean(squares)))
    lufs_i = None
    try:
        import pyloudnorm as pyln

        value = float(pyln.Meter(sample_rate).integrated_loudness(samples))
        lufs_i = value if math.isfinite(value) else None
    except (ImportError, ValueError, FloatingPointError):
        pass
    active_lufs_p25, active_lufs_p50, active_lufs_method = _windowed_lufs(
        samples, sample_rate, active_intervals, scope_start, scope_end, np
    )
    p10 = _quantile(active_dbfs, 0.10, np)
    p90 = _quantile(active_dbfs, 0.90, np)
    p05 = _quantile(active_dbfs, 0.05, np)
    p95 = _quantile(active_dbfs, 0.95, np)
    noise_floor = _quantile(inactive_dbfs, 0.10, np)
    active_p50 = _quantile(active_dbfs, 0.50, np)
    metrics: dict[str, Any] = {
        "audio_duration_sec": duration,
        "vad_threshold_dbfs": threshold_dbfs,
        "vad_active_duration_sec": active_duration,
        "vad_active_ratio": active_ratio,
        # ``speech_ratio`` mirrors the historic scan script naming.
        "speech_ratio": active_ratio,
        "speech_vad_active_duration_sec": speech_active,
        "speech_vad_active_ratio": speech_active_ratio,
        "rms_dbfs": _dbfs(global_rms),
        "peak_dbfs": _dbfs(peak),
        "lufs_i": lufs_i,
        "active_rms_dbfs_p25": _quantile(active_dbfs, 0.25, np),
        "active_rms_dbfs_p50": active_p50,
        "active_lufs_p25": active_lufs_p25,
        "active_lufs_p50": active_lufs_p50,
        "active_lufs_method": active_lufs_method,
        "active_rms_span_p90_p10_db": (p90 - p10) if p90 is not None and p10 is not None else None,
        "dynamic_range_db": (p95 - p05) if p95 is not None and p05 is not None else None,
        "noise_floor_dbfs": noise_floor,
        "snr_db": (active_p50 - noise_floor) if active_p50 is not None and noise_floor is not None else None,
        "clipping_ratio": float(np.mean(np.abs(samples) >= 0.999)),
    }
    return metrics, active_intervals


def _set_final_status(row: dict[str, Any]) -> None:
    errors: list[str] = []
    if row["asr_status"] != "ok":
        errors.append(f"asr: {row['asr_error'] or row['asr_status']}")
    if row["alignment_status"] != "ok":
        errors.append(f"alignment: {row['alignment_error'] or row['alignment_status']}")
    if row["audio_metrics_status"] != "ok":
        errors.append(f"metrics: {row['audio_metrics_error'] or row['audio_metrics_status']}")
    if not errors:
        row["status"] = "ok"
        row["error"] = None
    elif row["alignment_status"] == "ok":
        row["status"] = "partial"
        row["error"] = "; ".join(errors)
    else:
        row["status"] = "failed"
        row["error"] = "; ".join(errors)


def _process_record(
    record: dict[str, Any],
    line_number: int,
    input_ordinal: int,
    args: argparse.Namespace,
    models: LazyModels,
) -> dict[str, Any]:
    row = _base_row(record, line_number, input_ordinal, args)
    lexical = _lexical_characters(row["clean_text"])
    if not lexical:
        row["alignment_status"] = "invalid_text"
        row["alignment_error"] = "cleaned text has no CJK or ASCII alphanumeric characters"
        row["asr_status"] = "skipped"
        row["audio_metrics_status"] = "skipped"
        _set_final_status(row)
        return row

    audio_path = row["audio_path"]
    if not isinstance(audio_path, str) or not audio_path:
        row["alignment_status"] = "missing_audio_path"
        row["alignment_error"] = "record has no audio/audio_path/wav_path"
        row["asr_status"] = "skipped"
        row["audio_metrics_status"] = "skipped"
        _set_final_status(row)
        return row
    if not os.path.isfile(audio_path):
        row["alignment_status"] = "missing_audio"
        row["alignment_error"] = f"audio file not found: {audio_path}"
        row["asr_status"] = "skipped"
        row["audio_metrics_status"] = "skipped"
        _set_final_status(row)
        return row

    requested_language = _normalise_requested_language(args.language)
    detected_language: str | None = None
    try:
        asr_text, detected_language = _asr_text_and_language(
            models.asr_model(), audio_path, requested_language
        )
        row["asr_text"] = asr_text
        row["asr_normalized_text"] = normalized_transcript(asr_text)
        row["asr_cer"] = character_error_rate(row["normalized_text"], row["asr_normalized_text"])
        row["asr_language"] = detected_language or requested_language
        row["asr_status"] = "ok"
    except Exception as exc:
        row["asr_status"] = "error"
        row["asr_error"] = _exception_text(exc)

    alignment_language = requested_language or detected_language or "zh"
    row["alignment_language"] = alignment_language
    audio: Any = None
    sample_rate: int | None = None
    try:
        whisperx = models.whisperx_module()
        audio = whisperx.load_audio(audio_path)
        sample_rate = int(getattr(whisperx, "SAMPLE_RATE", 16000))
        if sample_rate <= 0:
            raise ValueError(f"invalid WhisperX sample rate: {sample_rate}")
        duration_sec = float(len(audio) / sample_rate)
        if duration_sec <= 0:
            raise ValueError("decoded audio is empty")
        row["audio_duration_sec"] = duration_sec
    except Exception as exc:
        row["alignment_status"] = "audio_decode_error"
        row["alignment_error"] = _exception_text(exc)
        row["audio_metrics_status"] = "error"
        row["audio_metrics_error"] = row["alignment_error"]
        _set_final_status(row)
        return row

    try:
        align_model, metadata = models.align_model(alignment_language)
        segment = {"start": 0.0, "end": row["audio_duration_sec"], "text": row["clean_text"]}
        alignment = whisperx.align(
            [segment],
            align_model,
            metadata,
            audio,
            models.device,
            return_char_alignments=True,
        )
        observed = _alignment_character_entries(alignment)
        character_rows, timed = _character_timestamp_rows(
            lexical, observed, float(row["audio_duration_sec"])
        )
        row["character_timestamps"] = character_rows
        row["alignment_observed_characters"] = len(observed)
        row["alignment_timed_characters"] = timed
        row["alignment_coverage"] = timed / len(lexical) if lexical else None
        speech_start, speech_end = _speech_bounds(character_rows)
        row["speech_start_sec"] = speech_start
        row["speech_end_sec"] = speech_end
        row["speech_span_sec"] = (speech_end - speech_start) if speech_start is not None and speech_end is not None else None
        if timed == 0:
            row["alignment_status"] = "no_timed_characters"
            row["alignment_error"] = "WhisperX returned no usable lexical character timestamps"
        else:
            row["alignment_status"] = "ok"
    except Exception as exc:
        row["alignment_status"] = "align_error"
        row["alignment_error"] = _exception_text(exc)

    try:
        assert sample_rate is not None
        metrics, active_intervals = _audio_metrics(
            audio,
            sample_rate,
            args.vad_top_db,
            row["speech_start_sec"],
            row["speech_end_sec"],
        )
        row.update(metrics)
        speech_span = finite_float(row["speech_span_sec"])
        speech_active = finite_float(row["speech_vad_active_duration_sec"])
        units = int(row["spoken_unit_count"])
        row["speech_cps"] = units / speech_span if speech_span and speech_span > 0 else None
        row["articulation_cps"] = units / speech_active if speech_active and speech_active > 0 else None
        if speech_span and speech_span > 0 and speech_active is not None:
            row["pause_ratio"] = min(1.0, max(0.0, 1.0 - speech_active / speech_span))
        row.update(
            _character_pause_rate_metrics(
                row["character_timestamps"],
                active_intervals,
                row["speech_start_sec"],
                row["speech_end_sec"],
                units,
                args.char_pause_min_sec,
                args.char_pause_vad_active_ratio_max,
            )
        )
        row.update(
            _paralinguistic_active_metrics(
                row["character_timestamps"],
                active_intervals,
                row["speech_start_sec"],
                row["speech_end_sec"],
            )
        )
        row["audio_metrics_status"] = "ok"
    except Exception as exc:
        row["audio_metrics_status"] = "error"
        row["audio_metrics_error"] = _exception_text(exc)

    _set_final_status(row)
    return row


def _same_path(left: str, right: str) -> bool:
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return os.path.abspath(left) == os.path.abspath(right)


def main() -> None:
    args = parse_args()
    if args.max_records is not None and args.max_records < 1:
        raise ValueError("--max-records must be >= 1")
    _validate_shard_settings(args.shard_count, args.shard_index)
    if not math.isfinite(args.vad_top_db) or args.vad_top_db <= 0:
        raise ValueError("--vad-top-db must be a positive finite number")
    if not math.isfinite(args.char_pause_min_sec) or args.char_pause_min_sec <= 0:
        raise ValueError("--char-pause-min-sec must be a positive finite number")
    if (
        not math.isfinite(args.char_pause_vad_active_ratio_max)
        or args.char_pause_vad_active_ratio_max < 0
        or args.char_pause_vad_active_ratio_max > 1
    ):
        raise ValueError("--char-pause-vad-active-ratio-max must be in [0, 1]")
    if _same_path(args.input_jsonl, args.output_jsonl):
        raise ValueError("--input-jsonl and --output-jsonl must be different files")

    excluded_sources = {str(value).strip().lower() for value in args.exclude_source if str(value).strip()}

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    models = LazyModels(args)
    counts: Counter[str] = Counter()
    alignment_counts: Counter[str] = Counter()
    scanned_records = 0
    excluded_source_records = 0
    selected_records = 0
    written_records = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for line_number, record in iter_jsonl(args.input_jsonl):
            if args.max_records is not None and scanned_records >= args.max_records:
                break
            input_ordinal = scanned_records
            scanned_records += 1
            if infer_source(record).lower() in excluded_sources:
                excluded_source_records += 1
                continue
            if not _input_ordinal_in_shard(input_ordinal, args.shard_count, args.shard_index):
                continue
            selected_records += 1
            try:
                row = _process_record(record, line_number, input_ordinal, args, models)
            except Exception as exc:  # Keep one malformed row from aborting a long audit run.
                row = _base_row(record, line_number, input_ordinal, args)
                row["asr_status"] = "error"
                row["asr_error"] = _exception_text(exc)
                row["alignment_status"] = "unexpected_error"
                row["alignment_error"] = _exception_text(exc)
                row["audio_metrics_status"] = "error"
                row["audio_metrics_error"] = _exception_text(exc)
                _set_final_status(row)
            counts[str(row["status"])] += 1
            alignment_counts[str(row["alignment_status"])] += 1
            if args.drop_failed and row["status"] != "ok":
                continue
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            written_records += 1

    print(f"shard_count={args.shard_count}")
    print(f"shard_index={args.shard_index}")
    print(f"scanned_records={scanned_records}")
    print(f"excluded_source_records={excluded_source_records}")
    print(f"selected_records={selected_records}")
    print(f"written_records={written_records}")
    print(f"status_counts={dict(sorted(counts.items()))}")
    print(f"alignment_status_counts={dict(sorted(alignment_counts.items()))}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
