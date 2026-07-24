"""Shared, dependency-light helpers for the control-data pipeline."""

from __future__ import annotations

import json
import os
import posixpath
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Iterator


EMOTION_RE = re.compile(r"\u3010([^\u3011]+)\u3011")
SQUARE_RE = re.compile(r"\[([^\]]+)\]")
ANGLE_RE = re.compile(r"</?[^>]+>")
ANGLE_TAG_RE = re.compile(r"<\s*([A-Za-z][A-Za-z0-9_-]*)\b[^>]*>", re.IGNORECASE)
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
ALNUM_RE = re.compile(r"[A-Za-z0-9]+")
WORD_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]|[A-Za-z0-9]+")
PUNCT_OR_SPACE_RE = re.compile(r"[^\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaffA-Za-z0-9]+")
WINDOWS_ABS_RE = re.compile(r"^[A-Za-z]:[\\/]")

SPEED_TAGS = ("speed_slow", "speed_normal", "speed_fast")
EFFORT_TAGS = ("effort_soft", "effort_normal", "effort_strong")
ALL_CONTROL_TAGS = frozenset((*SPEED_TAGS, *EFFORT_TAGS))
NON_SPEECH_EVENTS = frozenset(
    {
        "breath",
        "hold",
        "clucking",
        "cough",
        "cry",
        "crying",
        "fizz",
        "gasp",
        "hush",
        "laugh",
        "laughter",
        "pant",
        "panting",
        "scream",
        "sigh",
        "sing",
        "sob",
        "sneezing",
        "whisper",
        "yawn",
    }
)


def iter_jsonl(path: str | os.PathLike[str]) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield non-empty JSON object records with useful parse errors."""
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, record


def write_jsonl(path: str | os.PathLike[str], records: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_absolute_path(value: str) -> bool:
    return value.startswith("/") or WINDOWS_ABS_RE.match(value) is not None


def audio_value(record: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return the preferred audio value and its source field."""
    for field in ("audio", "wav_path", "audio_path"):
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip(), field
    return None, None


def resolve_audio(record: dict[str, Any], audio_root: str | None = None) -> str | None:
    value, _ = audio_value(record)
    if not value:
        return None
    value = value.replace("\\", "/")
    if is_absolute_path(value):
        return posixpath.normpath(value) if value.startswith("/") else os.path.normpath(value)
    if not audio_root:
        return value
    root = str(audio_root).replace("\\", "/")
    if root.startswith("/"):
        return posixpath.normpath(posixpath.join(root, value))
    return os.path.normpath(os.path.join(root, value))


def infer_source(record: dict[str, Any]) -> str:
    # Manifests produced by this pipeline preserve an explicit source.  Trust
    # that provenance before applying path heuristics: external datasets often
    # also use fields named ``audio_path`` or ``key``.
    for field in ("source", "dataset_source", "control_source"):
        declared = str(record.get(field) or "").strip().lower()
        if "bb03" in declared:
            return "bb03"
        if "instruct" in declared:
            return "instruct_tts"
        if "aopeng" in declared or "obs_0" in declared:
            return "aopeng"

    value, field = audio_value(record)
    normalized = (value or "").replace("\\", "/").lower()
    if "aopeng_bj4_obs_0" in normalized:
        return "aopeng"
    if "/instruct_tts/" in normalized or field == "wav_path":
        return "instruct_tts"
    # Raw BB03 uses a relative ``.../目标/...`` path, whereas a generic
    # ``audio_path`` or foreign key is not enough evidence to assign it to
    # the target-voice dataset. Check source-specific path evidence only.
    padded_path = f"/{normalized.strip('/')}"
    if "/bb03/" in padded_path or "/目标/" in f"{padded_path}/":
        return "bb03"
    return "unknown"


def _path_parts(record: dict[str, Any], *, prefer_manifest_path: bool = False) -> list[str]:
    """Return normalized path components without leaking an audio-root choice.

    BB03 manifests can carry both a stable relative ``audio_path`` and an
    environment-specific absolute ``audio`` field. Grouping must use the
    former so one recording keeps the same group on local and server machines.
    """
    value: Any = record.get("audio_path") if prefer_manifest_path else None
    if not isinstance(value, str) or not value.strip():
        value, _ = audio_value(record)
    return [part for part in str(value or "").replace("\\", "/").split("/") if part]


def infer_recording_group(record: dict[str, Any]) -> str:
    """Return a conservative group used only for within-condition comparisons."""
    source = infer_source(record)
    # BB03 has no reliable speaker_id in the supplied JSONL. Its stable
    # recording/session identifier is the complete prefix before /target/.
    # Prefer the relative manifest path so it stays invariant across machines.
    parts = _path_parts(record, prefer_manifest_path=source == "bb03")
    goal = "\u76ee\u6807"
    if source == "bb03" and goal in parts:
        prefix = "/".join(parts[:parts.index(goal)])
        if prefix:
            return f"bb03:recording:{prefix}"

    explicit = str(record.get("speaker_id") or "").strip()
    if explicit:
        return f"{source}:speaker:{explicit}"

    # For Aopeng retain nearby recording-batch and speaker components instead
    # of grouping separate microphone/session recordings together.
    if goal in parts:
        goal_index = parts.index(goal)
        prefix = "/".join(parts[max(0, goal_index - 2):goal_index])
        if prefix:
            return f"{source}:recording:{prefix}"

    if source == "instruct_tts":
        item = str(record.get("item_name") or "").strip()
        gender = str(record.get("gender") or "unknown").strip().lower() or "unknown"
        if item:
            base = re.sub(r"[_-]?\d+$", "", item)
            # The same item stem occurs with male, female, and child voices
            # in the supplied manifest.  Gender is imperfect speaker metadata,
            # but it is a necessary lower bound before doing within-group
            # effort ranking or choosing a clone reference.
            return f"instruct_tts:item:{base}:gender:{gender}"
        # Gender alone is far too broad to prove speaker identity.  Isolate
        # records without an item identifier so they cannot silently obtain a
        # cross-speaker effort threshold or reference pairing.
        if parts:
            return f"instruct_tts:unresolved:{'/'.join(parts)}"
        return "instruct_tts:unresolved"

    if parts:
        return f"{source}:path:{'/'.join(parts[:3])}"
    return f"{source}:unknown"


def clean_spoken_text(text: Any) -> str:
    """Remove annotation syntax while retaining lexical text inside angle tags."""
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = EMOTION_RE.sub("", value)
    value = SQUARE_RE.sub("", value)
    value = ANGLE_RE.sub("", value)
    return value.strip()


def normalized_transcript(text: Any) -> str:
    value = clean_spoken_text(text).lower()
    return PUNCT_OR_SPACE_RE.sub("", value)


def spoken_units(text: Any) -> list[str]:
    return WORD_RE.findall(clean_spoken_text(text))


def count_spoken_units(text: Any) -> int:
    return len(spoken_units(text))


def emotion_tags(text: Any) -> list[str]:
    values: list[str] = []
    for tag in EMOTION_RE.findall(str(text or "")):
        tag = tag.strip().lower()
        if tag and tag not in ALL_CONTROL_TAGS and not tag.startswith("speed_") and not tag.startswith("effort_"):
            values.append(tag)
    return values


def emotion_family(text: Any) -> str:
    tags = emotion_tags(text)
    if not tags:
        return "no_emotion"
    families = ("neutral", "happy", "sad", "angry", "fear", "surprise", "disgust", "calm")
    found = [family for family in families if any(tag.startswith(family) for tag in tags)]
    if len(found) == 1:
        return found[0]
    return "transition" if len(tags) > 1 else "other"


def event_tags(text: Any) -> set[str]:
    value = str(text or "")
    square = {tag.strip().lower() for tag in SQUARE_RE.findall(value) if tag.strip()}
    # ``<stress>`` and related prosody markup are intentionally not events.
    # Only known non-speech tags in angle syntax are carried into the filter.
    angle = {
        tag.strip().lower()
        for tag in ANGLE_TAG_RE.findall(value)
        if tag.strip().lower() in NON_SPEECH_EVENTS
    }
    return square | angle


def is_speed_excluded(record: dict[str, Any]) -> bool:
    text = str(record.get("text") or "")
    path, _ = audio_value(record)
    lowered_path = (path or "").lower()
    events = event_tags(text)
    if "whisper" in lowered_path or "\u6084\u6084\u8bdd" in (path or ""):
        return True
    if "\u6717\u8bf5" in (path or "") or "\u6c14\u5598\u5401\u5401" in (path or ""):
        return True
    return bool(events - {"breath", "hold"})


def add_control_tags(text: Any, *tags: str) -> str:
    unknown = [tag for tag in tags if tag not in ALL_CONTROL_TAGS]
    if unknown:
        raise ValueError(f"unsupported control tags: {unknown}")
    return "".join(f"\u3010{tag}\u3011" for tag in tags) + str(text or "").strip()


def levenshtein_distance(left: str, right: str) -> int:
    """Memory-efficient edit distance for transcript quality checks."""
    if len(left) < len(right):
        left, right = right, left
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for index, left_char in enumerate(left, start=1):
        current = [index]
        for right_index, right_char in enumerate(right, start=1):
            insertions = current[right_index - 1] + 1
            deletions = previous[right_index] + 1
            substitutions = previous[right_index - 1] + (left_char != right_char)
            current.append(min(insertions, deletions, substitutions))
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float | None:
    if not reference:
        return None
    return levenshtein_distance(reference, hypothesis) / len(reference)


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in (float("inf"), float("-inf")) else None


def record_identity_variants(record: dict[str, Any]) -> set[str]:
    """Return raw and pipeline identity forms for exclusion/leakage checks."""
    values: set[str] = set()
    candidates: list[Any] = [
        record.get("key"),
        record.get("record_id"),
        record.get("control_candidate_id"),
    ]
    selection = record.get("control_selection")
    if isinstance(selection, dict):
        candidates.append(selection.get("record_id"))
    for candidate in candidates:
        value = str(candidate or "").strip()
        if not value:
            continue
        values.add(value)
        base = value.split("::", 1)[0]
        values.add(base)
        for prefix in ("bb03:", "aopeng:", "instruct_tts:"):
            if base.startswith(prefix):
                values.add(base.removeprefix(prefix))
                break
    return values
