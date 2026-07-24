"""Attach one clean, fixed BB03 reference to BB03 control records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from .common import infer_source, iter_jsonl, resolve_audio
else:  # Direct execution must prefer this directory over a site-packages ``common``.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import infer_source, iter_jsonl, resolve_audio  # type: ignore[no-redef]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach a non-self, fixed BB03 reference audio to every BB03 target.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--reference-jsonl", required=True, help="Complete BB03 JSONL containing --ref-key.")
    parser.add_argument("--ref-key", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--require-audio-exists", action="store_true")
    return parser.parse_args()


def normalized(value: str) -> str:
    return value.replace("\\", "/").rstrip("/").lower()


def main() -> None:
    args = parse_args()
    reference = None
    for _, record in iter_jsonl(args.reference_jsonl):
        if str(record.get("key") or "") == args.ref_key:
            reference = resolve_audio(record, args.audio_root)
            break
    if not reference:
        raise ValueError(f"reference key not found: {args.ref_key}")
    if args.require_audio_exists and not Path(reference).is_file():
        raise FileNotFoundError(f"reference audio does not exist: {reference}")

    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = 0
    bb03_records = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for line_number, record in iter_jsonl(args.input_jsonl):
            result = dict(record)
            if infer_source(result) == "bb03":
                audio = resolve_audio(result, args.audio_root)
                if not audio:
                    raise ValueError(f"line {line_number}: missing BB03 audio")
                if normalized(audio) == normalized(reference):
                    raise ValueError(f"line {line_number}: target audio is the fixed reference")
                if args.require_audio_exists and not Path(audio).is_file():
                    raise FileNotFoundError(f"line {line_number}: target audio does not exist: {audio}")
                result["audio"] = audio
                result["ref_audio"] = reference
                result["ref_policy"] = "bb03_fixed_neutral_reference"
                result["speaker_id"] = "BB03"
                bb03_records += 1
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            records += 1
    report = {"records": records, "bb03_records": bb03_records, "ref_key": args.ref_key, "ref_audio": reference}
    destination = Path(args.report_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
