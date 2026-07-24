"""Create held-out, fixed-text evaluation prompts for speed and effort controls."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

if __package__:
    from .common import (
        EFFORT_TAGS,
        SPEED_TAGS,
        add_control_tags,
        count_spoken_units,
        infer_source,
        is_speed_excluded,
        iter_jsonl,
        record_identity_variants,
    )
else:  # Direct execution must prefer this directory over a site-packages ``common``.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import (  # type: ignore[no-redef]
        EFFORT_TAGS,
        SPEED_TAGS,
        add_control_tags,
        count_spoken_units,
        infer_source,
        is_speed_excluded,
        iter_jsonl,
        record_identity_variants,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create A/B control-evaluation prompts from held-out BB03 text.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument(
        "--exclude-keys-file",
        default=None,
        help="Optional raw BB03 keys or candidate IDs that must not enter evaluation.",
    )
    parser.add_argument(
        "--exclude-training-jsonl",
        action="append",
        default=[],
        help="One or more assembled BB03 training JSONLs; their keys are excluded automatically.",
    )
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--min-units", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def key_variants(value: str) -> set[str]:
    """Map raw keys and ``bb03:<key>::<kind>`` IDs to the raw BB03 key."""
    value = value.strip()
    if not value:
        return set()
    variants = {value}
    base = value.split("::", 1)[0]
    variants.add(base)
    if base.startswith("bb03:"):
        variants.add(base.removeprefix("bb03:"))
    return variants


def excluded_keys(path: str | None, training_paths: list[str]) -> set[str]:
    excluded: set[str] = set()
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip() and not line.lstrip().startswith("#"):
                    excluded.update(key_variants(line))
    for training_path in training_paths:
        for _, record in iter_jsonl(training_path):
            for identity in record_identity_variants(record):
                excluded.update(key_variants(identity))
    return excluded


def main() -> None:
    args = parse_args()
    if args.count < 1 or args.min_units < 1:
        raise ValueError("--count and --min-units must be positive")
    excluded = excluded_keys(args.exclude_keys_file, args.exclude_training_jsonl)
    pool = []
    rejections = Counter()
    for _, record in iter_jsonl(args.input_jsonl):
        if infer_source(record) != "bb03":
            rejections["not_bb03"] += 1
            continue
        key = str(record.get("key") or "").strip()
        if not key or key in excluded:
            rejections["missing_or_excluded_key"] += 1
            continue
        text = str(record.get("text") or "").strip()
        if any(f"\u3010{tag}\u3011" in text for tag in (*SPEED_TAGS, *EFFORT_TAGS)):
            rejections["already_control_tagged"] += 1
            continue
        if count_spoken_units(text) < args.min_units or is_speed_excluded(record):
            rejections["ineligible_text"] += 1
            continue
        pool.append(record)
    if len(pool) < args.count:
        raise ValueError(f"only {len(pool)} eligible held-out BB03 rows, requested {args.count}")
    rng = random.Random(args.seed)
    selected = rng.sample(sorted(pool, key=lambda item: str(item.get("key"))), args.count)

    records = []
    variants = [("untagged", None)]
    variants += [(tag, tag) for tag in SPEED_TAGS]
    variants += [(tag, tag) for tag in EFFORT_TAGS]
    for sample_index, source in enumerate(selected, start=1):
        source_key = str(source["key"])
        base_text = str(source["text"])
        for variant, tag in variants:
            records.append({
                "case_id": f"{sample_index:03d}_{variant}",
                "sample_id": f"{sample_index:03d}",
                "source_key": source_key,
                "variant": variant,
                "control_dimension": "speed" if tag in SPEED_TAGS else "effort" if tag in EFFORT_TAGS else "none",
                "prompt_text": add_control_tags(base_text, tag) if tag else base_text,
                "base_text": base_text,
                "ref_audio": args.ref_audio,
                "expected_speed_order": list(SPEED_TAGS),
                "expected_effort_order": list(EFFORT_TAGS),
            })
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    report = {
        "input_jsonl": args.input_jsonl,
        "source_candidates": len(pool),
        "selected_texts": len(selected),
        "output_cases": len(records),
        "variants_per_text": len(variants),
        "excluded_training_keys": len(excluded),
        "exclude_training_jsonls": args.exclude_training_jsonl,
        "rejections": dict(rejections),
    }
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
