# BB03-only Control Curation

This is a standalone data-curation path for the complete raw BB03 manifest. It
does not mix in aopeng or instruct_tts, does not write audio codes, and does
not start SFT.

The output directory is created only when the exporter runs:

```text
work/speed_effort_curation_v1/bb03_control_library/
  audio/speed_slow/
  audio/speed_normal/
  audio/speed_fast/
  audio/effort_soft/
  audio/effort_normal/
  audio/effort_strong/
  manifests/
```

## 1. Full BB03 alignment

Run the eight shards, then merge them into
`work/speed_effort_curation_v1/alignment_bb03.jsonl`. The alignment command
uses the local Faster-Whisper and Chinese CTC directories and writes
character timestamps, pause-excluded CPS, and active loudness metrics.

## 2. Build BB03-only candidates

Run this after the merged alignment file exists. Because this invocation has
only BB03 input, all speed boundaries are calibrated from BB03 itself.

```bash
cd /home/ma-user/work/dataset/csh_bj/lite/Qwen3-TTS/script

/home/ma-user/work/dataset/csh_bj/jiangziyue/myenv/bin/python \
  -m control_pipeline.build_control_candidates \
  --input-jsonl work/speed_effort_curation_v1/alignment_bb03.jsonl \
  --output-jsonl work/speed_effort_curation_v1/bb03_control_candidates.jsonl \
  --report-json work/speed_effort_curation_v1/bb03_control_candidates_report.json \
  --speed-tier-strategy extreme-middle \
  --speed-slow-quantile 0.20 \
  --speed-normal-low-quantile 0.40 \
  --speed-normal-high-quantile 0.60 \
  --speed-fast-quantile 0.80
```

Speed uses `pause_excluded_cps`, sorted from low to high:

- `speed_slow`: CPS at or below P20.
- `speed_normal`: CPS from P40 through P60.
- `speed_fast`: CPS at or above P80.
- P20-P40 and P60-P80 are deliberately omitted.

Effort already uses the same separated policy inside each `recording_group`:
P20 or below is `effort_soft`, P40-P60 is `effort_normal`, and P80 or above
is `effort_strong`. It is not an absolute playback-volume label. The script
also rejects recording groups without enough samples or enough active RMS/LUFS
spread.

Those effort values are the defaults, so they are intentionally omitted from
the command above. To tighten effort to 5% / middle 5% / 5%, append these
four arguments to the same build_control_candidates.py command:

    --effort-soft-quantile 0.05 \
    --effort-normal-low-quantile 0.475 \
    --effort-normal-high-quantile 0.525 \
    --effort-strong-quantile 0.95

For a first effort review, the default P20 / P40-P60 / P80 policy is usually
preferable. Effort classification requires agreement across several active
loudness metrics inside each recording group, so a nominal 5% tail can yield
far fewer usable files.

## 3. Copy BB03 audio into six folders

The exporter accepts the candidate JSONL, not the alignment JSONL. It copies
only `source=bb03` candidates and creates a CSV manifest, a JSONL provenance
manifest, an error list, and an export report.

```bash
/home/ma-user/work/dataset/csh_bj/jiangziyue/myenv/bin/python \
  -m control_pipeline.export_bb03_control_library \
  --input-jsonl work/speed_effort_curation_v1/bb03_control_candidates.jsonl \
  --output-dir work/speed_effort_curation_v1/bb03_control_library \
  --copy-mode copy \
  --per-tag 100
```

Use a new output directory name on a rerun; the exporter refuses to overwrite
an existing non-empty library. It exports at most 100 examples per tag, prefixes
the files with a stable rank, writes a sorted CSV manifest, and creates the
index.html listener at the library root.

The selection and display order are:

- speed_slow: lowest CPS first.
- speed_fast: highest CPS first.
- speed_normal: 100 closest to the normal-category CPS median, displayed by CPS.
- effort_soft and effort_strong: low/high within-recording-group relative
  active LUFS first.
- effort_normal: 100 closest to the normal-category relative-LUFS median,
  displayed by relative LUFS.

To listen, change into the exported library and run Python's HTTP server on
port 8000. Open the server address followed by /index.html in a browser.

## 4. Review effort labels

Speed samples can be listened to directly from the six folders. For effort,
create a blind review package that includes active-loudness-normalized audio:

```bash
/home/ma-user/work/dataset/csh_bj/jiangziyue/myenv/bin/python \
  control_pipeline/export_effort_audit.py \
  --input-jsonl work/speed_effort_curation_v1/bb03_control_library/manifests/bb03_control_candidates.jsonl \
  --output-dir work/speed_effort_curation_v1/bb03_effort_blind_review \
  --per-tag 100 \
  --seed 42
```

Reject a putative effort sample when its apparent difference disappears after
active-loudness normalization, or when it is whisper, clipping, noise, or an
emotion/style change rather than a natural change in vocal effort.
