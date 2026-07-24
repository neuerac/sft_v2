# 语速与力度数据筛选、导出和人工复核

本指南只完成数据准备，不进行 codec 编码、SFT 或推理。最终产物是：

- 对齐清单：每条音频的 ASR 质检、字级时间戳、停顿和响度指标；
- `control_candidates.jsonl`：自动筛出的六类自然录音候选；
- 分类音频库：复制或硬链接后的 `speed_*`、`effort_*` 音频；
- speed 和 effort 的人工复核清单。

## 0. 服务器路径

先将本目录部署到服务器的 `$ROOT/script/control_pipeline`。下面命令只使用你已说明的路径，不依赖训练脚本。

```bash
ROOT=/home/ma-user/work/dataset/csh_bj/lite/Qwen3-TTS
SCRIPT=$ROOT/script
PIPELINE=$SCRIPT/control_pipeline
WORK=$SCRIPT/work/speed_effort_curation_v1

BALANCED=$SCRIPT/balanced_data_pretrain.jsonl
BB03_ROOT=/home/ma-user/work/dataset/csh_bj/BB03
BB03_JSONL=$BB03_ROOT/BB03_51h_cleaned.jsonl

mkdir -p "$WORK"

test -f "$PIPELINE/build_alignment_manifest.py"
test -f "$PIPELINE/build_control_candidates.py"
test -f "$PIPELINE/export_control_library.py"
test -f "$BALANCED"
test -f "$BB03_JSONL"
test -d "$BB03_ROOT"

python -m pip install -r "$PIPELINE/requirements.txt"
```

不要运行旧的 `speed_loud_v3`。它使用全片段 CPS、变速增强和增益增强，不适合作为本轮自然控制数据。

## 1. 小样本检查

先确认服务器音频路径、Faster-Whisper 和 WhisperX 都可用。`balanced` 开头包含 BB03 子集，因此这里扫描 10,000 条并排除 BB03；检查控制台的 `selected_records` 必须大于 0。

```bash
CUDA_VISIBLE_DEVICES=0 python "$PIPELINE/build_alignment_manifest.py" \
  --input-jsonl "$BALANCED" \
  --output-jsonl "$WORK/alignment_balanced_smoke.jsonl" \
  --exclude-source bb03 \
  --max-records 10000 \
  --device cuda:0 --compute-type float16 --language auto --model large-v3

CUDA_VISIBLE_DEVICES=0 python "$PIPELINE/build_alignment_manifest.py" \
  --input-jsonl "$BB03_JSONL" \
  --output-jsonl "$WORK/alignment_bb03_smoke.jsonl" \
  --audio-root "$BB03_ROOT" \
  --max-records 100 \
  --device cuda:0 --compute-type float16 --language zh --model large-v3
```

抽查 smoke JSONL 中的 `status`、`asr_cer`、`alignment_coverage`、`character_timestamps` 和 `pause_excluded_cps`。如果出现系统性的 `missing_audio`、`align_error` 或语言检测错误，先修路径/环境，不要跑全量。

## 2. 全量字级对齐

`balanced` 中的 BB03 子集必须跳过，完整 BB03 从原始 JSONL 单独跑，避免重复音频和不同文本标记混在一起。

```bash
export SHARDS=4

for SHARD in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$SHARD" python "$PIPELINE/build_alignment_manifest.py" \
    --input-jsonl "$BALANCED" \
    --output-jsonl "$WORK/alignment_balanced.shard-${SHARD}-of-${SHARDS}.jsonl" \
    --exclude-source bb03 \
    --shard-count "$SHARDS" --shard-index "$SHARD" \
    --device cuda --compute-type float16 --language auto --model large-v3 &
done
wait

cat "$WORK"/alignment_balanced.shard-*-of-"$SHARDS".jsonl \
  > "$WORK/alignment_balanced.jsonl"
```

BB03 也可单卡运行；若显存和 GPU 空闲，按同样方式加 `--shard-count/--shard-index` 分片。

```bash
CUDA_VISIBLE_DEVICES=0 python "$PIPELINE/build_alignment_manifest.py" \
  --input-jsonl "$BB03_JSONL" \
  --output-jsonl "$WORK/alignment_bb03.jsonl" \
  --audio-root "$BB03_ROOT" \
  --device cuda:0 --compute-type float16 --language zh --model large-v3
```

此步骤默认保留失败行，方便追溯；它们会在下一步自动拒绝，不会被导出。

## 3. 自动筛出六类候选

```bash
python "$PIPELINE/build_control_candidates.py" \
  --input-jsonl "$WORK/alignment_bb03.jsonl" "$WORK/alignment_balanced.jsonl" \
  --output-jsonl "$WORK/control_candidates.jsonl" \
  --report-json "$WORK/control_candidates_report.json"
```

先阅读 `control_candidates_report.json`，至少确认：

- `speed_calibration.primary_metric` 是 `pause_excluded_cps`；
- `speed_calibration.source_counts` 包含预期的来源，没有意外的 `unknown`；
- `output_labels` 里有 `speed_slow`、`speed_normal`、`speed_fast`；
- effort 数量不足或录音组太碎是允许的，此时先只 review speed，不要强行放宽力度阈值。

## 4. 复制并按类别整理音频

导出器会创建如下目录，不修改源音频：

```text
review_library/
  audio/speed_slow/<source>/...
  audio/speed_normal/<source>/...
  audio/speed_fast/<source>/...
  audio/effort_soft/<source>/...
  audio/effort_normal/<source>/...
  audio/effort_strong/<source>/...
  manifests/library_manifest.csv
  review/speed_review.csv
  review/speed_review.html
  review/effort_review_candidates.jsonl
```

物理复制会占用额外磁盘空间；若新目录与原音频在同一文件系统，推荐 `hardlink`，文件看起来仍在新目录中，但不重复占用数据块。要坚持真实复制时用 `copy`。

```bash
python "$PIPELINE/export_control_library.py" \
  --input-jsonl "$WORK/control_candidates.jsonl" \
  --output-dir "$WORK/review_library" \
  --copy-mode hardlink \
  --speed-review-per-tag 50 \
  --effort-review-per-tag 50 \
  --seed 42
```

若硬链接跨文件系统失败，脚本会自动回退为复制，并在 `manifests/report.json` 中记录。若你明确需要真实副本，将 `--copy-mode hardlink` 改为 `--copy-mode copy`。

## 5. 人工复核

### 速度

```bash
cd "$WORK/review_library/review"
python -m http.server 8000
```

浏览器打开 `speed_review.html`。对每条填写 `speed_review.csv`：

- `review_status`: `keep` 或 `reject`；
- `review_note`: 例如 `actual pace wrong`、`bad alignment`、`long non-speech`。

速度复核重点是：标签是否符合实际吐字速度，而不是总音频时长是否更长。先看 CSV 中的 `pause_excluded_cps`、`char_pause_ratio`、CER 和覆盖率，再听音频。

### 力度

力度必须比较原始音频与统一活动响度后的版本。导出器已经从分类库中抽出分层样本；下面命令生成盲听包：

```bash
python "$PIPELINE/export_effort_audit.py" \
  --input-jsonl "$WORK/review_library/review/effort_review_candidates.jsonl" \
  --output-dir "$WORK/effort_blind_review" \
  --per-tag 50 \
  --seed 42

cd "$WORK/effort_blind_review"
python -m http.server 8001
```

打开 `index.html`，只根据听感填写 `review.csv` 的 `review_status` 和 `review_note`。统一响度后力度差异消失的样本应记为 `reject`；`whisper` 不能作为 `effort_soft`。

此阶段结束时保留以下文件即可，暂不进入训练：

- `control_candidates.jsonl`；
- `control_candidates_report.json`；
- `review_library/`；
- 更新后的 `speed_review.csv`、`effort_blind_review/review.csv`。
