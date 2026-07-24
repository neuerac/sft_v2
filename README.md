# Qwen3-TTS 语速与发声力度控制数据管线

本目录实现新的自然录音控制数据流程，不复用 `speed_loud_v3` 的全片段 CPS、`atempo` 或 `+2 dB` 增强监督。训练文本只使用以下六个显式标签：

```text
【speed_slow】 【speed_normal】 【speed_fast】
【effort_soft】 【effort_normal】 【effort_strong】
```

`effort` 是发声力度，不是播放音量。绝对播放响度始终在生成后用 `apply_playback_gain.py` 调整，不能作为 SFT 标签或训练音频增益对。

> 当前只需要筛选、分类导出和人工复核时，先按 [DATA_CURATION.md](DATA_CURATION.md) 执行。它使用实际服务器的 `script/`、BB03 路径，并且不包含 codec 或 SFT 步骤。下面的完整训练流程仍需在训练脚本完成适配后再使用。

候选构建以清洗后的已知文本为强制对齐目标：Faster-Whisper 只用于 ASR CER 质检，WhisperX CTC 产出字级时间戳。默认语速指标是 `pause_excluded_cps`：仅扣除相邻已对齐字之间、超过阈值且被 VAD 确认静音的长间隔；`speech_cps` 与 `articulation_cps` 仅保留作诊断。所有通过质量过滤的来源共同计算一套全局分位点边界；既有 `instruct_tts` 的 `speed_tag`、`volume_tag` 仅保留在审计字段中。

## 0. 服务器准备

以下路径是示例。先替换为服务器上的实际路径，特别是学长 checkpoint、tokenizer 和 BB03 根目录。`balanced_data_pretrain.jsonl` 中已有绝对 `audio` 路径；原始 BB03 JSONL 只有相对 `audio_path`，所以后者必须传 `--audio-root`。

```bash
cd /home/ma-user/work/dataset/csh_bj/lite/Qwen3-TTS

export ROOT="$PWD"
export PIPELINE="$ROOT/Script/control_pipeline"
export WORK="$ROOT/work/control_v1"
export BALANCED="$ROOT/data/balanced_data_pretrain.jsonl"
export BB03_JSONL="$ROOT/data/BB03_51h_cleaned.jsonl"
export BB03_ROOT=/home/ma-user/work/dataset/csh_bj/BB03
export TOKENIZER=/home/ma-user/work/dataset/csh_bj/jiangziyue/pretrained/Qwen3-TTS-Tokenizer-12Hz
export INIT_MODEL=/path/to/senior_voice_clone_checkpoint

# 先试听确认该参考确实是干净中性、与任何训练目标不是同一条音频。
export BB03_REF_KEY=0822_情感陪伴-目标音色-倾听反馈-002_000004
export BB03_REF_AUDIO="$BB03_ROOT/0822/目标/单句/情感陪伴-目标音色-倾听反馈-002/000004.wav"

mkdir -p "$WORK"
python -m pip install -r "$PIPELINE/requirements.txt"
```

Python、PyTorch 和 CUDA 必须与服务器 GPU 匹配。首次下载 `large-v3` 和 WhisperX 中文 CTC 模型需要网络或预置的本地模型缓存。

先确认 tokenizer 对全局增益的敏感性。无论这个测试结果如何，绝对播放响度都留给后处理。

```bash
export GAIN_TEST_AUDIO="$BB03_ROOT/0822/目标/单句/情感陪伴-目标音色-倾听反馈-002/000004.wav"

CUDA_VISIBLE_DEVICES=0 python "$ROOT/Script/check_codec_gain_sensitivity.py" \
  --audio "$GAIN_TEST_AUDIO" \
  --tokenizer_model_path "$TOKENIZER" \
  --device cuda:0 \
  --dtype bfloat16 \
  --scales 1.0,0.5,0.25 \
  --output_dir "$WORK/codec_gain_check"
```

检查 `$WORK/codec_gain_check/report.json` 的 code 相同率及 decoded LUFS。若缩放后的 code 或 decoded LUFS 基本不变，绝对音量尤其不能作为模型监督目标。

## 1. 建立统一对齐清单

先跑小样本。`balanced` 文件开头恰好是 BB03 子集，因此在排除 BB03 的 smoke 命令中扫描前 5,000 条，以保证仍选到非 BB03 样本。不要在 smoke test 中使用 `--drop-failed`，这样可直接检查 `status`、`error`、CER 和对齐覆盖率。

```bash
CUDA_VISIBLE_DEVICES=0 python "$PIPELINE/build_alignment_manifest.py" \
  --input-jsonl "$BALANCED" \
  --output-jsonl "$WORK/alignment_balanced_smoke.jsonl" \
  --exclude-source bb03 --max-records 5000 \
  --device cuda:0 --compute-type float16 --language auto --model large-v3

CUDA_VISIBLE_DEVICES=0 python "$PIPELINE/build_alignment_manifest.py" \
  --input-jsonl "$BB03_JSONL" \
  --output-jsonl "$WORK/alignment_bb03_smoke.jsonl" \
  --audio-root "$BB03_ROOT" \
  --max-records 100 \
  --device cuda:0 --compute-type float16 --language zh --model large-v3
```

`$BALANCED` 含有中英文混合记录，因此使用 `--language auto`（也是脚本默认值）：Faster-Whisper 会逐条检测语言，检测结果会传给相应的 WhisperX CTC 对齐器。只有已知全为中文的输入（例如 BB03）才显式固定为 `--language zh`。终端应显示大多数记录为 `status=ok`，并且不存在系统性的 `missing_audio`、`align_error` 或 ASR 文本错语言问题。修正路径或模型环境后再跑全量；全量清单保留失败行作为质检证据，后续候选脚本会自行过滤。

```bash
CUDA_VISIBLE_DEVICES=0 python "$PIPELINE/build_alignment_manifest.py" \
  --input-jsonl "$BALANCED" \
  --output-jsonl "$WORK/alignment_balanced.jsonl" \
  --exclude-source bb03 \
  --device cuda:0 --compute-type float16 --language auto --model large-v3

CUDA_VISIBLE_DEVICES=0 python "$PIPELINE/build_alignment_manifest.py" \
  --input-jsonl "$BB03_JSONL" \
  --output-jsonl "$WORK/alignment_bb03.jsonl" \
  --audio-root "$BB03_ROOT" \
  --device cuda:0 --compute-type float16 --language zh --model large-v3
```

### 可选：104k balanced 多 GPU 分片

每个分片按跳过空行后的零基输入序号取模；因此无论文件中的空行如何变化，同一条有效 JSONL 记录始终归属同一分片。每个 GPU 必须写入不同的输出文件。下面例子用 4 张 GPU 处理完整 `$BALANCED`，完成后再合并：

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

全量分片运行不要设置 `--max-records`。该参数始终限制分片选择前扫描的非空输入前缀，而不是“每个分片各处理 N 条”；例如 4 个分片同时设置 `--max-records 100` 时，每个分片约只会选中 25 条。控制台会输出 `scanned_records`、`selected_records` 和 `written_records`。某个任务中断时，用相同的 `--shard-count`、`--shard-index` 和该分片的输出路径重跑即可，然后在所有分片完成后重新合并。

每条成功清单记录包含来源、录音组、原文/清洗文本/ASR 文本、CER、字级时间戳、首末发声边界、`speech_cps`、`articulation_cps`、`pause_excluded_cps` 及其长停顿审计字段、活动语音 RMS/LUFS、动态范围、噪声和削波指标。候选构建默认拒绝缺少 `pause_excluded_cps` 的行；只有显式传入 `--allow-legacy-speed-fallback` 才会退回旧的全跨度 `speech_cps`，绝不退回 `articulation_cps`。不要改用 JSONL 旧的 `timestamp` 字段计算语速。

## 2. 构建自然控制候选

`balanced` 已包含 4,717 条 BB03 子集，而原始 BB03 清单有 14,966 条，且部分文本标记不同。全量运行时在 balanced 对齐步骤用 `--exclude-source bb03`，再单独对齐原始 BB03；这样不会重复计算，也会以原始 BB03 的文本作为唯一目标版本。候选构建把原始 BB03 清单放在前面，随后是已去除 BB03 的 balanced 清单；三档边界仍由两者合并后的全部高置信自然语音共同计算。

```bash
python "$PIPELINE/build_control_candidates.py" \
  --input-jsonl "$WORK/alignment_bb03.jsonl" "$WORK/alignment_balanced.jsonl" \
  --output-jsonl "$WORK/control_candidates.jsonl" \
  --report-json "$WORK/control_candidates_report.json"
```

检查报告中的以下内容后才继续：

- `speed_calibration.scope` 为 `global_all_high_confidence_speed_items`，`primary_metric` 为 `pause_excluded_cps`，并且 `source_counts` 覆盖所有通过质量过滤的来源；
- `rejections` 中的低 CER/覆盖率、朗诵、whisper、长副语言、长停顿和异常声学记录符合预期；
- `output_labels` 的六类候选数量足够；
- `effort_groups_with_thresholds` 合理，且 `effort_group_rejections` 没有提示大量录音组质量问题。

候选文本会在原始情绪/副语言文本前插入一个控制标签。每个控制行只有一个控制维度；同一自然录音可以分别作为 `speed_normal` 和 `effort_normal` 的独立训练行，但不能在同一行同时放两个控制标签。候选会删除原有 `audio_codes`，后续必须重新编码。

默认的 `speed_*` 候选只依赖完整字级对齐、CER/VAD/副语言过滤和全局 `pause_excluded_cps` 边界，**不要求**该录音组能获得 `effort_normal`。这使说话人分组较碎的数据仍能用于语速控制；对应的 `control_selection.confound_filter` 会明确写为 `effort_not_conditioned_for_speed`。`effort_*` 候选仍必须在同一录音组内有足够样本，并要求 `speed_normal`。默认不把纯非 CJK 文本与中文字符/秒混在同一速度边界；要显式纳入它们时传入 `--include-non-cjk-speed`。仅在需要复现实验中的更严格交叉去混淆筛选时，才传入 `--require-normal-effort-for-speed`。

## 3. 发声力度盲听与批准

所有 effort 行都需要人工审核。导出器按标签、情绪、速度和时长分层抽样，默认 HTML 不显示自动标签。为了把审核量和训练量对应起来，令每类导出量高于计划训练的 `--per-control`；若要保留全部 effort 候选，令 `--per-tag` 至少等于该类总数。

```bash
export EFFORT_AUDIT_PER_TAG=150

python "$PIPELINE/export_effort_audit.py" \
  --input-jsonl "$WORK/control_candidates.jsonl" \
  --output-dir "$WORK/effort_audit" \
  --per-tag "$EFFORT_AUDIT_PER_TAG" \
  --seed 42

cd "$WORK/effort_audit"
python -m http.server 8000
```

审核时同时听原始音频和活动语音 RMS 统一后的版本。统一后力度差异消失的样本是麦克风增益或距离差，应拒绝；`whisper` 不是 `effort_soft`。先基于 `review.csv` 的盲审编号作出保留/拒绝决定，再用 `labels_private.csv` 解除自动标签以复核。不要让审核者只看波形或 LUFS 数字。

创建 `$WORK/approved_effort_ids.txt`，每行放一个批准样本的 `approval_id`，例如：

```text
0822_情感陪伴-目标音色-倾听反馈-002_000005::effort
minimax_mmx_00001::effort
```

这里必须使用 `labels_private.csv` 中的 `approval_id`，不是 `review_id`、不是自动标签、也不是 BB03 以外来源不稳定的 `key`。后续 `--approved-keys-file` 就读取该文件。

## 4. 组装两阶段训练数据

组装器对六个控制标签等量采样，并在每个标签内按来源和情绪分层；报告会给出最终分布。选择 `GENERIC_PER_TAG`、`BB03_PER_TAG` 时，不能超过相应报告中的每类候选数和每类人工批准 effort 数。除非已审核全部 effort 候选，否则不要使用 `--per-control 0`。

```bash
export GENERIC_PER_TAG=100
export BB03_PER_TAG=100
```

### 阶段一：多说话人通用控制

通用阶段使用 aopeng 与 instruct_tts 的候选，混入没有控制标签的原始数据 replay。它不会带入 BB03。

```bash
python "$PIPELINE/assemble_stage_dataset.py" \
  --candidates-jsonl "$WORK/control_candidates.jsonl" \
  --replay-jsonl "$BALANCED" \
  --stage generic \
  --output-jsonl "$WORK/generic_stage.jsonl" \
  --report-json "$WORK/generic_stage_report.json" \
  --approved-keys-file "$WORK/approved_effort_ids.txt" \
  --require-approved-effort \
  --per-control "$GENERIC_PER_TAG" \
  --replay-ratio 1.0 \
  --seed 42

python "$PIPELINE/attach_group_references.py" \
  --input-jsonl "$WORK/generic_stage.jsonl" \
  --reference-manifest-jsonl "$WORK/alignment_balanced.jsonl" \
  --output-jsonl "$WORK/generic_stage_paired.jsonl" \
  --report-json "$WORK/generic_reference_report.json" \
  --require-audio-exists

python "$PIPELINE/validate_control_dataset.py" \
  --input-jsonl "$WORK/generic_stage_paired.jsonl" \
  --report-json "$WORK/generic_precodec_validation.json" \
  --require-ref-audio --require-audio-exists
```

`attach_group_references.py` 必须以零退出，并且报告中的 `self_reference_records` 必须为 0；否则不要使用它写出的文件。该脚本只接受同一 `recording_group` 的干净、非自身参考。录音组不足时应补齐配对或修正分组，不要静默退化成自引用。

### 阶段二：BB03 目标音色适配

BB03 阶段使用相同的六标签体系和 untagged BB03 replay，但所有目标都绑定同一个、非自身的中性 BB03 参考。

```bash
python "$PIPELINE/assemble_stage_dataset.py" \
  --candidates-jsonl "$WORK/control_candidates.jsonl" \
  --replay-jsonl "$BB03_JSONL" \
  --stage bb03 \
  --output-jsonl "$WORK/bb03_stage.jsonl" \
  --report-json "$WORK/bb03_stage_report.json" \
  --approved-keys-file "$WORK/approved_effort_ids.txt" \
  --require-approved-effort \
  --exclude-key "$BB03_REF_KEY" \
  --per-control "$BB03_PER_TAG" \
  --replay-ratio 1.0 \
  --seed 42

python "$PIPELINE/attach_bb03_reference.py" \
  --input-jsonl "$WORK/bb03_stage.jsonl" \
  --reference-jsonl "$BB03_JSONL" \
  --ref-key "$BB03_REF_KEY" \
  --audio-root "$BB03_ROOT" \
  --output-jsonl "$WORK/bb03_stage_paired.jsonl" \
  --report-json "$WORK/bb03_reference_report.json" \
  --require-audio-exists

python "$PIPELINE/validate_control_dataset.py" \
  --input-jsonl "$WORK/bb03_stage_paired.jsonl" \
  --report-json "$WORK/bb03_precodec_validation.json" \
  --require-ref-audio --require-audio-exists
```

`attach_bb03_reference.py` 会拒绝目标音频等于固定参考的记录。若这条参考不再适合，应选择另一条已试听的中性 BB03 音频，同时更新 `BB03_REF_KEY` 和 `BB03_REF_AUDIO`。

## 5. 重新生成 codec 并做最终校验

控制候选的文本发生变化，因此不能复用其旧 `audio_codes`。`prepare_data.py` 会保留 replay 中合法的 `[frames, 16]` code，并只编码缺失项；不要传 `--overwrite_audio_codes`。

```bash
CUDA_VISIBLE_DEVICES=0 python "$ROOT/Script/prepare_data.py" \
  --device cuda:0 \
  --tokenizer_model_path "$TOKENIZER" \
  --input_jsonl "$WORK/generic_stage_paired.jsonl" \
  --output_jsonl "$WORK/generic_train_formatted.jsonl" \
  --batch_size 32

python "$PIPELINE/validate_control_dataset.py" \
  --input-jsonl "$WORK/generic_train_formatted.jsonl" \
  --report-json "$WORK/generic_final_validation.json" \
  --require-ref-audio --require-audio-codes --require-audio-exists

CUDA_VISIBLE_DEVICES=0 python "$ROOT/Script/prepare_data.py" \
  --device cuda:0 \
  --tokenizer_model_path "$TOKENIZER" \
  --input_jsonl "$WORK/bb03_stage_paired.jsonl" \
  --output_jsonl "$WORK/bb03_train_formatted.jsonl" \
  --batch_size 32

python "$PIPELINE/validate_control_dataset.py" \
  --input-jsonl "$WORK/bb03_train_formatted.jsonl" \
  --report-json "$WORK/bb03_final_validation.json" \
  --require-ref-audio --require-audio-codes --require-audio-exists
```

两份最终 validation report 都必须是 `valid: true`。若失败，不要手改 JSONL 绕过校验；按 failure reason 回到候选、参考或 codec 步骤修复。

## 6. 两阶段 SFT

组装数据没有 `sample_weight` 字段，因此必须使用 `--sampling_mode none`，不能使用 `--sampling_mode control`。两阶段都设 `--output_model_type base`，这样才能保留 speaker encoder 和 `generate_voice_clone()` 能力。

以下学习率和轮数只是保守起点，应根据 held-out 控制测试和未标注语音质量选择 checkpoint；阶段二必须低于阶段一。

```bash
export NUM_GPUS=8
export GENERIC_LR=5e-7
export GENERIC_EPOCHS=3
export BB03_LR=1e-7
export BB03_EPOCHS=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes "$NUM_GPUS" \
  "$ROOT/Script/sft.py" \
  --init_model_path "$INIT_MODEL" \
  --output_model_path "$WORK/ckpt/generic_control" \
  --train_jsonl "$WORK/generic_train_formatted.jsonl" \
  --output_model_type base \
  --sampling_mode none \
  --batch_size 2 \
  --lr "$GENERIC_LR" \
  --num_epochs "$GENERIC_EPOCHS" \
  --param_dtype fp32 \
  --attn_implementation sdpa \
  --save_every_steps 200 --save_total_limit 5 --save_each_epoch \
  --seed 42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes "$NUM_GPUS" \
  "$ROOT/Script/sft.py" \
  --init_model_path "$WORK/ckpt/generic_control/checkpoint-final" \
  --output_model_path "$WORK/ckpt/bb03_control" \
  --train_jsonl "$WORK/bb03_train_formatted.jsonl" \
  --output_model_type base \
  --sampling_mode none \
  --batch_size 2 \
  --lr "$BB03_LR" \
  --num_epochs "$BB03_EPOCHS" \
  --param_dtype fp32 \
  --attn_implementation sdpa \
  --save_every_steps 100 --save_total_limit 5 --save_each_epoch \
  --seed 42
```

保留阶段一各 epoch checkpoint，并以同一参考、同一随机种子、同一解码参数比较。先确认无控制标签的音色、情绪和副语言没有退化，再判断控制是否成立。

## 7. 构建 held-out BB03 控制验收集

先从 BB03 训练阶段导出所有已用 key，避免评测文本泄漏。这个列表包含控制行和 replay 行。

```bash
python "$PIPELINE/build_control_eval_manifest.py" \
  --input-jsonl "$BB03_JSONL" \
  --output-jsonl "$WORK/bb03_control_eval.jsonl" \
  --report-json "$WORK/bb03_control_eval_report.json" \
  --ref-audio "$BB03_REF_AUDIO" \
  --exclude-training-jsonl "$WORK/bb03_stage.jsonl" \
  --count 50 --min-units 10 --seed 42
```

每个 held-out 文本会产生 7 个 case：无控制、三档 speed、三档 effort。用现有推理脚本逐条读取 `prompt_text` 和 `ref_audio`，并固定模型、参考音频、情绪文本、随机种子及解码参数。推理输出清单至少应包含：

```json
{"case_id":"001_speed_slow","audio":"/abs/path/001_speed_slow.wav","text":"【speed_slow】原始文本"}
```

把生成结果做相同的 ASR/CTC 对齐，以便量化时长和 CPS：

```bash
CUDA_VISIBLE_DEVICES=0 python "$PIPELINE/build_alignment_manifest.py" \
  --input-jsonl "$WORK/bb03_control_eval_rendered.jsonl" \
  --output-jsonl "$WORK/bb03_control_eval_aligned.jsonl" \
  --device cuda:0 --compute-type float16 --language zh --model large-v3
```

逐个 `sample_id` 检查 `slow > normal > fast` 的生成时长，及 `pause_excluded_cps(slow) < pause_excluded_cps(normal) < pause_excluded_cps(fast)`；同时检查 `speech_cps`，防止控制仅通过插入静音实现。对 effort 做盲听，并观察活动响度/力度趋势；它不应被当成精确的绝对 dB 控制。还要单独听未标签文本、whisper 和副语言，确认情绪、音色、清晰度没有退化。

## 8. 推理后的播放音量

模型生成结束后才调整播放级别。固定增益与目标 LUFS 二选一，工具会在写出后复测 true peak、削波和 LUFS，并默认限制在 `-1 dBFS` true peak。

```bash
python "$PIPELINE/apply_playback_gain.py" \
  --input "$WORK/generated.wav" \
  --output "$WORK/generated_plus3db.wav" \
  --gain-db 3 \
  --true-peak-dbfs -1.0 \
  --metrics-json "$WORK/generated_plus3db.metrics.json"

python "$PIPELINE/apply_playback_gain.py" \
  --input "$WORK/generated.wav" \
  --output "$WORK/generated_target_lufs.wav" \
  --target-lufs -16 \
  --true-peak-dbfs -1.0 \
  --metrics-json "$WORK/generated_target_lufs.metrics.json"
```

查看 sidecar metrics 中的 `after.true_peak_dbfs`、`after.lufs_i`、`ceiling_reduction_db` 和 clipping ratio。若 limiter 压缩明显，降低目标 LUFS 或请求增益；不要把受限后的响度差回填为新的 effort 训练标签。

## 9. 不应绕过的边界

- 不把 `atempo`、重采样变速、全局增益或 legacy V3 标签放回控制监督。
- 不用每个说话人自己的分位点定义 speed；边界由所有高置信、具有 `pause_excluded_cps` 的自然录音统一全局校准。
- 不用 ASR 转写替代 JSONL 已知文本作为强制对齐目标。
- 不把 `whisper` 当作低力度，也不因活动响度低就自动接受 `effort_soft`。
- 不跳过 effort 的原始/统一响度双版本试听。
- 不把 `playback_gain` 写入 SFT 文本或训练 JSONL。
