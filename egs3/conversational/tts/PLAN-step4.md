
# Step 4: infer stage + evaluation battery

## Goal

Make the recipe evaluable end to end: an espnet3 `infer` stage that batch-generates multi-channel conversations from manifest windows (plus ground-truth and vocoder-resynthesis anchor modes), and a `measure` stage running a custom metric battery (ASR, speaker, interaction, quality) whose speaker-attributed metrics need no diarization because channel = speaker.
First run target: the assembled pretrained model (zero-init gates) on about 50 valid-split windows, as a debugging pass and pre-finetuning baseline.

Do NOT implement the mixdown-diarization comparability protocol (pyannote + cpWER/cpSIM on mixed audio); it is explicitly deferred.
Do NOT modify anything outside `egs3/conversational/tts/`; `espnet2/` and `espnet3/` are read-only (the `measure` machinery in `espnet3/systems/base/metric.py` is used as-is).
Do NOT touch the training code paths; this step must merge without retraining anything.

## Context (verified against the code, 2026-07-14)

- `run.py` currently has stages `["create_dataset", "train"]`; the libritts recipe (`egs3/libritts/tts/run.py`) shows the house pattern for adding `infer` and `measure` stages plus `--inference_config` / `--metrics_config` arguments and `conf/metrics.yaml`.
- `espnet3/systems/base/metric.py::measure` instantiates each entry of `metrics_config.metrics` (must be an `espnet3.components.metrics.base_metric.BaseMetric`), resolves test sets from `metrics_config.dataset.test` or by scanning `inference_dir`, loads the SCPs named in each metric's `inputs:` mapping from `inference_dir/<test_name>/<name>.scp`, calls the metric, and writes `inference_dir/metrics.json`.
- `BaseMetric.iter_inputs(data, *keys)` streams multiple SCPs in lockstep and asserts identical utterance ids, so every metric should iterate ONE granularity; the contract below therefore keys everything by window id with a per-window JSON carrying channel-level detail.
- `local/generate_dev.py` already implements single-window multi-channel inference: dataset load with `inference=True`, `ConversationalTextPreprocessor`, `model.cfm.sample(cond=speech, text=text, duration=total_frames, counts=[n], lens=lens, steps, cfg_strength, sway_sampling_coef, seed)`, Vocos vocoding, per-channel wav + mixdown output.
  Factor its model/vocoder/dataset loading into a shared module rather than duplicating; keep `generate_dev.py` working as the quick listening tool.
- The window manifests (`data/manifest/{train,valid,test}.jsonl`) carry per-window `turns` with absolute-second `start`/`end`, `num_active_speakers`, `channel_speech_sec`, `exchange_count`; turn timestamps exist precisely for evaluation and never become tokens.
- The windowing eligibility rule (no turn strictly contains the cut instant) lives in `dataset/preprocessing/windows.py`; reuse it for prompt-boundary snapping.
- `local/crosstalk_report.py` has interval helpers (`merge_intervals`, `subtract_intervals`, `solo_regions`) and solo-region energy measurement; import/reuse them for the generated-bleed metric.
- Delta specifics (corpus, checkpoints, pixi env) are in the memory notes; local development uses fabricated fixtures only, following the existing test suites.

## Infer stage

New `src/inference.py` (stage implementation) + `conf/inference_conversational.yaml`; wire `infer` into `run.py` mirroring libritts.

- **Window selection**: split (default `valid`), filters (`num_active_speakers == 2`, optional duration band), `num_windows` cap with a fixed seed; selection is logged and reproducible.
- **Prompt-boundary snapping**: target `prompt_sec` (default 3.0) with band `[prompt_min, prompt_max]` (default 2.0-10.0 s); choose the eligible turn boundary closest to the target inside the band; windows with no eligible boundary in the band are skipped and counted.
  The prompt region of every channel is the ground-truth audio before the boundary; the infilling mask covers the remainder; conditioning and sampling exactly as `generate_dev.py`.
- **Modes** (`mode: generate | gt | resynth`): `generate` runs the model; `gt` copies the ground-truth generated-region audio into the output layout; `resynth` round-trips ground truth through mel + Vocos.
  Anchors thereby flow through the identical measure stage with zero metric-side special-casing.
- **Output contract**, under `inference_dir/<test_name>/`:
  - `meta.scp`: `<window_id> <path to meta JSON>` - the PRIMARY input every metric iterates.
  - Per-window meta JSON: prompt boundary (seconds and frames), window duration, sample rate, per-channel relative paths (generated wav, prompt wav, ground-truth generated-region wav), per-channel reference text for the generated region (turns with `start >= boundary`, order preserved, normalized text as in the manifests), ground-truth turn spans shifted to window time, RTF (generate mode).
  - Convenience SCPs for listening/interop (not consumed by metrics): channel-level `wav.scp` / `prompt.scp` / `text.scp` (`<window_id>_ch<k>` rows) and window-level `mix.scp`.
- Checkpoint handling as `generate_dev.py` (`--ckpt` optional, EMA default, omit for the pretrained zero-gate model).

## Measure stage

Wire `measure` into `run.py`; `conf/metrics.yaml` lists the four metric classes below, each with `inputs: {meta: meta}` and its own knobs.
All metrics live in `src/metrics/`, subclass `BaseMetric`, iterate `meta.scp`, write per-window JSONL plus any distribution artifacts under `inference_dir/<test_name>/scoring/<metric_name>/`, and return summary floats (the `VersaMetric` precedent).
Every metric takes injectable backends (transcriber, embedder, VAD, MOS predictor) as constructor arguments with real defaults, so unit tests run CPU-only with fakes and the real models are exercised by asset-gated tests.
Audio is 24 kHz; each backend resamples to its native rate (16 kHz for whisper/WavLM-SV/UTMOS) internally via a shared loader.

Shared utility `src/metrics/segments.py`: wav loading + resampling, VAD wrapper (silero), IPU construction with the dGSLM 200 ms rule (configurable `min_silence`, `min_speech`, edge padding), and re-exported interval helpers from `crosstalk_report.py`.

1. **`ConversationASRMetric`** (faster-whisper large-v3, word timestamps, transcribed per IPU to avoid silence hallucination; jiwer for alignment):
   - per-channel WER (concatenated IPU hypotheses vs channel reference; Whisper English text normalizer applied to both sides),
   - channel-permutation cpWER (min over channel-to-speaker assignments; permutations are trivial for N <= 4) with `swap` flagged when argmin is not identity,
   - script following: monotonic edit-distance alignment of the channel's hypothesis words to its scripted turn texts, realized turn time = mean timestamp of aligned words, then turn-order accuracy, Kendall tau, and turn-count ratio across channels.
   - Summary keys: `wer_ch_mean`, `wer_ch_worst`, `cpwer`, `swap_rate`, `turn_order_acc`, `kendall_tau`, `turn_count_ratio`.
2. **`SpeakerDynamicsMetric`** (WavLM-large SV embedding, cosine):
   - `sim_o_mean` (generated speech regions vs own prompt), per-IPU cross-turn consistency `sim_consistency` (mean pairwise), drift curve with `sim_drift_slope`,
   - cross-channel confusion `confusion_mean` (gen ch_i vs prompt ch_j, i != j),
   - generated bleed dB over ground-truth solo spans (reusing solo-region logic): `bleed_db_p50`, `bleed_db_p90`.
3. **`InteractionMetric`** (VAD only, no ASR):
   - dGSLM battery per condition: for each event type in {IPU, pause, gap, overlap}, events per minute and cumulated duration per minute, computed identically on generated and paired ground-truth audio,
   - Wasserstein-1 distance per event-duration distribution vs the paired ground truth (`w1_gap`, `w1_overlap`, `w1_pause`, `w1_ipu`),
   - backchannel proxy: short-IPU-during-overlap rate; optional laughter events per minute and mean duration behind a config gate and soft import (candidate tool jrgillick/laughter-detection; timebox it, drop if unusable).
4. **`ChannelQualityMetric`**: UTMOS (speechmos, utmos22-strong) per IPU, speech-duration-weighted mean per channel: `utmos_mean`; optional DNSMOS behind a config flag.

Plus `local/eval_report.py`: reads `metrics.json` from several inference dirs (conditions: gt / resynth / pretrained / finetuned) and emits one Markdown comparison table, conditions as columns.

## Dependencies

Eval-only extras, exact pins decided against the Delta pixi env at implementation time: `faster-whisper`, `silero-vad`, `speechmos`, `jiwer`, `scipy` (W1), WavLM-large SV checkpoint; optional laughter tool.
None of these may become imports of the training path.

## Task breakdown (one subagent per task, each independently testable)

Order: 1 -> 2, then 3/4/5 in parallel, then 6.

1. **Shared inference library + infer stage**: refactor `generate_dev.py` internals into a shared module; implement the stage (selection, snapping, three modes, output contract); fixture tests with the tiny random-init DiT and fabricated manifests assert the SCP/meta contract exactly (golden meta JSON), snapping correctness (no turn straddles the boundary), and mode `gt`/`resynth` layout parity; `generate_dev.py` still works.
2. **Measure wiring + segments utility**: `run.py` stage, `conf/metrics.yaml`, `src/metrics/segments.py` with synthetic-audio tests (known silence/speech patterns produce exact IPU/gap/pause/overlap intervals); a stub metric proves the `measure` round trip writes `metrics.json`.
3. **`ConversationASRMetric`**: fake-transcriber unit tests with canned words/timestamps covering WER bookkeeping, cpWER permutation and swap flag, and script-following on constructed misorderings; asset-gated real-whisper smoke.
4. **`SpeakerDynamicsMetric`**: fake-embedder unit tests (deterministic vectors) for SIM-o, consistency, drift, confusion; bleed-dB tests on synthetic two-channel audio with known leakage; asset-gated real-WavLM smoke.
5. **`InteractionMetric` + `ChannelQualityMetric`**: exact-expectation tests on synthetic on/off speech patterns (constructed gaps, overlaps, backchannel IPUs); W1 against hand-computed values; UTMOS weighting tests with a fake MOS backend; laughter gated and skippable.
6. **End-to-end + docs**: asset-gated smoke running `infer` (pretrained, tiny window count) then `measure` on Delta-shaped fixtures locally; `local/eval_report.py` with tests; README section (running the battery, metric definitions, summary-key glossary); update `conf/metrics.yaml` defaults to the debug-run configuration.

Acceptance for the whole step: `python run.py --stages infer --inference_config conf/inference_conversational.yaml` then `--stages measure` produces `metrics.json` on fixtures without corpus or checkpoint; all new tests pass in the existing suite command; no training-path file modified.

## First real run (after merge, on Delta)

1. `infer` in `gt` and `resynth` modes on the same 50-window valid selection (anchors).
2. `infer` with the assembled pretrained model (no `--ckpt`, zero gates).
3. `measure` on all three dirs; `local/eval_report.py` table.
4. Expected signature: WER/SIM/UTMOS near resynth anchor, interaction metrics collapsed (heavy unstructured overlap, script order violated); investigate as a code bug only if A/B metrics are also broken.
5. Repeat with the fine-tuned checkpoint from the Delta batch run once it exists.
