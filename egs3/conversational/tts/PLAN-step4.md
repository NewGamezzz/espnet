
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

## Revision (2026-07-15, PR #10 review)

Thanapat's review of PR #10 asked for two changes, implemented as two follow-up tasks on the same branch.

### New prompt scheme

The infer stage no longer snaps a prompt boundary inside the evaluated window.
Instead it samples ONE turn per channel from ELSEWHERE in the same conversation (never from inside the evaluated window, so target leakage is never allowed), via a relaxation ladder (band-and-solo, then solo, then non-window; the non-window leakage rule is never relaxed).
Those turns are concatenated non-overlapping at the start of the conditioning audio, and the model generates the FULL ground-truth window, not just a continuation after a boundary.
Every channel is therefore guaranteed a voice reference in its prompt.
The meta JSON gained top-level `mix_wav` and a `prompt` block (`turns`, `total_sec`, `total_frames`); `prompt_boundary_sec` and `prompt_boundary_frames` were removed.
`channels[ch].gen_wav` now covers the whole window, `channels[ch].prompt_wav` is that channel's own solo turn block, and `channels[ch].ref_text` covers all of that channel's turns in the window.
See `src/inference.py`'s module docstring for the authoritative contract.

### Lean metric scope

The metric battery was cut down to three classes so the PR is easy to review: `ConversationASRMetric` (corpus-level `wer_channel` and `wer_mix`, pooled counts, never a mean of per-utterance WERs), `SpeakerSimilarityMetric` (`sim_o_mean`, prompt vs. whole generated channel, no VAD or segmentation), and `QualityMetric` (`utmos_mean`, one UTMOS call per window on the mixdown).
All VAD/IPU machinery (`src/metrics/segments.py`) was deleted along with it, since none of the three remaining metrics need it.

Deferred to a later PR (not on this branch):

- `InteractionMetric` (the dGSLM event battery, Wasserstein-1 duration distances, backchannel proxy).
- Cross-turn speaker consistency, drift, cross-channel confusion, and generated bleed dB.
- cpWER channel-permutation search and the `swap` flag.
- Script-following (turn-order accuracy, Kendall tau, turn-count ratio, and the word-timestamp machinery that fed it).
- Per-IPU anything (transcription, MOS weighting, embeddings).
- DNSMOS.
- The mixdown-diarization comparability protocol (already deferred before this revision).

See README.md's Evaluation section for the current metric glossary and the full deferred list.
5. Repeat with the fine-tuned checkpoint from the Delta batch run once it exists.

## Revision (2026-07-16, first Delta run)

The first real run (gt + resynth anchors + zero-gate pretrained on ~50 valid windows) exposed that the mixdown-only UTMOS definition is the least interpretable one: whole mixdowns embed overlap and silence, which UTMOS punishes regardless of audio quality (gt scored 1.52 on mixdowns vs the SSSD paper's per-utterance 2.55 +/- 0.72).
Delta-side debugging validated per-IPU scoring (Silero VAD, 200 ms min-silence, IPUs >= 1 s): per-IPU vs manifest-per-turn UTMOS agreed within ~0.1 on both anchors, and manifest turn spans are alignment-invalid for generated audio anyway (the model places speech at times of its own choosing).

`QualityMetric` therefore changed (this revision's PR):

- `utmos_ipu_mean` (PRIMARY) - UTMOS per VAD-derived IPU on every channel, pooled over the run.
- `utmos_mix_mean` - the previous `utmos_mean` (one whole-mixdown score per window), renamed and kept for continuity.
- `ipu_count` - number of scored IPUs; a free over-generation diagnostic (pretrained produced ~2x gt's count by filling the forced window duration).

The VAD backend is faster-whisper's bundled Silero ONNX model (`SileroVADSegmenter`), so no new dependency; it is constructor-injectable and lazy like every other backend.
"Per-IPU anything" in the deferred list above narrows accordingly: per-IPU transcription and per-IPU embeddings stay deferred, per-IPU MOS is now shipped.

## Revision (2026-07-16, InteractionMetric)

The dGSLM turn-taking battery (design ratified 2026-07-14, deferred by the PR #10 review) lands as `InteractionMetric`, scoped with Thanapat to core events + W1:

- Per-channel Silero VAD IPUs (the 200 ms rule; reuses PR #12's `SileroVADSegmenter`, injectable) -> IPU/pause/gap/overlap events per window, window-edge silences skipped (no before/after speaker to classify).
- `{e}_per_min` and `{e}_sec_per_min` per event type, POOLED over the run (total / total window minutes, never a mean of per-window rates - the corpus-WER pooling rule applied to rates).
- `{e}_dur_w1`: Wasserstein-1 distance between generated and ground-truth event-duration distributions, pooled; the reference comes from `channels[ch].gt_wav` in the same meta (no cross-directory pairing), so in `gt` mode every defined W1 collapses to ~0 as a built-in sanity check.

Still deferred from family D: the backchannel proxy (short-IPU-during-overlap), laughter statistics, script-following, and the Fisher reference corpus for cross-corpus W1.
