# Conversational F5-TTS recipe (SSSD)

Multi-channel conversational TTS fine-tuning on the ScalableSpontaneousSpeechDataset (SSSD).
The data pipeline windows long dyadic sessions into training segments, preprocesses transcripts into per-branch masked token sequences, and provides a dataset plus a packed collator; the trainer fine-tunes pretrained F5TTS_Base as a multi-branch CFM with TAC exchanges injected at every block (see the Training section).
The `src/branch_exchange/` package (communication modules between transformer blocks) is documented in its own module docstrings.

## The masking scheme

Each conversation window has N audio channels (branches).
Branch `i` receives, for each turn in conversation order: one `<turn>` separator token, then the turn text as characters if the turn belongs to speaker `i`, else exactly one `<OTHER>` token per character of the turn text.

```
Spk1: Good afternoon. How are you?
Spk2: Good. What about you?
Spk1: Good, but I have a problem with...

Input branch 1: <turn> Good afternoon. How are you? <turn> <OTHER>*21 <turn> Good, but I have a problem with...
Input branch 2: <turn> <OTHER>*30 <turn> Good. What about you? <turn> <OTHER>*35
```

Rules (fixed by design, see `dataset/preprocessing/text.py`):

- Turn ORDER only: no timestamps, durations, or alignment information ever appear in the token sequence.
- One `<OTHER>` per character preserves the conversation's length budget without using timestamps.
- `<OTHER>` is a new vocab token, distinct from F5's internal filler ("another speaker is talking" vs "text has ended").
- Turn markers carry NO speaker identity: a single `<turn>` token precedes every turn, identical across branches and speakers, so no vocab token depends on the speaker count.
- No trailing padding in preprocessing; the F5 model pads text up to the mel length itself.

## New vocab tokens

The builder appends exactly two tokens, `<turn>` and `<OTHER>`, at the END of the user-supplied base vocab, so every pretrained token id is unchanged.
The extended vocab is written as a new file (`data/tokens/vocab.txt`, pure token-per-line; the original is never edited).
Because the line index IS the token id, the vocab file itself carries no header comment; the new ids are documented in `data/tokens/vocab_meta.json`:

```json
{
  "base_vocab_path": "...", "base_vocab_size": 2545,
  "base_vocab_sha256": "<hash of the base vocab file bytes>",
  "new_tokens": {"<turn>": 2545, "<OTHER>": 2546}, "total_size": 2547
}
```

`base_vocab_size` and `base_vocab_sha256` are a provenance guard: before loading pretrained weights, training asserts them against the vocab shipped with the checkpoint, so a build against the wrong base vocab fails before it can corrupt the text-embedding alignment.

## Building

```bash
python -m egs3.conversational.tts.dataset.builder \
    --dataset-root /work/hdd/bbjs/ttrachu/dataset/ScalableSpontaneousSpeechDataset \
    --base-vocab-path /path/to/pretrained/char_tokens.txt \
    --seed 0
```

`base_vocab_path` is required (one token per line, `char_tokens.txt` format); the builder fails loudly without it.
Text normalization delegates to the F5 tokenizer, so the build environment needs `pip install rjieba pypinyin` (see the Pipeline section).
The build prints a summary: windows per split, window duration distribution, turns and exchanges per window, windows and hours by active speaker count, mini-window count and hours, overlap ratio, speaker overlap across splits, dropped-audio statistics (oversized blocked spans, slivers, and tails, all reported separately), and the distribution of the all-channel gap at chosen cut points including how many fall below 0.2 s (boundaries the former all-channel-silence rule could not use).
Single-speaker windows are NOT filtered at build time; the per-window speaker-activity fields exist so filtering or weighting can happen at training time without a rebuild.
`SSSDBuilder` subclasses the espnet3 `DatasetBuilder`, so it also plugs into the stage machinery once a `run.py` exists.

### Dataset-root remapping rule

Absolute audio paths inside `recordings.jsonl.gz` are valid only on the machine that wrote them.
The loader keeps only `original/<basename>` and joins it onto `dataset_root`, resolved as: explicit argument > `$SSSD_ROOT` > `builder.dataset_root` in `dataset/config.yaml`.
The corpus directory is treated as strictly read-only; only `lhotse_manifests_48/` and `original/` (48 kHz) are used.

## Pipeline

The pure algorithms live in the `dataset/preprocessing/` package; `dataset/builder.py` (build time), `dataset/dataset.py`, and `dataset/preprocessor.py` (training time) orchestrate them.

1. **Turn construction** (`dataset/preprocessing/sssd.py`): per session, supervisions are sorted by start; consecutive same-channel utterances merge into one turn when the gap is below `merge_gap` (default 1.0 s); texts join with single spaces.
2. **Text normalization** (`dataset/preprocessing/text.py`): turn texts are normalized ONCE at build time, so `<OTHER>` counts can never desync between branches.
   `normalize_text` first runs the pretrained checkpoint's own tokenizer (`espnet2.text.f5_pinyin.convert_char_to_pinyin`, requires `rjieba` + `pypinyin`), so fine-tuning text matches the F5TTS_Base pretraining distribution by construction - including its `;` -> `,` translation and jieba's space after hyphen compounds (`Turn-taking` -> `Turn- taking`); a CJK character (multi-char pinyin token) fails the build loudly, since the char-level `<OTHER>` budget cannot represent it.
   Charset filtering follows (whitespace collapse, lowercase fallback, OOV drop).
3. **Windowing** (`dataset/preprocessing/windows.py`): sessions are cut into windows at eligible utterance boundaries, with target duration uniform in `[window_min, window_max]` (default 10-60 s).
   A time instant `t` is an eligible boundary iff every merged turn on every channel ends at least `boundary_guard` before `t` or starts at least `boundary_guard` after it; with the default `boundary_guard: 0.0` this means no turn strictly contains `t`, so zero-gap speaker exchanges are valid cut points and no utterance is ever truncated.
   The eligibility rule follows CoVoMix's Fisher segmentation ([arXiv:2404.06690](https://arxiv.org/abs/2404.06690)); `boundary_guard` exists because SSSD timestamps are Parakeet pseudo-labels rather than human alignments, so a positive guard rejects boundaries where a neighbor's alignment jitter could leak un-covered speech into the window.
   The placement search is ours (CoVoMix streams to the first clean boundary and has no target duration): each window cuts at the eligible boundary in `[window_min, window_max]` from the current position closest to its drawn target.
   When one blocked span covers that whole range, the search restarts at the span's start edge, emitting the prefix as a mini-window if it is at least `tail_min` (mid-session windows in `[tail_min, window_min)` are legal) and counting it as a dropped sliver otherwise; a span longer than `window_max` is then dropped exactly, never the audio adjacent to it.
   The default `window_max: 60` exceeds the F5 pretraining clip regime (< 30 s Emilia clips) deliberately, to capture longer interactions; revisit if fine-tuning quality degrades on long windows.
   Force-splitting oversized spans at internal supervision pauses (ZipVoice-Dialog style) is future work.
   The session tail is emitted iff it is at least `tail_min` (default 5 s); windows without speech are dropped.
4. **Splits**: session-level train/valid/test split, seeded, ratios in config; speaker overlap between splits is reported (not enforced).
5. **Audio loading** (`dataset/dataset.py`, on the fly): only the window's segment is seek-read from the FLAC, channels stay separate, and audio is resampled 48 -> 24 kHz with `torchaudio.functional.resample`; no precomputed audio copies.

## Window-manifest schema (`data/manifest/{train,valid,test}.jsonl`)

One JSON object per line:

```json
{
  "window_id": "<session>_w00007",
  "session_id": "<session>",
  "audio_relpath": "original/<session>_mixed.flac",
  "num_channels": 2,
  "sample_rate": 48000,
  "t0": 123.456, "t1": 145.052, "duration": 21.596,
  "num_active_speakers": 2,
  "channel_speech_sec": [12.4, 6.1],
  "exchange_count": 3,
  "turns": [
    {"channel": 0, "speaker": "<hash>", "text": "can you hear me", "start": 124.01, "end": 126.33}
  ]
}
```

Turn `start`/`end` are absolute session seconds and exist for windowing and later evaluation only; they never become tokens.
`num_active_speakers` (channels with at least one turn), `channel_speech_sec` (per-channel sum of turn durations), and `exchange_count` (speaker alternations in start order; 0 for single-speaker windows) are derived from the turns and enable training-time filtering (e.g. a `min_active_speakers` threshold) or interaction-density weighting without a rebuild.

## Dataset, preprocessor, and packed collator

`ConversationDataset[idx]` is vocab-agnostic and returns raw material: per-channel audio `(N, T)` at 24 kHz, the window's turns, and the channel permutation applied.
Turn `channel` fields are already remapped to post-permutation row indices, so everything downstream is permutation-agnostic.
Tokenization happens in `ConversationalTextPreprocessor` (`dataset/preprocessor.py`), configured with the extended vocab as `token_list`; it derives the N per-branch token-id tensors from the turns and fills in the item's `text` key.
This mirrors the libritts recipe shape (there `CommonPreprocessor` fills the slot): training configs wire it via the `DataOrganizer` `preprocessor:` slot, e.g.

```yaml
preprocessor:
  _target_: egs3.conversational.tts.dataset.preprocessor.ConversationalTextPreprocessor
  token_list: ${data_dir}/tokens/vocab.txt
```

Unlike `CommonPreprocessor` there is deliberately no cleaner and no `<blank>/<unk>/<sos/eos>` symbols: tokenization must stay exactly the pretrained F5TTS_Base convention (raw characters, case preserved, id = vocab line index), or the ids stop aligning with the pretrained text-embedding matrix.
Per-sample channel permutation augmentation (train split only by default) is applied consistently to audio channels and turn channels; it guards against systematic ch0/ch1 artifacts in the corpus.
`collate_conversations` emits the packed layout of the merged `branch_exchange` package: a per-conversation `counts` list plus row-stacked tensors with no padding rows on the branch axis.

```
counts:          [N_1, ..., N_B]                    -> BranchContext.branches(counts)
speech:          float32 (sum(counts), T_max)       pad 0.0
speech_lengths:  int64   (sum(counts),)
speech_mask:     bool    (sum(counts), T_max)       True = valid
text:            int64   (sum(counts), L_max)       pad -1 (F5 shifts ids by +1; 0 = filler)
text_lengths:    int64   (sum(counts),)
window_ids:      [str] * B
```

Batches do not need a homogeneous channel count; a duration-bucketed sampler should budget by total rows (sum of N_i x frames), not by conversation count.

## `dataset/config.yaml` parameters

| key | default | meaning |
|---|---|---|
| `builder.dataset_root` | Delta corpus path | corpus root; overridden by `$SSSD_ROOT` or explicit args |
| `builder.manifests_subdir` | `lhotse_manifests_48` | 48 kHz lhotse manifests (never the 16 kHz copies) |
| `builder.audio_subdir` | `original` | 48 kHz FLAC directory |
| `builder.base_vocab_path` | `null` (required) | pretrained char vocab to extend |
| `builder.seed` | `0` | drives windowing and splits |
| `builder.merge_gap` | `1.0` | max same-channel gap (s) merged into one turn |
| `builder.window_min/max` | `10.0` / `60.0` | window target duration range (s); 60 exceeds the F5 pretraining clip regime on purpose |
| `builder.boundary_guard` | `0.0` | margin (s) every turn must keep from a cut point; 0 = CoVoMix-faithful zero-gap boundaries |
| `builder.tail_min` | `5.0` | shortest emitted window below `window_min` (session tails and mini-windows) (s) |
| `builder.split_ratios` | 0.96/0.02/0.02 | session-level split |
| `dataset.sample_rate` | `24000` | training rate (48 kHz source is downsampled 2:1) |
| `dataset.text_pad_value` | `-1` | F5 text padding convention |

## Training (multi-branch CFM POC)

`run.py --stages train --training_config conf/training_poc.yaml` fine-tunes pretrained F5TTS_Base on SSSD windows as a multi-branch CFM: every channel of a window is one packed transformer row, and the injected `branch_exchange` modules are the only cross-channel communication.

### Model assembly (`src/build_model.py`, order is load-bearing)

1. Build the DiT/CFM with the F5TTS_Base architecture values and `text_num_embeds` = size of the EXTENDED vocab from step 2.
2. Load the pretrained checkpoint with text-embedding surgery: every original embedding row is copied bit-exactly (step 2 appended the new tokens at the end), `<turn>` is warm-started from the space character's row and `<OTHER>` from the filler row 0 (F5's internal padding token), each plus small Gaussian noise; every other weight must load exactly (strict load).
   Before any weight is read, `vocab_meta.json`'s `base_vocab_sha256`/`base_vocab_size` are asserted against the vocab file shipped with the checkpoint.
3. `inject_exchange` with the configured schedule (POC: `{"1-22": "P+TAC"}`, `TACExchange`).
   Gates are zero-init, so at this instant the model computes exactly N independent pretrained F5 passes.
4. At `configure_optimizers` time the recipe LightningModule (`src/lit_module.py`) builds ONE AdamW over two param groups: exchange parameters at `optim.lr_exchange`, backbone at `optim.lr_backbone`.

### Shared span and shared flow time

The infilling span is sampled once per conversation and shared by all its channels, so the region the model must generate is time-aligned across speakers while every channel's unmasked remainder acts as that speaker's voice prompt.
The flow time is likewise shared per conversation because at inference all channels ride one ODE trajectory at a common t, and training must match; only the noise stays independent per channel.

### Batching (`src/sampler.py`)

A recipe-local duration-bucketed batch sampler computes each window's cost from metadata alone (`round(24000 * (t1 - t0))` sample-rows per channel; never loads audio) and packs batches under `dataloader.<split>.batch_bins`, padded to the longest window in the batch.
The stock espnet3 iter_factory path was rejected: its shape files would have to stay in sync with the `min_active_speakers` filter, and `DataLoaderBuilder._build_iter_factory` calls `build_iter(epoch, shuffle=False)`, freezing the batch order across epochs even when the config says `shuffle: true`.
The recipe LightningModule instead builds a standard `DataLoader` around the sampler each epoch (espnet3 forces `reload_dataloaders_every_n_epochs=1`) with the current epoch as the shuffle seed, and the sampler reproduces the iter_factory path's DDP policy (drop tail batches to a multiple of world size, stride by rank).

### Config knobs (`conf/training_poc.yaml`)

| key | default | meaning |
|---|---|---|
| `model.arch.*` | F5TTS_Base values | copied verbatim; wrong `text_mask_padding`/`pe_attn_head` loads cleanly but produces noise |
| `model.exchange.schedule` | `{"1-22": P+TAC}` | 1-indexed inclusive block ranges; `P` = no exchange at that depth |
| `model.exchange.hidden` | `null` (= dim) | TAC hidden width |
| `model.init_noise_scale` | `0.02` | Gaussian noise on the two warm-started embedding rows |
| `optim.lr_exchange` / `optim.lr_backbone` | `1e-4` / `1e-5` | the two param groups of the single AdamW |
| `dataset.*.data_src_args.min_active_speakers` | `2` | drop windows with fewer active speakers (knob, not a rebuild; relax if per-channel quality drifts) |
| `dataloader.train.batch_bins` | `6000000` | packed row budget in sample-rows (N x T_24k, padded); a 60 s N=2 window costs 2.88M |
| `dataloader.train.min_batch_size` | `1` | counts conversations, not rows |

### Running the smoke training

```bash
cd egs3/conversational/tts
huggingface-cli download SWivid/F5-TTS F5TTS_Base/model_1200000.safetensors --local-dir downloads
huggingface-cli download SWivid/F5-TTS F5TTS_Base/vocab.txt --local-dir downloads
python run.py --stages create_dataset          # SSSD manifests + extended vocab
python run.py --stages train                   # logs total + per-channel losses
```

`loss_ch{k}` is logged without DDP sync (its key set varies with the batch's largest channel count, and ragged key sets across ranks would deadlock the synced path).
Under the random channel permutation the per-channel curves are symmetric in expectation, so they are a row-symmetry bug canary, not a quality metric: ch0/ch1 separating during training signals a broken branch symmetry (injection, collator ordering, masking).

### Sanity generation

```bash
python local/generate_dev.py \
    --training_config conf/training_poc.yaml \
    --ckpt exp/train_poc_multibranch_f5/checkpoints/last.ckpt \
    --index 0 --prompt_sec 3.0 --out_dir exp/generate_dev
```

One dev window; the first `--prompt_sec` seconds of every channel are the acoustic prompt, each branch is conditioned on its full masked script, and the joint ODE runs with the exchanges active and CFG as in the single-channel inference path.
Outputs per-channel wavs, a mixdown, and a dump of the masked scripts; separated channels with sensible turn-taking = POC signal (quality is not the criterion).
Omitting `--ckpt` generates with the freshly assembled pretrained model (zero-init gates = N independent F5 passes), which is the audible baseline; with `--ckpt` the pretrained load is skipped entirely, so the `downloads/` dir is not needed on the generating machine.
`--seed` reproduces a run bit-exactly while keeping the per-channel noise independent (`CFM.sample`'s upstream per-row reseed would start every channel from identical noise, off the training distribution).

## Evaluation

The eval battery is two espnet3 stages plus a report tool: `infer` (`src/inference.py`, `conf/inference_conversational.yaml`) batch-generates multi-channel conversations from manifest windows in one of three modes; `measure` (`espnet3.systems.base.metric.measure`, driven by `conf/metrics.yaml`) runs four custom metric classes (`src/metrics/`) over `infer`'s output and writes `metrics.json`; `local/eval_report.py` turns one or more `metrics.json` files into a single Markdown comparison table.
No diarization is needed anywhere in the battery: this recipe's channels ARE speakers (channel = speaker, by construction, since SSSD audio is per-participant), so every speaker-attributed metric below reads straight off the channel index.

### Running the battery

```bash
cd egs3/conversational/tts

# infer: batch-generate from the valid split (mode: generate is the shipped
# default, i.e. the assembled model, zero-init gates unless --ckpt is wired
# into conf/training_poc.yaml); writes exp/train_poc_multibranch_f5/infer_generate/valid/
python run.py --stages infer --inference_config conf/inference_conversational.yaml

# measure: score whatever infer just wrote; writes metrics.json next to it
python run.py --stages measure --metrics_config conf/metrics.yaml

# report: one Markdown table, conditions as columns
python local/eval_report.py \
    --label pretrained exp/train_poc_multibranch_f5/infer_generate \
    -o exp/train_poc_multibranch_f5/report.md
```

Output paths land under `exp/train_poc_multibranch_f5/` (the TRAINING config's `exp_tag`), not the inference config's own `exp_tag: eval_pretrained`: `run.py` always loads `conf/training_poc.yaml`, and the runner propagates its `exp_tag`/`exp_dir` into the inference and metrics configs with the same overwrite-on-differ rule described below, so the inference yaml's `exp_tag` never takes effect in the default invocation.

`mode: generate | gt | resynth` selects what `infer` writes for the SAME window selection and prompt-boundary logic: `generate` runs the model; `gt` copies the ground-truth generated-region audio unchanged; `resynth` round-trips that same ground truth through the F5 mel front-end and Vocos, with no model involved.
`gt` and `resynth` are anchors, not skipped conditions: they flow through the identical `infer` output contract and the identical `measure` stage as `generate`, with zero metric-side special-casing, so a `generate`-mode score is only meaningful read AGAINST them (a mediocre WER/SIM/UTMOS next to an equally mediocre `resynth` anchor points at the vocoder/front-end, not the model).
`mode` has no CLI override flag (this recipe's stage configs are plain YAML, not hydra dotlist overrides), so scoring the anchors means pointing `--inference_config`/`--metrics_config` at copies of the two shipped files with `mode:` changed - keep the two copies' `mode:` in lockstep (`conf/metrics.yaml`'s `inference_dir: ${exp_dir}/infer_${mode}` formula needs its own `mode:` to match whatever `infer` actually wrote, see that file's header comment).
CRITICAL: the `measure` invocation must ALSO pass the mode-edited `--inference_config`, even though the measure stage never runs inference.
`run.py` always loads an inference config (defaulting to the shipped generate-mode file), and the runner's context propagation copies `inference_dir` from inference_config into metrics_config whenever the two resolved values DIFFER (overwrite-on-differ, with only a logged warning) - so a `measure` run that omits the flag silently scores `infer_generate` instead of the anchor directory.
This exact scenario is pinned by `tests/test_run.py::TestAnchorModePropagation`.

```bash
for m in gt resynth; do
  sed "s/^mode: generate/mode: $m/" conf/inference_conversational.yaml > conf/inference_$m.yaml
  sed "s/^mode: generate/mode: $m/" conf/metrics.yaml               > conf/metrics_$m.yaml
  python run.py --stages infer   --inference_config conf/inference_$m.yaml
  python run.py --stages measure --inference_config conf/inference_$m.yaml \
                                 --metrics_config   conf/metrics_$m.yaml
done

python local/eval_report.py \
    --label gt         exp/train_poc_multibranch_f5/infer_gt \
    --label resynth    exp/train_poc_multibranch_f5/infer_resynth \
    --label pretrained exp/train_poc_multibranch_f5/infer_generate \
    -o exp/train_poc_multibranch_f5/report.md
```

`--label NAME INFERENCE_DIR` may be repeated for any number of conditions (add a fine-tuned checkpoint's `infer_generate` dir the same way once one exists); columns appear in the given order and a condition with no `metrics.json` yet (or missing a particular metric/key) renders as `-` rather than failing the report.

### Metric glossary (summary keys in `metrics.json`)

Every key below is a MEAN OVER WINDOWS of a per-window (or per-channel-then-window) scalar unless stated otherwise; full per-window detail lives alongside `metrics.json` under `<inference_dir>/<test_name>/scoring/<metric_name>/windows.jsonl`.

**`ConversationASRMetric`** (faster-whisper, transcribed per IPU, not per whole channel, to avoid whisper's silence-hallucination failure mode on long quiet stretches):

- `wer_ch_mean` / `wer_ch_worst` - per-window mean / worst per-channel WER (normalized hypothesis vs. reference text), meaned over windows.
- `cpwer` - channel-permutation WER, minimized over every channel<->speaker assignment, meaned over windows.
- `swap_rate` - fraction of windows whose cpWER-minimizing assignment was not the identity (a real channel/speaker mixup, not just transcription noise).
- `turn_order_acc` - fraction of realized turns, pooled chronologically ACROSS channels, whose time-rank matches script order.
- `kendall_tau` - Kendall rank correlation between script order and realized-time order over the same pooled, cross-channel turns.
- `turn_count_ratio` - realized / scripted turn count (a turn with zero aligned hypothesis words is "missing").

**`SpeakerDynamicsMetric`** (WavLM-SV x-vector embeddings, cosine similarity):

- `sim_o_mean` - a channel's generated speech vs. that SAME channel's own acoustic prompt (identity preservation).
- `sim_consistency` - mean pairwise cosine among a channel's per-IPU embeddings (identity stability within the generated region).
- `sim_drift_slope` - least-squares slope of per-IPU-to-prompt cosine against IPU index (identity drift across the generated region; negative = drifting away from the prompt voice).
- `confusion_mean` - a channel's generated speech vs. every OTHER channel's prompt (cross-speaker identity leakage; higher is worse).
- `bleed_db_p50` / `bleed_db_p90` - percentiles of generated-audio energy leaking into another channel's ground-truth SOLO spans (channel k's energy while only channel j is scripted to speak, relative to channel j's own energy there), POOLED across every scored pair and every window (not a mean of per-window means - the population of interest is "how bad is a bleed pair", not "how bad is a window").

**`InteractionMetric`** (VAD only, no ASR; dGSLM-style event battery over IPU/pause/gap/overlap):

- `{ipu,pause,gap,overlap}_per_min` - event rate per minute of generated-region audio.
- `{ipu,pause,gap,overlap}_dur_per_min` - cumulated event duration per minute.
- `w1_{ipu,pause,gap,overlap}` - Wasserstein-1 distance between the generated and the PAIRED ground-truth event-duration distributions for that same window, computed per window then meaned (the pairing is by construction: every window's `gen_wav` has a same-window `gt_wav`, so this is never an unpaired distribution comparison across different windows).
- `backchannel_per_min` - rate of short IPUs occurring during overlap (a backchannel proxy).
- `laughter_per_min` / `laughter_mean_dur` - optional, present only when a laughter detector is configured (off by default).

**`ChannelQualityMetric`** (UTMOS22-strong via `torch.hub`, per IPU):

- `utmos_mean` - speech-duration-WEIGHTED mean UTMOS per channel (longer IPUs count for more), meaned over channels and windows.
- `dnsmos_mean` - optional, present only when `enable_dnsmos: true` (off by default).

### Conventions that matter when reading the numbers

- **Diarization-free by construction**: every speaker-attributed metric (WER, SIM, bleed) reads the speaker directly off the channel index; there is no diarization step and none is planned for the per-channel numbers (a separate mixdown-diarization comparability protocol against cpWER/cpSIM baselines is explicitly out of scope for this step).
- **Per-IPU transcription, not per-channel**: `ConversationASRMetric` calls the transcriber once per VAD-derived IPU (dGSLM 200 ms rule) and concatenates the results, rather than transcribing a channel's full generated region in one call, specifically to avoid whisper hallucinating words into long silent stretches.
- **UTMOS is relative-use-only**: UTMOS is a no-reference MOS predictor trained on a particular corpus distribution; treat `utmos_mean` as meaningful only RELATIVE TO the `gt`/`resynth` anchor runs on the same window selection, never as an absolute cross-corpus quality number.
- **W1 is paired per window**: every `w1_*` key compares a window's generated event-duration distribution against that SAME window's ground truth, then means the per-window distances - it is never a pooled generated-vs-pooled-ground-truth comparison across different windows.
- **Bleed percentiles are pooled, not per-window**: `bleed_db_p50`/`bleed_db_p90` are percentiles over the flat list of every scored bleed pair across every window in the run, unlike every other summary key (which is a mean of per-window values).

### Dependencies (eval-only, all lazy, none needed for training)

`faster-whisper` (transcription), `openai-whisper` (only for `whisper.normalizers.EnglishTextNormalizer`; missing at real-runtime raises loudly with an install hint rather than silently falling back to a weaker normalizer, since that would corrupt cross-run WER comparability), `transformers` (WavLM-SV x-vector embedding), `torch.hub` (silero VAD, shared by all four metrics; UTMOS22-strong), and optionally `speechmos` (DNSMOS, config-gated, off by default).
Every backend is constructor-injectable and every real default defers its import/download to the first actual call, never to module import or `__init__`, so constructing any metric class - including hydra-instantiating the full `conf/metrics.yaml` - is always safe offline; unit tests exercise the metric math with injected fakes, and none of this is imported anywhere on the training path.

### Fail-fast on bad audio (v1 design decision)

A corrupt or missing wav aborts the whole `measure` run: no metric catches or skips a bad window internally, so a single bad file fails loudly rather than the run silently under-reporting a metric over fewer windows than it claims.
Graceful per-window skip-and-count is future work, not implemented in v1.

## Debug tools

```bash
# Mixdown wav + human-readable branch texts (<turn> as |, <OTHER> as #) for eyeballing:
python local/dump_debug.py --split valid --num-windows 5 --out-dir exp/debug_dump

# Channel-bleed measurement over solo-speech regions (report, not an assertion):
python local/crosstalk_report.py --num-sessions 20 --out exp/crosstalk_report.tsv
```

## Tests

```bash
# All three suites in one run (the tests dirs are packages with relative conftest imports):
pytest egs3/conversational/tts/src/branch_exchange/tests \
       egs3/conversational/tts/tests \
       egs3/conversational/tts/dataset/tests
SSSD_ROOT=/path/to/corpus pytest egs3/conversational/tts/dataset/tests/test_integration_sssd.py
```

All unit tests run on fabricated fixtures (synthetic FLAC files, hand-built manifests, random-init tiny DiT; CPU-only, no corpus or checkpoint needed); the integration test over the real corpus is skipped unless `SSSD_ROOT` is set.

## Verification against the pretrained checkpoint

The unit suite proves invariants on tiny random-init models; two extra test files close the gap against the REAL `downloads/F5TTS_Base` assets and are skipped automatically when the download is absent (so CI stays green).

```bash
cd egs3/conversational/tts
huggingface-cli download SWivid/F5-TTS F5TTS_Base/model_1200000.safetensors --local-dir downloads
huggingface-cli download SWivid/F5-TTS F5TTS_Base/vocab.txt --local-dir downloads
# Optional, enables the forward-loss sanity test and the listening script:
curl -L -o downloads/ref/basic_ref_en.wav --create-dirs \
    https://github.com/SWivid/F5-TTS/raw/main/src/f5_tts/infer/examples/basic/basic_ref_en.wav

pytest tests/test_pretrained_real.py tests/test_preprocessing_parity.py
```

`tests/test_pretrained_real.py` covers vocab provenance, the surgery loader (every backbone tensor bit-exact, the checkpoint's `mel_spec.mel_stft.*` DSP buffers validated against a freshly built transform), zero gates after injection, a forward-loss sanity bound on real speech, and the gold check: with `counts=[1]` and zero gates the assembled model's `sample()` must match the baseline espnet2 `CFM` on identical inputs and seed.
`tests/test_preprocessing_parity.py` pins id-level parity between the conversational pipeline (normalize + per-char encode) and F5's own `text_to_pinyin_ids` on raw English text - `normalize_text` delegates to F5's tokenizer, so the formerly documented divergences (`;` -> `,`, jieba's hyphen-compound space) are now reproduced rather than diverged from - plus normalization idempotency, the loud CJK guard, and the `T_wav // hop + 1` frame convention with padded-row frame stability.

Listening artifacts need no SSSD data (unlike `local/generate_dev.py`):

```bash
python local/verify_pretrained_gen.py --steps 32 --two_channel --out_dir exp/verify
```

`single_multibranch.wav` must sound exactly like stock F5 (correct words in the reference voice); `single_baseline.wav` is the A/B render through the baseline `CFM` and the printed max mel diff must be ~0.
The `twochannel_*` renders are the qualitative pre-finetuning baseline (`<turn>`/`<OTHER>` embeddings are warm-started but untrained), not a pass/fail gate.
