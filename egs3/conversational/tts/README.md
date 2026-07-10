# Conversational F5-TTS recipe (SSSD)

Data pipeline for multi-channel conversational TTS fine-tuning on the ScalableSpontaneousSpeechDataset (SSSD).
It windows long dyadic sessions into training segments, preprocesses transcripts into per-branch masked token sequences, and provides a dataset plus a packed collator ready for training.
The trainer, model wrapper, and training loop are a later task; this recipe currently covers data only.
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
2. **Text normalization** (`dataset/preprocessing/text.py`): turn texts are normalized ONCE at build time against the extended vocab charset (whitespace collapse, lowercase fallback, OOV drop), so `<OTHER>` counts can never desync between branches.
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

## Debug tools

```bash
# Mixdown wav + human-readable branch texts (<turn> as |, <OTHER> as #) for eyeballing:
python local/dump_debug.py --split valid --num-windows 5 --out-dir exp/debug_dump

# Channel-bleed measurement over solo-speech regions (report, not an assertion):
python local/crosstalk_report.py --num-sessions 20 --out exp/crosstalk_report.tsv
```

## Tests

```bash
pytest egs3/conversational/tts/dataset/tests   # CPU-only, no corpus needed
SSSD_ROOT=/path/to/corpus pytest egs3/conversational/tts/dataset/tests/test_integration_sssd.py
```

All unit tests run on fabricated fixtures (synthetic FLAC files and hand-built manifests); the integration test over the real corpus is skipped unless `SSSD_ROOT` is set.
