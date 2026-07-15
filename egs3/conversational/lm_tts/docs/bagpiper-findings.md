# BagPiper findings (Phase 0 gate)

## SFT data schema (Task 2)

Source: `JinchuanTian/bagpipier_tts` (HF dataset repo), archives `advanced_tts.tar.gz` (1.7 MB, text-only eval prompts) and `dev_multi_talker.tar.gz` (64 MB, real training pairs with audio), downloaded via `hf download` (the `huggingface-cli` binary on PATH is the deprecated wrapper and printed only a deprecation notice with no download; `hf download` worked). Also pulled `bagpiper_tts.yaml` (maps every named split to a `.../stage5_dialogues/dataset.json` or `stage3_dialogues/<app>/dataset.json` path) and `manifest.json` (per-archive packing stats; confirms `advanced_tts` and `dev_multi_talker` are indeed the two smallest of ~14 named datasets, the rest ranging from hundreds of MB to 97 GB in multi-part tarballs).

Both archives extract to a `dataset.json` per "app" (`intent_to_speech`, `multi_talker`, `role_play`, `svs` for `advanced_tts`; a single `multi_talker` set for `dev_multi_talker`), each pointing at a sibling `dialogues.jsonl` that holds the actual records.

**`dataset.json` fields** (identical wrapper shape in every file found):
```json
{
  "data_entry": [
    {"name": "dialogue", "path": "<original absolute path to dialogues.jsonl>", "reader": "dialogue"}
  ],
  "samples": ["adv_multi_talker_001", "adv_multi_talker_002", ...]
}
```
- `data_entry`: single-element list, `path` is the (stale, machine-local) absolute path to the paired `dialogues.jsonl`; `reader` is always the literal string `"dialogue"`.
- `samples`: list of `example_id` strings. Verified this is a **sorted index into `dialogues.jsonl` by `example_id`, not a positional/ordered list** - for `dev_multi_talker`, `set(dataset.json["samples"]) == set(example_id for line in dialogues.jsonl)` is True but the literal order differs (`dataset.json["samples"] == sorted(dataset.json["samples"])`, while `dialogues.jsonl` is in original generation order). A pipeline consumer must join on `example_id`, not index position.

**`dialogues.jsonl` record fields** (one JSON object per line; this is the real training-record schema):
```json
{
  "example_id": "<string, unique>",
  "messages": [
    ["system", "text", "<system prompt string>"],
    ["user", "text", "<caption string, the TTS instruction/description>"],
    ["assistant", "text", "<optional: chain-of-thought <think>...</think> block>"],
    ["assistant", "audio", "<absolute .wav path>"]
  ],
  "metadata": { "...": "app-specific, see below" }
}
```
- Each entry in `messages` is a 3-element list `[role, modality, content]`. Observed `role` values: `system`, `user`, `assistant`. Observed `modality` values: `text`, `audio`.
- `advanced_tts` (all 4 apps, "test" split) entries have **only 2 messages** - `system` + `user` (text). There is **no `assistant` turn and no audio reference anywhere in this archive** - confirmed by grepping every `dialogues.jsonl` in `advanced_tts`: `roles={'user','system'}`, `modalities={'text'}` for all 4 apps (300 lines each). Cross-checked `stages/test/stage2_filtered/multi_talker/ground_truth.jsonl`, which likewise only carries `idx`, `transcriptions`, `user_request` - no audio path. **`advanced_tts` is a caption-only eval/prompt set, not a trainable (input, target-audio) pair set.**
- `dev_multi_talker` entries have **4 messages**: `system` text, `user` text (caption), `assistant` text (a `<think>...</think>` CoT block), `assistant` audio (one `.wav` path). This is the complete trainable schema with a real target.
- Audio content for the `assistant`/`audio` message is a **file path string**, not embedded token IDs, e.g. `/mnt/home/jinchuat-andr-d6b58f/jinchuat/espnet_speechlm_tts/egs2/gigaspeech/asr1/data/dev_multi_talker/audio/YOU1000000035/YOU1000000035_M0000024.wav`. The path is stale (points at the original training host's `/mnt/home/...` tree) but the **basename matches a real file shipped in the archive** at `downloads/bagpiper_sft/dev_multi_talker/audio/YOU1000000035/YOU1000000035_M0000024.wav` (verified present, 820 KB). No pre-extracted codec/discrete-token files ship in either archive - only raw `.wav`. A data pipeline must re-tokenize with the codec (Xcodec) itself; there is no shortcut token cache here.
- No explicit duration field anywhere in the records or metadata (`dev_multi_talker` metadata is just `{"utt_id": "..."}`); durations would have to be read from the wav headers.
- `metadata` shape differs by archive/app:
  - `advanced_tts` (`intent_to_speech`, `role_play`, `svs`, `multi_talker`): `{"app": "<app name>", "original_idx": "<source id>", "transcriptions": [...] (absent for intent_to_speech), "slot": {<app-specific templated fields>}}`.
  - `dev_multi_talker`: just `{"utt_id": "YOU1000000035_M0000024"}` - no `app`/`slot`/`transcriptions` keys.

**Caption examples (verbatim quotes, `user` message text)**

1. Multi-talker instruction style (`advanced_tts/multi_talker`, `adv_multi_talker_001`):
   > "Use a trembling, bright child voice that sounds enthusiastic, followed by a soft, warm adult male voice sounding weary, as a mentor gives casual advice. \"Okay, I know this seems tough, but you can totally ace this project!\" \"I understand how you feel; just pace yourself and remember it's okay to ask for help.\""

2. Intent-to-speech style (`advanced_tts/intent_to_speech`, `adv_intent_to_speech_001`):
   > "Help me express my disagreement at a parent‑teacher meeting, but do it in a hesitant, uncertain tone so it sounds respectful and careful."

3. Role-play style (`advanced_tts/role_play`, `adv_role_play_001`), byte-exact JSON-escaped form (note the two trailing spaces before `\n` - a markdown hard line break present in the source string):
   ```text
   "Khariton Volkov, a battle-scarred Siberian veteran of the Great Game, now a railway foreman in 1880s St. Petersburg, recounts his harrowing escape from a Cossack ambush.  \n\"I slipped through the snowdrifts as gunfire cracked, my heart pounding like the drums of war, and the frost bit my fingers as I vanished.\""
   ```

4. SVS style (`advanced_tts/svs`, `adv_svs_001`):
   > "Give me a rendition of a playful yodeling opera carol, bright and silly. \"Jingle bells on high, echo through the snowy hall, yodeling notes cascade, merry voices rise and fall.\""

5. Real multi-talker training caption (`dev_multi_talker`, `multi_talker_tts_YOU1000000040_M0000001`), byte-exact JSON-escaped form (each line ends with two trailing spaces before `\n` in the source):
   ```text
   "Female host: bright, upbeat, and energetic with a higher pitch  \nMale host: warm, deep, and enthusiastic with a resonant tone  \nNarrator: calm, clear, and professional with a neutral delivery  \n\nThe female host says: \"We're in Istanbul, Turkey.\"  \nThe male host says: \"The food culture here will blow your mind. We're so excited to be filming a bunch of food videos.\"  \nThe female host says: \"Let's get food hunting.\"  \nThe narrator says: \"Istanbul has an exciting and vibrant food scene. This city's diverse heritage is reflected in its incredible food culture. In this five-part series, we're going to show you some delicious local Turkish food.\""
   ```

**One complete multi-talker entry, verbatim** (`dev_multi_talker/stages/v1/stage5_dialogues/dialogues.jsonl`, line 1):
```json
{
  "example_id": "multi_talker_tts_YOU1000000035_M0000024",
  "messages": [
    ["system", "text", "You are a multi-talker text-to-speech system."],
    ["user", "text", "A British female narrator speaks with a clear, articulate, and professionally polished tone, her accent and rhythm reflecting a UK broadcast style. Her voice is calm and emotionally neutral, delivering information with precision and grace.  \nShe says: “Later that week I dropped by to meet a very special lady indeed. Sheila Reed, respected across the UK for her expertise in the Kangal breed. Both her and her husband Michael had kept Kangals for decades now, and had traveled out to Turkey many times to source dogs from working lines. If there was anyone who knew the Kangal better, then it would be her.”\n\nThen, a different woman’s voice comes in—more intimate and conversational, with a slightly lower pitch and a distinct Turkish accent—suggesting this is a real interview moment, captured with emotional authenticity. Her tone carries warmth and personal connection, as if recalling a meaningful memory.  \nShe says: “We fell in love with the breed.”"],
    ["assistant", "text", "<think>\n... (full chain-of-thought reasoning about voice/emotion/pacing planning; truncated here for length, present verbatim in the file) ...\n</think>"],
    ["assistant", "audio", "/mnt/home/jinchuat-andr-d6b58f/jinchuat/espnet_speechlm_tts/egs2/gigaspeech/asr1/data/dev_multi_talker/audio/YOU1000000035/YOU1000000035_M0000024.wav"]
  ],
  "metadata": {"utt_id": "YOU1000000035_M0000024"}
}
```

**One-stream question, answered explicitly:** yes - the multi-talker format puts **all speakers into ONE token stream**. Evidence: every multi-talker entry has exactly **one** `assistant`/`audio` message (one single `.wav` path covering the entire dialogue with both speakers), not one audio message per speaker/turn. Speaker turns are **not** delimited by special tokens, IDs, or structured per-turn fields in the text - they are delimited purely by **natural-language speaker labels embedded in the single `user` caption string**, in patterns like:
- `"Female host: ... \nMale host: ... \n\nThe female host says: \"...\" \nThe male host says: \"...\""` (labeled speaker profile block, then labeled quoted lines)
- `"First speaker: ... \nSecond speaker: ... \nFirst speaker: \"...\" \nSecond speaker: \"...\""` (`multi_talker_tts_POD1000000010_M0000016`)
- `"A British female narrator speaks with... She says: \"...\" Then, a different woman's voice comes in... She says: \"...\""` (descriptive narrative transition, no explicit "Speaker N" label at all)

So there is no fixed delimiter token (no `<sep>`, no `[SPK1]`); the model is expected to infer turn boundaries from free-text description plus quotation marks, and produce one continuous audio/token stream for the whole exchange. This is a significant finding for the TAC branch-exchange design: there is no existing per-speaker channel boundary in the SFT text or audio to hook into - any TAC injection scheme needs to either (a) parse speaker spans out of the caption text itself, or (b) operate on the single combined stream without per-speaker separation at this data layer.

## Model code (Task 3)

Reference branch `whr/speechlm_inference` (remote `whr` = `https://github.com/whr-a/espnet.git`, tip `ae896993f "Add SpeechLM inference code and OpusLM v2 experiment configs"`, 2026-06-12) fetched and read via `git show whr/speechlm_inference:<path>`. `espnet2/speechlm` was then synced into the worktree (see item 5) so all line numbers below are citable directly in the local tree unless noted.

**1. `job_type` value**

`espnet2/speechlm/model/__init__.py:8`: `_all_job_types = {"speechlm": SpeechLMJobTemplate}` - this dict has exactly **one** entry, so any train config that goes through this code path has `job_type: speechlm`. No ambiguity to resolve in Task 4.

Likewise `espnet2/speechlm/model/speechlm/speechlm_job.py:34`: `_lms = {"parallel": ParallelHFModel}` has exactly one entry, so `model.model_choice` must be `parallel`.

Strong circumstantial confirmation (not the BagPiper checkpoint's actual config, but the closest match found in the reference branch - no `bagpiper` string appears anywhere in `whr/speechlm_inference`): `egs2/opuslm_v2/speechlm1/conf/train_stage1_qwen3.yaml` (`git show whr/speechlm_inference:egs2/opuslm_v2/speechlm1/conf/train_stage1_qwen3.yaml`) has:
```yaml
job_type: speechlm
multimodal_io:
    discrete_audio:
        codec_choice: Xcodec
        codec_hf_model_tag: hf-audio/xcodec-hubert-general
        delay_interleave: true
        stream_weights: [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
model:
    model_choice: parallel
    model_hf_tag: Qwen/Qwen3-8B
trainer:
    freeze_param: [multimodal_io_dict.discrete_audio, multimodal_io_dict.continuous_audio, model.layers]
```
This matches the brief's "Qwen3-8B backbone, Xcodec codec, 8 parallel token streams per step" exactly (8-entry `stream_weights` = `codec_max_token_per_frame` default of 8, see item 4). **Candidate for the BagPiper checkpoint's train config, to be confirmed against the actual file in Task 4**, but given `_all_job_types`/`_lms` each have only one entry, `job_type: speechlm` / `model_choice: parallel` are correct regardless of which exact yaml BagPiper used.

**2. Model class, file, and decoder-layers attribute path**

`build_model()` in `espnet2/speechlm/model/speechlm/speechlm_job.py:129-152`:
```python
def build_model(self) -> torch.nn.Module:
    model_config = self.config["model"]
    model_class = _lms[model_config["model_choice"]]          # line 137: ParallelHFModel
    model = model_class(
        model_hf_tag=model_config["model_hf_tag"],
        multimodal_io=self.multimodal_io,
        vocab=self.vocab,
        vocab_intervals=self.vocab_intervals,
        **model_config["model_conf"],
    )
    ...
    return model
```
`ParallelHFModel` (factory function, `espnet2/speechlm/model/speechlm/lm/parallel.py:14-27`) calls `build_parallel_hf_class(model_hf_tag)` which does (lines 30-44):
```python
config = AutoConfig.from_pretrained(model_hf_tag)
architecture = config.architectures[0]              # "Qwen3ForCausalLM" for Qwen/Qwen3-8B
architecture = getattr(transformers, architecture)
class ParallelLLM(architecture):                     # dynamically subclasses transformers.Qwen3ForCausalLM
    ...
```
`model = model_class.from_pretrained(model_hf_tag, **kwargs)` then returns a `ParallelLLM` instance, i.e. **the object `job.build_model()` returns IS a `transformers.Qwen3ForCausalLM` subclass instance directly - there is no wrapping module, no `.corelm` attribute anywhere in this codebase.** `bin/inference.py` confirms this is also the object `load_checkpoint()`/`load_state_dict()` operate on directly (`espnet2/speechlm/bin/inference.py:178-183`):
```python
job_template_class = _all_job_types[train_config["job_type"]]
job_template = job_template_class(train_config, is_train=False)
model = job_template.build_model()
model = load_checkpoint(model, model_checkpoint_path)   # checkpoint["module"] -> model.load_state_dict(..., strict=True)
```
Standard `transformers==4.57.1` `Qwen3ForCausalLM.__init__` (installed at `.../envs/lib/python3.10/site-packages/transformers/models/qwen3/modeling_qwen3.py:436`): `self.model = Qwen3Model(config)`. `Qwen3Model.__init__` (`modeling_qwen3.py:343-345`): `self.layers = nn.ModuleList([Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])`.

**Exact attribute path from the built model object to the decoder-layers `nn.ModuleList`: `<built_model>.model.layers`** (i.e. `job.build_model().model.layers` after checkpoint loading - NOT `model.corelm.model.model.layers`, that path doesn't exist in this codebase). Confirmed independently by the codebase's own trainer: `deepspeed_trainer.py:73-77` freezes parameters by prefix-matching `model.named_parameters()` against strings from `trainer.freeze_param`, and the `train_stage1_qwen3.yaml` config above literally freezes `model.layers` (see item 1) - i.e. the actual training code already treats `model.layers` as the correct parameter-name prefix for the Qwen3 decoder stack. Also confirmed both training (`parallel.py:217`, `forward()`) and inference/generation (`parallel.py:634`, `_step()`) call `self.model(inputs_embeds=..., ...)` - i.e. `.model.layers` is exercised identically on both code paths, so an injection there affects training-loss computation and autoregressive decoding the same way.

**3. Decoder-layer return type and regex-collision risk**

Return type: `transformers.models.qwen3.modeling_qwen3.Qwen3DecoderLayer.forward` (installed 4.57.1, matching the pin in `espnet2/speechlm/requirement.txt: transformers==4.57.1`) is declared `-> torch.Tensor` and literally does `return hidden_states` (a bare tensor, not a tuple) - `modeling_qwen3.py:246-277`. This is the modern (post-refactor) transformers decoder-layer contract; older transformers versions returned `(hidden_states, attn_weights, present_key_value)` tuples, so any wrapper code written against an older transformers assumption will break here. `Qwen3Model.forward` (the caller) does not tuple-unpack the layer's return value either.

Regex collision: **YES, the regex `(?:.*\.)?layers\.(\d+)` matches names outside the target Qwen3 decoder stack.** `named_modules()` on the full BagPiper model (`ParallelLLM` instance) contains at least these additional `...layers.<N>` matches, both verified against the installed `transformers==4.57.1` model classes referenced by the BagPiper-shaped config in item 1. **The two collisions have different certainty - do not treat them as equally established:**
- **Unconditional / high-confidence:** `multimodal_io_dict.discrete_audio.codec_model.semantic_model.encoder.layers.<N>` - the Xcodec codec (`codec_choice: Xcodec` -> `XcodecModel.from_pretrained("hf-audio/xcodec-hubert-general")`, `espnet2/speechlm/model/speechlm/multimodal_io/audio.py:192`) instantiates an internal Hubert-family "semantic" sub-model via `AutoModel.from_config(config.semantic_model_config)` (`transformers/models/xcodec/modeling_xcodec.py:409`; `semantic_model_config` defaults to `HubertConfig()`, `transformers/models/xcodec/configuration_xcodec.py:126-127`). `HubertEncoder.layers` is an `nn.ModuleList` (`transformers/models/hubert/modeling_hubert.py:424`, default 12 layers) - this exists regardless of the IO's own `ssl_choice` setting, since it's internal to the codec checkpoint, not the separate optional SSL tokenizer. **This one is essentially certain for BagPiper**: `discrete_audio` (Xcodec) is the TTS output codec, so any BagPiper train config must include it, and `semantic_model` is always constructed and used by `XcodecModel.encode`/`decode` - not conditional on any BagPiper-specific config choice. This collision alone is sufficient to make the naive regex unsafe.
- **Conditional / lower-confidence:** `multimodal_io_dict.continuous_audio.model.model.layers.<N>` - `ContinuousAudioIO` (used for `audio_input`, per item 1's `preprocessor.audio_input: continuous_audio`) sets `self.model = full_model.thinker` where `full_model` is loaded from `encoder_hf_model_tag: Qwen/Qwen3-Omni-30B-A3B-Instruct` (`espnet2/speechlm/model/speechlm/multimodal_io/audio.py:909`). `Qwen3OmniMoeThinkerForConditionalGeneration.__init__` sets `self.model = Qwen3OmniMoeThinkerTextModel(...)` (`modeling_qwen3_omni_moe.py:1853`), and `Qwen3OmniMoeThinkerTextModel.__init__` sets `self.layers = nn.ModuleList(...)` (`modeling_qwen3_omni_moe.py:1621`) - i.e. an **independent Qwen-family decoder stack with the exact same `model.model.layers.<N>` shape as the injection target**, just under a different top-level prefix. There is also a third, unrelated `layers` ModuleList inside the Omni audio tower (`Qwen3OmniMoeAudioEncoder.layers`, `modeling_qwen3_omni_moe.py:652`). **This collision's presence in BagPiper is unconfirmed**: it comes only from the candidate `opuslm_v2` config (item 1), not BagPiper's actual train config - a pure-TTS setup could plausibly configure `multimodal_io` without a `continuous_audio` entry at all (the only hard requirement, per `speechlm_job.py:_build_vocabulary`, is at least one *discrete* IO), in which case this specific collision would not exist. Task 4 must check the real config's `multimodal_io` keys before relying on this one.

**Consequence: the naive regex is not safe as specified**, and this holds regardless of the `continuous_audio` question above, because the Xcodec/Hubert collision is unconditional. It must be anchored to the specific top-level attribute path (e.g. match only `^model\.layers\.(\d+)$` against the built model's own module names, not a bare `layers\.(\d+)$` suffix match against the whole tree), or the injection registry must explicitly scope the `named_modules()` walk to `built_model.model` before applying the regex. This is a correctness risk that must be fixed in Task 9's injection code, not just noted.

**Caveat on the `model.layers.N` naming itself:** this naming assumes `compile_transformer_body: false` (present in the candidate config, item 1). If a BagPiper config instead sets it `true`, `parallel.py`'s `from_pretrained` runs `model.model = torch.compile(model.model)` (`parallel.py:174-176`), and `torch.compile`'s wrapping renames the submodule path to `model._orig_mod.layers.N` - both the plain attribute path and any anchored regex would need to account for this. Low risk (the candidate config has it `false`) but worth a one-line check in Task 4/9.

**4. Stream 1-7 heads, delay pattern, loss entry point**

There are **no separate per-stream "head" modules** (module names like `stream1_head`, `aux_head_2`, etc do not exist). Instead:
- A single shared `nn.Linear` `lm_head` (rebuilt over the full multimodal vocab, `parallel.py:97-98`, `126-127`) is reused for every stream. Stream 0 (text/special tokens) uses the full `lm_head.weight` (`parallel.py` `_loss`, full-vocab branch). Streams 1+ (audio codebooks) use **the same `lm_head.weight` matrix sliced by vocab interval** (`self.lm_head.weight[start:end].T`, inside `_loss`, streams-1+ branch) - the per-stream separation is achieved purely through non-overlapping vocabulary ranges (`model.loss_intervals`, built in `from_pretrained` around `parallel.py:158-176`), not through separate weight matrices.
- A single `nn.Embedding` `model.stream_emb` of shape `[num_stream, hidden_size]` (`parallel.py:137-138`) is added to the shared transformer's `last_hidden_state` (broadcast per stream, stream 0 forced to zero) before the shared `lm_head` projection, both in `forward()` (`parallel.py:222-226`) and `_step()` (`parallel.py:640-644`). This is the only "per-stream" parameter besides the vocab-interval slicing.
- `num_stream` (= 8 for the candidate BagPiper-shaped config: `codec_max_token_per_frame: int = 8` default, `multimodal_io/audio.py:78`) is computed from `DiscreteAudioIO.num_stream()` = `self.ssl_n_streams + self.codec_n_streams` (`multimodal_io/audio.py:701-705`).

8-stream delay pattern: implemented entirely in the **data/IO layer**, not the model - `DiscreteAudioIO._apply_delay_interleave` / `_apply_delay_deinterleave` (`multimodal_io/audio.py:737-789`), gated by the `delay_interleave: bool` config flag (`audio.py:84`, `115`). Progressive per-stream delay: "Stream 0: no delay; Stream 1: delayed by 1 frame; Stream 2: delayed by 2 frames, etc." (docstring + implementation, `audio.py:737-761`); applied to the encoded codes tensor `[batch, time, n_streams]` before it enters the multi-stream token sequence (`encode_batch`, `audio.py:471-472`), and reversed on decode (`decode_batch`, `audio.py:506-508`).

Loss computation entry point: `ParallelLLM._loss(self, hidden_states, input_ids, loss_mask, router_logits)` (`parallel.py:338` onward), called from `forward()` at `parallel.py:229-234`, right after the `self.model(...)` call (`parallel.py:217-220`) and stream-embedding addition (`parallel.py:222-226`). `forward()` itself (`parallel.py:194-234`) is the outer entry point invoked by the training loop (`model(**batch)` returns `{"loss": ..., "stats": ...}`, `parallel.py:234`).

**5. `espnet2/speechlm` diff vs our branch, and sync decision**

`git diff --stat HEAD whr/speechlm_inference -- espnet2/speechlm` (before syncing) showed 31 files changed, 1437 insertions(+), 3493 deletions(-), entirely confined to `espnet2/speechlm` (no other paths touched - the diff command was already scoped there). Investigation of branch history: `git merge-base HEAD whr/speechlm_inference` = `6ce825d` (2026-02-24, a `whr-a/ci_test` PR merge). Our branch's last commit touching `espnet2/speechlm` was `83702b8` (2026-06-15, "move `audio_tokenizer.py` to `espnet2/beats/`" - a housekeeping move, not speechlm development). `whr/speechlm_inference`'s tip, `ae896993f` (2026-06-12, "Add SpeechLM inference code and OpusLM v2 experiment configs"), is the commit that *introduced* the `job.build_model()` / clean-`ParallelHFModel` / `bin/inference.py` API this whole plan depends on. **Our worktree branch predates and lacks that inference-focused rewrite** - its own `lm/parallel.py` (669 lines, pre-sync) has a materially different `from_pretrained` signature (`vocab_meta` dict instead of separate `vocab`/`vocab_intervals` args, a `fused_cross_entropy_loss` import, flash-attention assertions, z-loss) and its `espnet2/speechlm` tree additionally carries pipeline-parallel/Titan-trainer machinery (`lm/parallel_pp.py`, `trainer/titan_trainer.py`, `trainer/titan_trainer_pp.py`, `parallel_utils/*`) that `whr/speechlm_inference` deleted entirely - i.e. our branch is a divergent, training/pipeline-parallel-oriented line, not simply an older copy of the same file.

**Decision: synced.** Ran `git checkout whr/speechlm_inference -- espnet2/speechlm` (scoped exactly to `espnet2/speechlm`, no other paths touched). This updated 21 tracked files (`git diff --cached --stat`: 1437 insertions(+), 644 deletions(-)) and added 9 new files (`README.md`, `__init__.py`, `bin/modify_parquet.py`, `bin/prepare_audio_arkive.py`, `moe_utils/launch_test.sh`, `moe_utils/replace_moe_layer.py`, `requirement.txt`, `trainer/sample_deepspeed_config.json`, `utils/parquet_dump.py`). Verified post-sync: `git show whr/speechlm_inference:espnet2/speechlm/model/speechlm/lm/parallel.py | diff - espnet2/speechlm/model/speechlm/lm/parallel.py` is empty (identical). Files that exist only on our branch and were **not** touched by the scoped checkout (left in place deliberately, since removing them would break existing unit tests that still reference them - `test/espnet2/speechlm/model/test_parallel_utils.py`, `test/espnet2/speechlm/model/speechlm/lm/test_parallel_pp.py`, `test/espnet2/speechlm/model/speechlm/lm/test_loss.py`, `test/espnet2/speechlm/trainer/test_titan_trainer.py`, `test/espnet2/speechlm/trainer/test_titan_trainer_pp.py`): `lm/loss.py`, `lm/parallel_pp.py`, `parallel_utils/__init__.py`, `parallel_utils/grouped_moe.py`, `parallel_utils/parallel_dims.py`, `parallel_utils/pipeline.py`, `parallel_utils/qwen3.py`, `tokenizer/abs_tokenizer.py`, `trainer/titan_trainer.py`, `trainer/titan_trainer_pp.py` - these are now orphaned/unreferenced by the synced code path (confirmed `speechlm_job.py` imports only `from espnet2.speechlm.model.speechlm.lm.parallel import ParallelHFModel`, not `parallel_pp`) but harmless to leave since nothing in the synced path imports them.

**Risk this sync creates for Task 4 (flag, not yet resolved):** our pre-sync `lm/parallel.py` had a materially different `from_pretrained` signature and forward/loss internals (see divergence description above). Swapping to whr's version bets that the **parameter names** whr's `ParallelLLM.from_pretrained` produces match the state dict keys actually saved in the BagPiper DeepSpeed checkpoint - `bin/inference.py`'s `load_checkpoint()` calls `model.load_state_dict(state_dict, strict=True)` (`espnet2/speechlm/bin/inference.py:141`), which raises on any key mismatch. This bet is well-motivated (the plan's `job.build_model()` API was explicitly "confirmed from the reference inference script", i.e. this synced code), but it is unverified until Task 4 actually loads the checkpoint. If `strict=True` load fails on key mismatch, that reopens this sync decision (e.g. the checkpoint might have been trained against a parallel.py revision between our old branch and whr's tip) - it is not just a matter of picking the right train-config yaml.
## Checkpoint (Task 4)

**Download.**
Source: HF repo `wanghaor/transfer`, file `bagpiper_vLLM_all/bagpiper_vLLM_all.tar`, downloaded with `hf download` to `downloads/bagpiper_vLLM_all/bagpiper_vLLM_all.tar`.
Local size 18,200,709,120 bytes, byte-exact match with the HF LFS metadata (`lfs.size: 18200709120`, `lfs.oid sha256: 4982810167de18e8e279951f95e6815b04e86557a2c881ee00a958a0abd3b2e0`).
Extracted to `downloads/bagpiper/`.

**Tar contents inventory (complete, 18 files).**

| Path in tar | Size | What it is |
|---|---|---|
| `./convert_ckpt.py` | 7,921 B | ESPnet DeepSpeed `.pt` -> vLLM/HF safetensors converter (guidance goldmine, see below) |
| `./env.md` | 608 B | Setup guide: vLLM fork install + pointers to the two scripts |
| `./serve_cfg_1.sh` | 2,563 B | vLLM OpenAI-API server launch script (CFG-enabled) |
| `./client_all.py` | 17,810 B | Stress-test client; documents request format and sampling flags |
| `./speechlm-qwen3-8b/config.json` | 1,809 B | vLLM model config - **the token-layout ground truth** |
| `./speechlm-qwen3-8b/model-0000{1..4}-of-00004.safetensors` | 5.36+5.30+4.55+2.98 GB | The model weights, 4 BF16 shards |
| `./speechlm-qwen3-8b/model.safetensors.index.json` | 162,892 B | Shard index; full state-dict key list |
| `./speechlm-qwen3-8b/{generation_config,added_tokens,special_tokens_map}.json` | small | HF tokenizer/generation config |
| `./speechlm-qwen3-8b/{tokenizer.json,tokenizer_config.json,vocab.json,merges.txt}` | ~12 MB | Qwen2-family BPE tokenizer files |
| `./speechlm-qwen3-8b/chat_template.jinja` | 1,003 B | Documents the exact ESPnet training prompt format |

**Headline finding: the tar contains NO DeepSpeed `.pt` checkpoint and NO train config yaml.**
It is a vLLM deployment bundle, not an ESPnet training checkpoint.
The expected `train_stage2_qwen3_base_v3.yaml` (reference-launch naming) is absent; there is no yaml of any kind in the tar.
A repo-wide search of all 11,838 files in `wanghaor/transfer` found no ESPnet train config either - the only yamls are dataset-registry files (`inference_data/sft.yaml`, `jctian_data/sft.yaml`, both just name->jsonl-path maps) and an unrelated olmo2 config under `siddhant_dump/`.
**Concern: Task 5 cannot read the train config from disk; it must be reconstructed** (starting point: `egs2/opuslm_v2/speechlm1/conf/train_stage1_qwen3.yaml` from Task 3, validated against the `config.json` facts below).

**But the safetensors ARE the raw ESPnet state dict.**
`convert_ckpt.py` docstring (verbatim): "Convert an ESPnet SpeechLM DeepSpeed checkpoint to vLLM-compatible format. Takes the mp_rank_00_model_states.pt from an ESPnet training run and produces a complete HuggingFace-style checkpoint directory with sharded safetensors".
Its `load_checkpoint()` does `ckpt = torch.load(input_path, ...)`; `state_dict = ckpt["module"]` ("Extracted 'module' key from DeepSpeed checkpoint.") and then shards that dict **without renaming a single key**.
So the shard keys are exactly the keys `ParallelLLM.load_state_dict(..., strict=True)` expects, minus the `["module"]` wrapper.
This also confirms the source checkpoint's top-level layout: a DeepSpeed dict whose `"module"` key holds the model state dict (per the script's own branch logic).

**State-dict inventory** (from `model.safetensors.index.json` + safetensors headers; no torch load needed):
- 1,381 tensors, 9,093,843,593 parameters total, every tensor BF16, `metadata.total_size` 18,187,687,186 bytes (16.94 GiB).
- First 20 keys: `model.embed_tokens.weight`, then `model.layers.0.{self_attn.{q,k,v,o}_proj.weight, self_attn.{q,k}_norm.weight, mlp.{gate,up,down}_proj.weight, input_layernorm.weight, post_attention_layernorm.weight}`, then the same pattern for `model.layers.1.*`.
- Weight groups (prefix: tensor count): `model.layers` 396 (= 36 decoder layers x 11), `model.embed_tokens` 1, `model.norm` 1, `lm_head.weight` 1, `stream_emb.weight` 1, `adaptor.continuous_audio` 2, `multimodal_io_dict.continuous_audio` 525, `multimodal_io_dict.discrete_audio` 454.
- Shape spot-checks: `model.embed_tokens.weight` and `lm_head.weight` both `[160392, 4096]`; `stream_emb.weight` `[8, 4096]` (8 streams confirmed in the weights themselves); `adaptor.continuous_audio.weight` `[4096, 2048]` + bias `[4096]` (the 2048->4096 continuous-audio linear adaptor from `parallel.py:143-146`).
- Keys use `model.layers.N`, not `model._orig_mod.layers.N`, so the checkpoint was saved with `compile_transformer_body: false` semantics (Task 3 caveat resolved).

**Regex-collision question from Task 3, now settled against the real checkpoint.**
`continuous_audio` IS present in BagPiper (525 tensors), but it contains **only** `multimodal_io_dict.continuous_audio.model.audio_tower.*` - no thinker text decoder.
This matches `ContinuousAudioIO._init_encoder` (`espnet2/speechlm/model/speechlm/multimodal_io/audio.py`, "Remove unnecessary components, keep only audio tower": `del full_model.thinker.model`, `del full_model.thinker.visual`, `del full_model.thinker.lm_head`).
The actual `.layers.<N>.` collisions in this checkpoint are exactly two: `multimodal_io_dict.continuous_audio.model.audio_tower.layers.<N>` (32 layers, Whisper-style encoder) and `multimodal_io_dict.discrete_audio.codec_model.semantic_model.encoder.layers.<N>` (Hubert).
Task 3's hypothesized `continuous_audio.model.model.layers.<N>` collision does not exist; the anchored-regex fix (`^model\.layers\.(\d+)$`) remains required and sufficient.

**Guidance facts from `config.json` (verbatim, the token-layout ground truth):**
```json
"num_stream": 8,
"vocab_size": 160392,
"codec_base_offset": 152192,
"codec_layer_size": 1025,
"text_token_offset": 256,
"text_token_end": 152192,
"xcodec_hf_model_tag": "hf-audio/xcodec-hubert-general",
"xcodec_sample_rate": 16000,
"pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2, "eot_token_id": 3,
"system_token_id": 4, "user_token_id": 5, "assistant_token_id": 6,
"text_token_id": 7, "audio_token_id": 8
```
This confirms the design doc's backbone facts exactly: 8 streams, audio-vocab offset 152192, per-stream stride 1025, and 152192 + 8 x 1025 = 160392 = vocab_size.
Sample rate is 16 kHz (Xcodec).
Text tokens occupy `[256, 152192)`; ids 0-8 are the special/control tokens listed above; sampling defaults baked into the config: `"audio_temperature": 0.8, "audio_topk": 20, "text_temperature": 0.6, "text_topk": 20`.
Backbone dims (in `text_config`): `hidden_size 4096, num_hidden_layers 36, num_attention_heads 32, num_key_value_heads 8, head_dim 128, intermediate_size 12288, rope_theta 1000000.0` - Qwen3-8B, matching the 36 `model.layers` in the state dict.

**Guidance facts from `chat_template.jinja` (verbatim comment):**
"Prompt format (from ESPnet training): `<|bos|>[<|system|><|text|>sys<|eos|>]<|user|><|text|>text<|eos|><|assistant|>` / `<|bos|>[<|system|><|text|>sys<|eos|>]<|user|><|audio|><|eos|><|assistant|>`" - directly usable for Task 6's teacher-forced sequence construction.

**Guidance facts on inference flags (`serve_cfg_1.sh` + `client_all.py`):**
- Server: `--max-model-len 16384`, `--trust-remote-code`, `--limit-mm-per-prompt '{"audio": 1}'`; "CFG is triggered per-request when the client sends `"cfg": N` (N > 1) in vllm_xargs".
- Audio-gen request: `vllm_xargs: {"mode": "text_audio", "phase": "text", "cfg": 3.0, "text_temperature": 0.6, "audio_temperature": 0.8, "audio_topk": 20}`, `max_tokens` up to 12000.
- Text-only request: `"stop_token_ids": [3]` (stop on `eot_token_id` 3).
- `env.md`: inference uses a **forked vLLM** (`git clone https://github.com/whr-a/vllm.git`, python 3.11, `VLLM_USE_PRECOMPILED=1 uv pip install --editable .`, `pip install vllm[audio]`) - not needed for our ESPnet-side gate, but it is the only supported serving path shipped with the tar.

**Raw DeepSpeed `.pt` checkpoints DO exist elsewhere in the same HF repo** (not in the tar; found while searching for the train config):
- `checkpoints/global_step267558/mp_rank_00_model_states.pt` (18,188,293,711 B)
- `checkpoints/global_step299985/mp_rank_00_model_states.pt` (18,188,293,839 B)
- `ckpt/step_275140/global_step275128/mp_rank_00_model_states.pt` (18,188,293,711 B)
- plus optimizer-state-only dirs (`checkpoints/step_150000/`, `ckpt/v1/step_275000/`, `ckpt/v2/opuslm_v2_stage3_sft_qwen3_combine_v2_3node/step_272500/` - that last run name, and `convert_ckpt.py`'s example output `speechlm-qwen3-8b-step272500`, are the best available hints at the training run identity).
It is not recorded which `.pt` the tar's safetensors were converted from.

**Task 5 loader recipe implied by this inventory** (no `.pt` download needed):
```python
import glob
from safetensors.torch import load_file
sd = {}
for shard in sorted(glob.glob("downloads/bagpiper/speechlm-qwen3-8b/model-*.safetensors")):
    sd.update(load_file(shard))
model = job_template.build_model()
model.load_state_dict(sd, strict=True)
```
i.e. replace `bin/inference.py`'s `torch.load(...)["module"]` with a shard merge; everything downstream is unchanged because the keys are identical.
Alternatively download one of the raw `.pt` files above (~18 GB more) to use `load_checkpoint()` verbatim.
**Second Task 5 concern:** `build_model()` with a `continuous_audio` IO calls `ContinuousAudioIO._init_encoder`, which does `from_pretrained("./Qwen-Qwen3-Omni-30B-A3B-Instruct")` - a local directory holding the full ~30B Qwen3-Omni model (only the audio tower is kept, but the full model is loaded first).
For a TTS-only teacher-forced gate it may be preferable to reconstruct the train config **without** `continuous_audio` and strict-load only the remaining keys (or load with a filtered state dict), rather than fetch ~60 GB of Omni weights; this is a decision for Task 5, flagged here.

**Train config used: none shipped.** Must be reconstructed; `config.json` above is the validation target for any reconstruction.
## Gate results (Tasks 5-7)

**CONDITIONAL GO** - espnet3 route confirmed viable on all software criteria; final sign-off pending the two PSC items below. Phase 1 (tiny-model work) proceeds; Phase 2+ (training) must not start until both PSC items pass.

### Criteria PASSED locally

1. **SFT schema documented** (Task 2): dataset.json/dialogues.jsonl structure, field semantics, and multi-talker one-stream layout fully inventoried and exemplified.
2. **Decoder path + return type + collision analysis** (Task 3): `model.layers.<N>` identified as `Qwen3DecoderLayer` (bare-tensor return), regex-collision risk catalogued (Xcodec/Hubert and optional Continuous-Audio layers), fix specified (`^model\.layers\.(\d+)$` anchoring).
3. **Checkpoint inventoried, config.json facts confirmed** (Task 4): 1,381 tensors total; 854 retained (continuous_audio excluded); vocab 160392, 8 streams, codec stride 1025, text offset 256, sample rate 16 kHz all validated. Synced whr/speechlm_inference code produces state-dict keys matching checkpoint names+shapes exactly (strict=True loading will succeed).
4. **Reconstructed train config validated, strict two-way load coverage** (Task 5): config.json facts (vocab_size, num_stream, offsets) match rebuilt YAML; verification script confirmed 854 expected names+shapes == 854 retained checkpoint keys, 0 missing/mismatched/unexpected. Continuous_audio excluded by design (INPUT codec only, unused for TTS teacher-forcing).
5. **Real preprocessor batch builds successfully** (Task 5): one dev_multi_talker sample hand-assembled, passed through `SpeechLMPreprocessor.collate_fn`, produced valid seqs (1, 2511, 8) int64 with delay-interleaved 8-stream layout and loss masks. Batch construction end-to-end verified.

### Criteria DEFERRED to PSC (with exact commands)

1. **Teacher-forced loss** (Task 5/6 combined): Run `PYTHONPATH=<espnet_bagpiper worktree>:<this recipe dir> python scripts/gate_teacher_forced.py` on a machine with >=24 GiB RAM (or any CUDA GPU). This machine's 16 GiB RAM is insufficient (retained model 16.88 GiB BF16 + intermediate allocations push peak to ~30 GiB). Expect finite low-single-digit cross-entropy loss if training converged correctly. Watch for bf16-codec-vs-float32-wav dtype clash in encode_batch; if it fires, cast the wav to the codec dtype in the gate script, not in espnet2/.

2. **Single-channel generation** (Task 6, to be written): Create `scripts/gate_generate.py` using `model.prepare_inference()` / `model.inference()` per the reference `inference.py`. Audio generation parameters: temperature 0.8, topk 20, CFG 3.0, max_step 1024. Produce one wav file covering the full dialogue with all speaker turns in a single audio stream (as trained). Then run whisper-transcribe to validate speaker content against the original caption. This script does not exist yet; implement during the PSC session.

3. **Zero-init TAC parity on the real checkpoint** (Task 12): `tests/test_pretrained_real.py` is asset-gated (skips locally; the 16 GiB dev machine cannot hold the ~16.9 GB bf16 model plus activations, same constraint as item 1 above). Run on the PSC box with:
   `BAGPIPER_CKPT=<abs path to the speechlm-qwen3-8b shard directory> PYTHONPATH=<espnet_bagpiper worktree>:<this recipe dir> python -m pytest tests/test_pretrained_real.py -v`
   `BAGPIPER_TRAIN_CONFIG` is optional and defaults to `conf/bagpiper_train_config.yaml` (committed) when unset. Expect 1 passed: the real 36-layer/4096-hidden model with `TACExchange` injected at depths 19-36 (zero-init gates) must reproduce the un-injected forward's pre-`lm_head` hidden state bit-exactly on a real `dev_multi_talker` sample duplicated into 2 branches (`counts=[2]`). Compares hidden states rather than full-vocab logits deliberately: with vocab 160392, `lm_head` on the (2, 2511, 8, *) hidden state would materialize ~12.9 GB of bf16 logits per call (~26 GB held for both sides of `torch.equal`, on top of the ~16.9 GB model) - exactly the blowup `_loss`'s interval-based softmax exists to avoid; `lm_head` is a fixed matmul independent of the injection, so hidden-state bit-exactness implies logit bit-exactness a fortiori. This test reuses `build_batch` from `scripts/gate_teacher_forced.py` unmodified (already top-level importable, so no edits were needed there) and shares its bf16-codec-vs-float32-wav landmine (item 1 above) since it drives the same `_embed` path - a PSC failure at the codec-encode step should be read as that landmine firing, not as a parity-logic bug.

**Loss value:** NOT MEASURED on this hardware (Task 5 amendment: RAM-blocked). All non-loss aspects of the gate verified above.

### Deferred criteria: RESULTS (Delta run, 2026-07-14)

The three deferred items were executed on NCSA Delta (A40/CPU compute nodes, interactive partitions; full run log in the controller-side report `delta-gate-report.md`).

1. **Teacher-forced loss: PASS.** `loss = 2.336375` on one real dev_multi_talker batch (A40, bf16) - finite low-single-digit CE with the expected texture (stream-0 text acc 0.43 >> audio-stream acc ~0.22-0.28).
   Both predicted dtype landmines fired and are fixed IN THE GATE SCRIPT via the reference pipeline's own `to_device(batch, device, dtype=bf16)` (casts float tensors only): float32 wav vs bf16 codec conv, and float64 `loss_masks` vs bf16 CE in `_loss`'s `masked_scatter_`.
2. **Zero-init TAC parity on the real checkpoint: PASS.** `1 passed` in 38 min (CPU, bf16): injecting `TACExchange` at depths 19-36 (zero gates) into the real 36-layer model changes not a single bit of the pre-`lm_head` hidden states.
   The run first exposed a REAL package gap: `inject_exchange`'s factory-built exchanges are float32 and crashed the bf16 backbone (`mat1 and mat2 must have the same dtype`). Root cause fixed in `src/branch_exchange/inject.py::_call_exchange` (activations adapt to the exchange dtype at the block boundary; bit-exact at zero gate by construction) with bf16 tiny-model regression tests; the Delta PASS itself used the interim `model.to(bf16)` workaround, so re-run this test on Delta once to confirm the mixed-dtype path on the real checkpoint.
3. **Single-channel generation: OPEN - degenerate output, checkpoint NOT impugned.** `scripts/gate_generate.py` (committed) runs the full pipeline (prompt `(1, 210, 8)`, `prepare_inference`, `model.inference`, wav writeout), but the espnet incremental-decode path emits a degenerate `<think>` text segment (repeated punctuation, no `<|eot|>`), so no audio segment is ever produced. Ruled out: prompt/input format, sampling (greedy reproduces it), CFG (text segment decodes with cfg=1), detokenization (control flow branches on raw ids). The failure is localized to the whr-synced generation machinery (`prepare_inference` + `inference_segment`/`_step` in `espnet2/.../lm/parallel.py`) - the one path the passing items never exercise - and the checkpoint's supported serving path is the forked vLLM, not this espnet script. Forward weights are proven good by items 1-2.
   **Follow-up options:** (a) verify generation through the vLLM fork (the supported path), or (b) diff `prepare_inference`/`inference_segment` config faithfulness against this checkpoint's config.json before trusting the espnet decode path. Phase 2+ SFT does not depend on the espnet decode path (training is teacher-forced), but the POC's lockstep inference loop will need this resolved or routed around.

**Gate verdict update (2026-07-14): criteria 1-2 of the CONDITIONAL GO are now MET; generation moves from "deferred" to "open with diagnosis" (see item 3). Phase 2 data/trainer work is unblocked; the step-4 lockstep inference loop must resolve item 3 first.**

### Generation fix + final verification (2026-07-14, second Delta pass)

The two generation root causes were fixed on the branch (commit `55d55be82`, `espnet2/speechlm/model/speechlm/lm/parallel.py`):
(1) `_step` skipped `_embed`'s stream>0 pad-embedding zeroing while the checkpoint's row-0 embedding is nonzero noise, corrupting every decode step's input (now shared via `_embed_and_sum_streams`);
(2) continuation segments re-prefilled the whole prompt onto the accumulated KV cache (now they inject only the `<|assistant|>` token).
Regression tests: `tests/test_parallel_inference.py` on a tiny ParallelLLM fixture reproducing the noisy-pad-row condition; both RED on pre-fix code.

Delta verification (all under interactive partitions, logs in /work/nvme/bbjs/ttrachu/):
1. Unit tests + full suite: 2 passed; 73 passed + 1 skipped.
2. Teacher-forced loss re-check: **exactly 2.336375** - the `_embed` refactor changed no numerics.
3. Generation: **the loop is repaired** - coherent 5144-char `<think>` plan ending in `<|eot|>`, audio segment decodes to 16.5 s / RMS 0.154 wav; whisper transcript's first ~5-6 s match the script ("Later that week I stopped by to meet a very special lady indeed. Sheila Reed...").
4. **Residual open item - long-horizon audio drift:** intelligibility degrades after ~5-6 s into garble-with-repetition. cfg=1 control is WORSE (degrades at ~2 s), so CFG is exculpated and actually stabilizes decode. Sampling params match the author's vLLM client exactly (survey: espnet path was never used by the author; serving is 100% the vLLM fork; no generated samples exist on their disk to benchmark). A/B test through the author's vLLM path on the same prompt is the decisive discriminator (espnet-decode divergence vs model free-running limit) - results to be appended.
5. Mixed-dtype parity re-run: **1 passed (bit-exact)** with exchanges deliberately fp32 on the bf16 backbone (no `model.to(bf16)` workaround) - the `_call_exchange` activation-casting path is confirmed on the real checkpoint.

**Gate verdict (final for steps 0-1): GO.** All CONDITIONAL GO criteria are met; the generation machinery is verified working end-to-end. The long-horizon drift is tracked as its own open item above (relevant to the step-4 lockstep loop; per-turn spans in conversational windows are typically shorter than the drift onset, and the TAC fine-tune retrains the decode regime on our windows regardless).

### A/B drift test: verdict B - model free-running limit, NOT an espnet-decode bug (2026-07-14)

The same sample was generated through the author's supported vLLM path (their SIF + checkpoint, verbatim client payload; prompt-token parity confirmed: 211 vs our (1,210,8)).
Result: identical drift shape - intelligible ~3-6 s, then garble; both independent decoders even converge on the SAME "if there was..." phrase-loop attractor (likely latching onto the plan's last line).
Both vLLM draws ended on natural `<|eos|>` (30.0 s and 23.5 s, `finish_reason=stop`), so the horizon is not a length-cap artifact.
Conclusion: the espnet decode path (with the two fixes) matches the author's shipping path; do NOT chase decode-parity bugs.
Productive levers are model/data-level: shorter target segments, sampling/guidance changes, or fine-tuning - which the POC's SSSD fine-tune does anyway, and conversational per-turn spans are mostly below the ~6 s onset.
Full writeup: controller-side `vllm-ab-test.md`; outputs under /work/nvme/bbjs/ttrachu/ab_test/.

### Task 5: load helper + teacher-forced gate

**Deliverables** (all committed; nothing under `downloads/`):
- `conf/bagpiper_train_config.yaml` - reconstructed train config (no yaml shipped with the checkpoint).
- `scripts/load_bagpiper.py` - `load_bagpiper(train_config_path, ckpt_path, device="cpu", dtype=torch.bfloat16)`, the Interfaces contract imported by Task 6 / Task 12. Loads the 4 BF16 safetensors shards (raw ESPnet keys, no `["module"]` wrapper), filters the excluded prefixes, asserts strict two-way coverage, then `load_state_dict(strict=True)`. `ckpt_path` is the shard DIRECTORY; a DeepSpeed `.pt` path (state dict under `"module"`) is also supported for future use.
- `scripts/verify_key_coverage.py` - RAM-free proof of the strict-coverage contract (runnable here; see below).
- `scripts/gate_teacher_forced.py` - the teacher-forced gate; `--build-only` builds the batch RAM-free.

**Config reconstruction.** Started from `whr/speechlm_inference:egs2/opuslm_v2/speechlm1/conf/train_stage2_qwen3_base_v3.yaml` (the SFT-stage reference). Base-vs-Instruct resolved from the shipped `tokenizer_config.json` -> `base_tokenizer: "Qwen/Qwen3-8B-Base"`, so `model_hf_tag` / `tokenizer_name` = `Qwen/Qwen3-8B-Base` (matches that reference; the stage1/stage2 non-base configs use `Qwen/Qwen3-8B` and are wrong for this checkpoint). Adaptations vs the reference candidate:
  - `continuous_audio` IO **removed** (see exclusion below); `preprocessor.audio_input: continuous_audio -> discrete_audio`.
  - `attn_implementation: flash_attention_3 -> sdpa` (no flash on CPU/MPS).
  - `activation_checkpointing: true -> false`, `audio_cfg: 0.05 -> 0.0` (deterministic gate).
  - `trainer`/`deepspeed`/`freeze_param` dropped (not read by `build_model()`).
Validated against `config.json`: built vocab = 256 special + 151936 text + 8*1025 codec = **160392** = `vocab_size`; text interval `[256, 152192)`, codec `[152192, 160392)` -> `codec_base_offset` 152192, `num_stream` 8, `codec_layer_size` 1025. All match.

**Exclusion (continuous_audio).** `build_model()`'s IO dict is config-driven (`SpeechLMJobTemplate.__init__` builds `multimodal_io` from `config["multimodal_io"]`), so dropping the `continuous_audio` key cleanly prevents constructing `ContinuousAudioIO` - which otherwise pulls `Qwen/Qwen3-Omni-30B-A3B-Instruct` (~60 GB) and is audio-INPUT only, unused for TTS teacher-forcing. No `espnet2/` change needed. Consequence for loading: checkpoint tensors under `multimodal_io_dict.continuous_audio.*` (525) and `adaptor.continuous_audio.*` (2) are excluded. **Excluded = 527 tensors; retained = 854 of 1381.**

**Strict-coverage verdict (verified RAM-free, names + shapes).** `scripts/verify_key_coverage.py` reconstructs the exact `{key: shape}` map `build_model()` would produce, deriving every size from the reconstructed config itself (not hardcoded): it instantiates `SpeechLMJobTemplate` (RAM-cheap - builds only the small IOs + the unified vocab, NOT the 16 GB transformer), asserts `len(job.vocab) == 160392` and `discrete_audio.num_stream() == 8` and the text/codec intervals, then builds the Qwen3ForCausalLM backbone via `torch.device("meta")` at that vocab + `stream_emb.weight [8,4096]` + the REAL small `DiscreteAudioIO` state-dict shapes prefixed `multimodal_io_dict.discrete_audio.` + the REAL `HuggingFaceTextIO` (0 persistent keys). Compared to the checkpoint's own names+shapes (safetensors headers) minus excluded prefixes. Result: **854 expected == 854 retained, 0 missing, 0 unexpected, 0 shape mismatches, both directions -> PASS.** Shapes matter because `load_state_dict` raises on shape mismatch independent of `strict` (e.g. a wrong `codec_max_token_per_frame` would give an identically-named `stream_emb.weight` of the wrong shape). So the whr-synced `parallel.py` parameters match the BagPiper checkpoint exactly in name AND shape (Task-3 sync risk resolved), and Task 6 / Task 12 can import `load_bagpiper` and get correct, complete coverage on adequate RAM. (Breakdown: backbone 399 = embed_tokens + norm + lm_head + 36 layers x 11; stream_emb 1; discrete_audio 454; text 0.)

**Batch construction (verified RAM-free, `--build-only`).** Per amendment 5, the gate drives the real synced ESPnet preprocessor (`SpeechLMJobTemplate.build_preprocessor()` + `SpeechLMPreprocessor.collate_fn`) rather than `DataIteratorFactory` (no registered datasets locally). One `dev_multi_talker` sample (`multi_talker_tts_YOU1000000035_M0000024`) is hand-assembled into the preprocessor's `{"dialogue": [[role, io_name, content], ...]}` form: SFT modality `text -> "text"`, `audio -> "discrete_audio"` (the stale wav path resolved by basename to the shipped `downloads/.../audio/.../*.wav`, loaded as `[channels, samples]`), `is_train=True` so the assistant turns get loss mask. The genuine preprocessor then tokenizes, lays out the 8 delay-interleaved streams, and builds the loss mask. Produced batch: `seqs (1, 2511, 8) int64`, `loss_masks (1, 2511, 8)`, `discrete_audio_indices (1, 3)`, `discrete_audio_feats (1, 409808, 1) float32` (~25.6 s wav; Xcodec-encoded inside the model's `_embed`), `discrete_audio_lengths (1,)`. Batch build succeeds end-to-end.

**Landmine flagged for Task 6 / 12 (first real forward).** `load_bagpiper` does `model.to(dtype=bfloat16)`, which also casts the Xcodec codec inside `discrete_audio`. But `discrete_audio_feats` (raw wav) enters `encode_batch` as float32, so the first real forward may hit a bf16-codec-vs-float32-input dtype clash (or slow/unsupported bf16 CPU codec ops). If so, cast the wav to the codec dtype at the `encode_batch` call site (in the gate / caller, not in `espnet2/`), or keep the codec in float32. Also `loss_masks` come out float64 (numpy default from python-float `stream_weights`); harmless (upcast in `_loss`) but worth a glance.

**Loss value: BLOCKED on this hardware (RAM).** The retained model is **16.88 GB in bf16** (8.44 B params); this machine has **16 GiB (17.18 GB) physical RAM** with ~8-9 GB free. The final model alone exceeds physical RAM, and `ParallelLLM.from_pretrained` additionally loads the full Qwen3-8B-Base base weights (~15 GB, overwritten by the checkpoint) before the shard load, so peak is ~30 GB -> the machine would thrash/swap. Per amendment 4 this is reported rather than attempted (running it risks destabilizing the machine Claude Code itself runs on; `flash_attention` is also unavailable on CPU, and MPS shares the same 16 GiB pool with a working-set cap that rejects a 15.7 GiB allocation). The numeric loss must be produced on a box with >=~24 GB RAM (or any CUDA GPU); `python scripts/gate_teacher_forced.py` will then print it with the coverage assertion already baked into `load_bagpiper`. All non-loss aspects of Task 5 (config validity, strict coverage, batch pipeline) are verified above.
