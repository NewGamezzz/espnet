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

3. Role-play style (`advanced_tts/role_play`, `adv_role_play_001`):
   > "Khariton Volkov, a battle-scarred Siberian veteran of the Great Game, now a railway foreman in 1880s St. Petersburg, recounts his harrowing escape from a Cossack ambush.\n\"I slipped through the snowdrifts as gunfire cracked, my heart pounding like the drums of war, and the frost bit my fingers as I vanished.\""

4. SVS style (`advanced_tts/svs`, `adv_svs_001`):
   > "Give me a rendition of a playful yodeling opera carol, bright and silly. \"Jingle bells on high, echo through the snowy hall, yodeling notes cascade, merry voices rise and fall.\""

5. Real multi-talker training caption (`dev_multi_talker`, `multi_talker_tts_YOU1000000040_M0000001`):
   > "Female host: bright, upbeat, and energetic with a higher pitch\nMale host: warm, deep, and enthusiastic with a resonant tone\nNarrator: calm, clear, and professional with a neutral delivery\n\nThe female host says: \"We're in Istanbul, Turkey.\"\nThe male host says: \"The food culture here will blow your mind. We're so excited to be filming a bunch of food videos.\"\nThe female host says: \"Let's get food hunting.\"\nThe narrator says: \"Istanbul has an exciting and vibrant food scene. This city's diverse heritage is reflected in its incredible food culture. In this five-part series, we're going to show you some delicious local Turkish food.\""

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
- job_type value:
- model class + file:
- decoder layers attribute path:
- block return type (tuple or tensor):
- streams 1-7 head module(s):
- loss computation entry point:
- espnet2/speechlm diff vs our branch (merge decision):
## Checkpoint (Task 4)
- tar contents inventory:
- train config used:
## Gate results (Tasks 5-7)
- teacher-forced loss value:
- single-channel generation reproduced (y/n + wav path + transcript):
- GO / NO-GO espnet3:
