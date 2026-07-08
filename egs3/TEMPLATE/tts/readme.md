# ESPnet3 TTS template

Shared default configs merged under every TTS recipe's configs by
`load_and_merge_config` (see e.g. `egs3/libritts/tts/run.py`).

Unlike the ASR template there is no runnable `run.py` here: a TTS recipe
directory may host several mutually incompatible architectures (e.g. VITS
vs. F5-TTS have unrelated `model.tts_conf` shapes), so these defaults are
kept as empty as possible.
Only generic path/system scaffolding lives here; everything
architecture-specific is left null for the recipe config to fill in.
