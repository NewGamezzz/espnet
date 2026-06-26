"""Recipe-local, espnet2-compatible TTS task.

Mirrors the spirit of ``espnet3/systems/asr/task.py`` (which reuses the espnet2
ASR task): this is a thin extension of espnet2's :class:`espnet2.tasks.tts.TTSTask`
that stays fully compatible with the built-in espnet2 TTS models (tacotron2,
fastspeech2, vits, jets, ...) while adding two F5-TTS components as named choices:

    * ``tts: f5tts``               -> :class:`src.f5_tts.model.f5tts.F5TTS`
    * ``feats_extract: vocoder_mel`` -> :class:`...feats_extract.vocoder_mel.VocoderMelSpec`

Selected in a training config via::

    task: src.f5_tts.task.TTSTask
    model:
      tts: f5tts
      tts_conf: {...}
      feats_extract: vocoder_mel
      feats_extract_conf: {...}

Because espnet2's ``TTSTask.build_model`` resolves components through the
module-level ``ClassChoices`` registries, we register the F5 classes into those
registries here. ``ClassChoices`` reads its ``classes`` dict live, so the new
names appear in ``--tts`` / ``--feats_extract`` and in ``build_model`` without
copying the task. Importing this module (which the recipe always does via
``task: src.f5_tts.task.TTSTask``) performs the registration.
"""

from espnet2.tasks.tts import TTSTask as _ESPnet2TTSTask
from espnet2.tasks.tts import feats_extractor_choices, tts_choices

from src.f5_tts.feats_extract.vocoder_mel import VocoderMelSpec
from src.f5_tts.model.f5tts import F5TTS

# Register F5-TTS components as named choices (idempotent).
tts_choices.classes.setdefault("f5tts", F5TTS)
feats_extractor_choices.classes.setdefault("vocoder_mel", VocoderMelSpec)


class TTSTask(_ESPnet2TTSTask):
    """espnet2 TTSTask + F5-TTS (``f5tts``) and vocoder mel (``vocoder_mel``)."""

    pass
