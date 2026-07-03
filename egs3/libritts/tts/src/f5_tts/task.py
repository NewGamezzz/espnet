"""Registration hook that adds F5-TTS to espnet2's TTS task.

This module is **not** a new task — it re-exports espnet2's
:class:`espnet2.tasks.tts.TTSTask` unchanged. Its only job is the import-time
side effect of registering two F5 components as named choices:

    * ``tts: f5tts``                 -> :class:`espnet2.tts.f5.f5tts.F5TTS`
    * ``feats_extract: vocoder_mel`` -> :class:`espnet2.tts.feats_extract.vocoder_mel.VocoderMelSpec`

Why this module has to exist
----------------------------
``TTSTask.build_model`` resolves these components **by name** through the
module-level ``ClassChoices`` registries (``tts_choices.get_class(args.tts)``,
see ``espnet2/tasks/tts.py``), not via a hydra ``_target_``. So the names must be
registered before ``build_model`` runs.

We deliberately register here rather than in ``espnet2/tasks/tts.py`` itself:
doing it in core would import ``F5TTS`` (and thus ``x_transformers``) on *every*
import of the TTS task, forcing that dependency on all ASR/TTS users. Keeping the
registration recipe-local confines the extra deps to runs that actually use F5.

The config line ``task: src.f5_tts.task.TTSTask`` is what imports this module and
triggers the registration; it then gets espnet2's real ``TTSTask``.
"""

from espnet2.tasks.tts import TTSTask, feats_extractor_choices, tts_choices

from espnet2.tts.f5.f5tts import F5TTS
from espnet2.tts.feats_extract.vocoder_mel import VocoderMelSpec

# Register F5-TTS components as named choices (idempotent). Re-exporting
# espnet2's TTSTask keeps `task: src.f5_tts.task.TTSTask` working unchanged.
tts_choices.classes.setdefault("f5tts", F5TTS)
feats_extractor_choices.classes.setdefault("vocoder_mel", VocoderMelSpec)

__all__ = ["TTSTask"]
