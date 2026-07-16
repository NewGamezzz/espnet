"""Measure-stage metrics for the conversational multi-branch F5 recipe.

Everything under this package is evaluation-only: it is driven by the
``measure`` stage (``run.py`` / ``conf/metrics.yaml``) and must never be
imported by the training path (``src/lit_module.py``, ``src/build_model.py``,
``src/system.py``'s ``train``/``_build_trainer``). Heavyweight model
dependencies (faster-whisper, WavLM, UTMOS, ...) are lazily imported inside
functions/constructors, never at module scope, so importing this package
stays CPU-only and network-free.
"""
