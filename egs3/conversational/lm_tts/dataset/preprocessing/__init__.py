"""Pure preprocessing algorithms behind the SSSD data pipeline.

No I/O side effects beyond reading the corpus manifests and (for the audio
tail) writing window wavs; no torch, no config access.

- ``sssd.py``       corpus manifest parsing, path remapping, turn merging
- ``windows.py``    boundary eligibility and window placement
- ``audio.py``      window audio extraction: resample to 16 kHz and write
                     per-channel + mixed-mono wavs
- ``attributes.py`` per-speaker voice-attribute measurement (pitch/rate/
                     gender bands) for caption generation

Ported from the F5 recipe (``egs3/conversational/tts/dataset/preprocessing``);
this package will grow the BagPiper-specific tail (text/token encoding,
record assembly) in later tasks.
"""
