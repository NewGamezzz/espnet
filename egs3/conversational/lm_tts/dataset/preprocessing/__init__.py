"""Pure preprocessing algorithms behind the SSSD data pipeline.

No I/O side effects beyond reading the corpus manifests, no torch, no config
access.

- ``sssd.py``     corpus manifest parsing, path remapping, turn merging
- ``windows.py``  boundary eligibility and window placement

Ported from the F5 recipe (``egs3/conversational/tts/dataset/preprocessing``);
this package will grow the BagPiper-specific tail (text/token encoding,
record assembly) in later tasks.
"""
