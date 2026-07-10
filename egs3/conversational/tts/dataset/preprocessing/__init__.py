"""Pure preprocessing algorithms behind the SSSD dataset pipeline.

No I/O side effects beyond reading the corpus manifests, no torch, no config
access; the package-level entry points (``builder.py``, ``dataset.py``,
``preprocessor.py``) orchestrate these modules:

- ``sssd.py``     corpus manifest parsing, path remapping, turn merging
- ``windows.py``  boundary eligibility and window placement
- ``text.py``     vocab extension, normalization, branch-text masking
"""
