"""On-disk layouts of the staged Emilia corpus.

``sub_shards`` (default) is the layout of the extracted Hugging Face tars:
``emilia/<LANG>/<LANG>-Bxxxxxx/<utt_id>.mp3`` with one ``<utt_id>.json`` next
to every audio file.

``speaker_tree`` is the copy shared on NCSA Delta:
``emilia/<LANG>/<LANG>_Bxxxxx/<speaker>/mp3/<utt_id>.mp3`` with one
``emilia/<LANG>/<LANG>_Bxxxxx.jsonl`` of records per batch. Every path
component is a prefix of the utterance id, so audio is located from the id
alone and a manifest built on the other layout loads unchanged.
"""

from __future__ import annotations

SUB_SHARDS = "sub_shards"
SPEAKER_TREE = "speaker_tree"
LAYOUTS = (SUB_SHARDS, SPEAKER_TREE)


def corpus_layout(builder_cfg) -> str:
    """Return the validated ``builder.corpus_layout`` of a dataset config."""
    layout = str(builder_cfg.get("corpus_layout", SUB_SHARDS))
    if layout not in LAYOUTS:
        raise ValueError(
            f"Unknown builder.corpus_layout {layout!r}; expected one of {LAYOUTS}"
        )
    return layout


def speaker_tree_audio_dir(utt_id: str) -> str:
    """Directory of an utterance's audio, relative to ``emilia/``.

    ``EN_B00012_S00001_W000000`` -> ``EN/EN_B00012/EN_B00012_S00001/mp3``.
    """
    lang, batch, speaker, _ = utt_id.split("_")
    return f"{lang}/{lang}_{batch}/{lang}_{batch}_{speaker}/mp3"
