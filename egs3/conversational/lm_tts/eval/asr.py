"""ASR wrapper (Task 4) and word-to-diarization-segment assignment.

`transcribe()` wraps a `transformers` `automatic-speech-recognition`
pipeline - the heavy, GPU-bound imports (`torch`, `transformers`) happen
lazily, inside the function only, so importing this module never pulls
them in. The pipeline is cached in a module global because loading it is
expensive and Task 8 calls `transcribe()` once per eval window.

`assign_words()` is PURE logic (no audio, no models): it maps ASR words
onto `eval.diarize.DiarSegment`s by time, for Task 8 to build per-cluster
hypothesis text ahead of cpWER scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

from eval.diarize import DiarSegment

_pipe = None


@dataclass
class Word:
    text: str
    start: float
    end: float


def transcribe(wav_path: str, device: str = "cuda") -> tuple[str, list[Word]]:
    """Transcribe `wav_path` with Whisper large-v3, returning the full text
    and a word list with timestamps.

    Lazily imports `torch` and `transformers.pipeline` (never at module
    scope) and builds an `automatic-speech-recognition` pipeline for
    `openai/whisper-large-v3` in fp16. Word timestamps come from
    `out["chunks"]`; a chunk whose end timestamp is `None` (a known
    Whisper long-form quirk) is patched with the next chunk's start, or -
    for the last chunk, where there is no "next" - with the chunk's own
    start.
    """
    global _pipe
    if _pipe is None:
        import torch
        from transformers import pipeline as hf_pipeline

        _pipe = hf_pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3",
            torch_dtype=torch.float16,
            device=device,
        )

    out = _pipe(
        wav_path,
        return_timestamps="word",
        chunk_length_s=30,
        generate_kwargs={"language": "english", "task": "transcribe"},
    )

    chunks = out["chunks"]
    words = []
    for i, chunk in enumerate(chunks):
        start, end = chunk["timestamp"]
        if end is None:
            end = chunks[i + 1]["timestamp"][0] if i + 1 < len(chunks) else start
        words.append(Word(text=chunk["text"].strip(), start=start, end=end))

    return out["text"], words


def assign_words(words: list[Word], segments: list[DiarSegment]) -> dict[str, str]:
    """Assign each word to the diarization segment containing its
    midpoint; a word whose midpoint falls in no segment snaps to the
    segment with the nearest boundary.

    Returns `{cluster: text}` for every cluster appearing in `segments`
    (clusters with no assigned words map to `""`), with each cluster's
    text joining its words in time order regardless of the input order of
    `words`.
    """
    words_by_cluster: dict[str, list[Word]] = {
        segment.cluster: [] for segment in segments
    }

    for word in words:
        midpoint = (word.start + word.end) / 2
        containing = next((s for s in segments if s.start <= midpoint <= s.end), None)
        target = containing or min(
            segments,
            key=lambda s: min(abs(midpoint - s.start), abs(midpoint - s.end)),
        )
        words_by_cluster[target.cluster].append(word)

    return {
        cluster: " ".join(w.text for w in sorted(cluster_words, key=lambda w: w.start))
        for cluster, cluster_words in words_by_cluster.items()
    }
