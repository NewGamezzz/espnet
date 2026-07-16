"""WER metrics (Task 3): pooled error counts, concat WER, and cpWER.

Pure scoring logic - no audio, no models. Downstream (Task 8) pools
``ErrorCounts`` across windows via ``__add__`` and calls ``wer_concat`` /
``cpwer`` per window. The binding project rule: WER is computed by pooling
I/D/S error counts and reference word counts across utterances THEN
dividing - never by averaging per-utterance WERs. ``ErrorCounts.__add__``
is the pooling primitive that makes this the only easy path.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

_normalizer = EnglishTextNormalizer()


@dataclass
class ErrorCounts:
    hits: int
    substitutions: int
    deletions: int
    insertions: int

    @property
    def ref_words(self) -> int:
        return self.hits + self.substitutions + self.deletions

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        if self.ref_words == 0:
            raise ValueError(
                "ref_words is zero; WER is undefined for an empty reference"
            )
        return self.errors / self.ref_words

    def __add__(self, other: "ErrorCounts") -> "ErrorCounts":
        if not isinstance(other, ErrorCounts):
            return NotImplemented
        return ErrorCounts(
            hits=self.hits + other.hits,
            substitutions=self.substitutions + other.substitutions,
            deletions=self.deletions + other.deletions,
            insertions=self.insertions + other.insertions,
        )


@dataclass
class CpWerResult:
    counts: ErrorCounts
    mapping: dict[str, str | None]


def normalize(text: str) -> str:
    """Fold case/punctuation/number-format variation via Whisper's English
    text normalizer (module-level singleton - construction is not free).
    """
    return _normalizer(text)


def count_errors(ref: str, hyp: str) -> ErrorCounts:
    """Word-level I/D/S counts between ``ref`` and ``hyp`` after
    ``normalize``. Raises ``ValueError`` (loud, no silent zero-division
    later) when the normalized reference is empty.
    """
    norm_ref = normalize(ref)
    norm_hyp = normalize(hyp)
    if not norm_ref.strip():
        raise ValueError(f"empty normalized reference: {ref!r}")
    out = jiwer.process_words(norm_ref, norm_hyp)
    return ErrorCounts(
        hits=out.hits,
        substitutions=out.substitutions,
        deletions=out.deletions,
        insertions=out.insertions,
    )


def wer_concat(turn_texts: list[str], hyp_text: str) -> ErrorCounts:
    """Concat-WER: reference is every turn's text joined with a single
    space, in order, scored against one hypothesis string.
    """
    ref = " ".join(turn_texts)
    return count_errors(ref, hyp_text)


def _unmapped_ref_counts(ref_text: str) -> ErrorCounts:
    """A speaker with no assigned cluster: its whole reference counts as
    deletions (equivalent to scoring against an empty hypothesis).

    Counted on the raw (pre-``normalize``) whitespace split, not the
    normalized token count: ``EnglishTextNormalizer`` collapses runs of
    spelled-out numbers into a single merged digit token (e.g. ``"four
    five"`` -> ``"45"``), which would undercount how many reference words
    are actually being penalized.
    """
    words = ref_text.split()
    return ErrorCounts(hits=0, substitutions=0, deletions=len(words), insertions=0)


def _unmapped_hyp_counts(hyp_text: str) -> ErrorCounts:
    """A cluster with no assigned speaker: its whole hypothesis counts as
    insertions (equivalent to scoring an empty reference against it). See
    ``_unmapped_ref_counts`` for why this counts raw words, not normalized
    tokens.
    """
    words = hyp_text.split()
    return ErrorCounts(hits=0, substitutions=0, deletions=0, insertions=len(words))


def cpwer(
    refs_by_speaker: dict[str, str], hyps_by_cluster: dict[str, str]
) -> CpWerResult:
    """Concatenated minimum-permutation WER (cpWER): brute-force over
    injective assignments of clusters to speakers, choosing the assignment
    that minimizes pooled ``errors``.

    Cluster and speaker keys are sorted before permuting so ties are
    broken deterministically by first-found in that fixed order. The
    smaller side is padded with pseudo entries: a speaker left unmapped
    scores its full reference as deletions; a cluster left unmapped scores
    its full hypothesis as insertions.
    """
    speakers = sorted(refs_by_speaker.keys())
    clusters = sorted(hyps_by_cluster.keys())
    n = max(len(speakers), len(clusters))
    padded_speakers = speakers + [None] * (n - len(speakers))
    padded_clusters = clusters + [None] * (n - len(clusters))

    best_counts: ErrorCounts | None = None
    best_mapping: dict[str, str | None] | None = None

    for perm in itertools.permutations(padded_speakers):
        pooled = ErrorCounts(hits=0, substitutions=0, deletions=0, insertions=0)
        mapping: dict[str, str | None] = {}
        for cluster, speaker in zip(padded_clusters, perm):
            if cluster is not None and speaker is not None:
                pair_counts = count_errors(
                    refs_by_speaker[speaker], hyps_by_cluster[cluster]
                )
                mapping[cluster] = speaker
            elif cluster is not None:  # speaker is None: unmapped cluster
                pair_counts = _unmapped_hyp_counts(hyps_by_cluster[cluster])
                mapping[cluster] = None
            elif speaker is not None:  # cluster is None: unmapped speaker
                pair_counts = _unmapped_ref_counts(refs_by_speaker[speaker])
            else:
                continue
            pooled = pooled + pair_counts

        if best_counts is None or pooled.errors < best_counts.errors:
            best_counts = pooled
            best_mapping = mapping

    if best_counts is None:
        best_counts = ErrorCounts(hits=0, substitutions=0, deletions=0, insertions=0)
        best_mapping = {}

    return CpWerResult(counts=best_counts, mapping=best_mapping)
