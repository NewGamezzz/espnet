"""Speaker similarity metrics (Task 5): reference embeddings, per-segment
cosine similarity against speaker references with an injective
cluster-to-speaker assignment (Set A path), and cross-run cluster
matching (Set B path).

Every function that needs an embedding takes an injected
`embed_fn: Callable[[np.ndarray], np.ndarray]` (16 kHz mono float32 audio
-> 1-D embedding vector), so unit tests exercise the pure
cropping/averaging/assignment logic with a cheap fake instead of paying
the WavLM load-and-inference cost. `default_embed_fn()` builds the real
one - `Wav2Vec2FeatureExtractor` + `WavLMForXVector` from
`microsoft/wavlm-base-plus-sv` - and is the only place `torch` /
`transformers` get imported, and only when called, so importing this
module never pulls in those heavy deps (Task 8 constructs the real
`embed_fn` once per eval run and threads it through the functions below).

Wav loading follows the resample idiom in
`dataset/preprocessing/audio.py`: `soundfile` read, `soxr` HQ resample to
16 kHz - no torch there either.

Segments shorter than `min_sec` are skipped for embedding everywhere in
this module; if a cluster has NO segment >= `min_sec` it simply has no
embedding and is excluded from assignment and mean computations rather
than crashing (see `_embed_segments_by_cluster`).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np
import soundfile as sf
import soxr

from eval.diarize import DiarSegment

_TARGET_SR = 16000

_feature_extractor = None
_model = None


@dataclass
class SimResult:
    sim_matrix: dict[tuple[str, str], float]
    assignment: dict[str, str]
    sim_own_mean: float
    margin_mean: float | None


def default_embed_fn(device: str = "cuda") -> Callable[[np.ndarray], np.ndarray]:
    """Build the real `embed_fn`: WavLM-base-plus-sv x-vector extraction.

    Lazily imports `torch` and `transformers` (never at module scope) and
    loads `microsoft/wavlm-base-plus-sv` - a `Wav2Vec2FeatureExtractor`
    plus `WavLMForXVector` - caching both in module globals so repeat
    calls (Task 8 builds one `embed_fn` per eval run) pay the load cost
    once. The returned callable takes 16 kHz mono float32 audio and
    returns its L2-normalized x-vector (`.embeddings[0]`) as a 1-D numpy
    array.
    """
    import torch
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

    global _feature_extractor, _model
    if _model is None:
        _feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "microsoft/wavlm-base-plus-sv"
        )
        _model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")
        _model.to(device)
        _model.eval()

    feature_extractor = _feature_extractor
    model = _model

    def embed_fn(audio: np.ndarray) -> np.ndarray:
        inputs = feature_extractor(audio, sampling_rate=_TARGET_SR, return_tensors="pt")
        with torch.no_grad():
            out = model(inputs.input_values.to(device))
        emb = torch.nn.functional.normalize(out.embeddings[0], dim=-1)
        return emb.cpu().numpy()

    return embed_fn


def _load_wav_mono_16k(path: str) -> np.ndarray:
    """Read `path` with `soundfile` and resample to 16 kHz mono float32
    via `soxr` HQ, following the single-resample-path idiom in
    `dataset/preprocessing/audio.py`. Multi-channel files are folded to
    mono by taking channel 0 - eval wavs (generated mixes, reference
    clips) are single-channel by construction upstream.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = np.ascontiguousarray(data[:, 0], dtype=np.float32)
    if sr != _TARGET_SR:
        mono = soxr.resample(mono, sr, _TARGET_SR, quality="HQ").astype(np.float32)
    return mono


def reference_embedding(
    ref_wav: str,
    turns_for_speaker: list[dict] | None,
    embed_fn: Callable[[np.ndarray], np.ndarray],
    max_sec: float = 30.0,
    min_sec: float = 1.0,
) -> np.ndarray | None:
    """Build one speaker's reference embedding from `ref_wav`.

    When `turns_for_speaker` is given (window-relative-seconds dicts with
    `"start"`/`"end"` keys), crops and concatenates only that speaker's
    turn spans, in the order given; otherwise embeds the whole file. The
    concatenated audio is capped to the first `max_sec` seconds before
    being passed to `embed_fn`.

    Returns ``None`` when less than `min_sec` of reference audio remains
    (every turn span fell outside the file or was zero-length, the file
    itself is empty, or the speaker's only turns are sub-second
    backchannels). Embedding an empty clip yields a NaN vector, which
    would then poison every cosine similarity against this speaker AND
    bias the injective assignment (a single NaN score makes every
    permutation total NaN), and real embedders' conv stacks crash outright
    on near-empty clips; callers must therefore skip a ``None`` reference,
    the same way clusters with no qualifying segment are skipped.
    """
    audio = _load_wav_mono_16k(ref_wav)

    if turns_for_speaker:
        pieces = []
        for turn in turns_for_speaker:
            start_sample = max(0, int(round(turn["start"] * _TARGET_SR)))
            end_sample = min(len(audio), int(round(turn["end"] * _TARGET_SR)))
            if end_sample > start_sample:
                pieces.append(audio[start_sample:end_sample])
        audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)

    max_samples = int(round(max_sec * _TARGET_SR))
    audio = audio[:max_samples]

    if audio.size < int(round(min_sec * _TARGET_SR)):
        return None

    return embed_fn(audio)


def _embed_segments_by_cluster(
    wav_path: str,
    segments: list[DiarSegment],
    embed_fn: Callable[[np.ndarray], np.ndarray],
    min_sec: float,
) -> dict[str, list[np.ndarray]]:
    """Embed every segment in `segments` whose duration is >= `min_sec`,
    grouped by `cluster`. Shorter segments are skipped entirely (never
    reach `embed_fn`). A cluster with no qualifying segment is simply
    absent from the returned dict.
    """
    audio = _load_wav_mono_16k(wav_path)
    by_cluster: dict[str, list[np.ndarray]] = {}
    for segment in segments:
        if segment.end - segment.start < min_sec:
            continue
        start_sample = max(0, int(round(segment.start * _TARGET_SR)))
        end_sample = min(len(audio), int(round(segment.end * _TARGET_SR)))
        if end_sample <= start_sample:
            continue
        emb = embed_fn(audio[start_sample:end_sample])
        by_cluster.setdefault(segment.cluster, []).append(emb)
    return by_cluster


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _best_injective_assignment(
    rows: list[str],
    cols: list[str],
    score: dict[tuple[str, str], float],
) -> dict[str, str]:
    """Brute-force injective `row -> col` assignment maximizing summed
    `score`, over `itertools.permutations` of `cols` (padded with `None`
    if shorter than `rows`) matched against `rows` in order.

    `rows` and `cols` must already be sorted by the caller so ties break
    deterministically on first-found-in-order, matching `eval.metrics.wer
    .cpwer`'s brute-force pattern. A `row` left unmatched (more rows than
    cols) is simply absent from the returned mapping.
    """
    if not rows or not cols:
        return {}

    n = max(len(rows), len(cols))
    padded_cols = cols + [None] * (n - len(cols))

    best_total: float | None = None
    best_assignment: dict[str, str] = {}
    for perm in itertools.permutations(padded_cols):
        assignment: dict[str, str] = {}
        total = 0.0
        for row, col in zip(rows, perm):
            if col is None:
                continue
            total += score[(row, col)]
            assignment[row] = col
        if best_total is None or total > best_total:
            best_total = total
            best_assignment = assignment
    return best_assignment


def segment_similarities(
    gen_wav: str,
    segments: list[DiarSegment],
    ref_embs: dict[str, np.ndarray | None],
    embed_fn: Callable[[np.ndarray], np.ndarray],
    min_sec: float = 1.0,
) -> SimResult:
    """Score each diarized cluster in `gen_wav` against every speaker
    reference embedding in `ref_embs` (Set A path).

    Embeds every segment >= `min_sec` (via `_embed_segments_by_cluster`;
    shorter segments are skipped and clusters with no qualifying segment
    are excluded entirely - see module docstring), then for each
    `(cluster, speaker)` pair averages the per-segment cosine similarity
    against that speaker's reference embedding into `sim_matrix`.

    `assignment` is the injective cluster -> speaker mapping (brute force
    over sorted cluster/speaker keys, via `_best_injective_assignment`)
    that maximizes summed own-similarity. `sim_own_mean` is the mean, over
    assigned clusters, of that cluster's similarity to its assigned
    speaker. `margin_mean` is the mean, over assigned clusters, of (own
    similarity - the max similarity to any *other* speaker); with only a
    single reference speaker there is no "other" to compare against, so
    `margin_mean` is `None`.
    """
    cluster_embs = _embed_segments_by_cluster(gen_wav, segments, embed_fn, min_sec)

    clusters = sorted(cluster_embs.keys())
    # Skip speakers whose reference crop was degenerate (None embedding);
    # including one would inject NaN similarities and bias the assignment.
    speakers = sorted(s for s, emb in ref_embs.items() if emb is not None)

    sim_matrix: dict[tuple[str, str], float] = {}
    for cluster in clusters:
        embs = cluster_embs[cluster]
        for speaker in speakers:
            ref = ref_embs[speaker]
            sim_matrix[(cluster, speaker)] = float(
                np.mean([_cosine(emb, ref) for emb in embs])
            )

    assignment = _best_injective_assignment(clusters, speakers, sim_matrix)

    own_sims = [
        sim_matrix[(cluster, speaker)] for cluster, speaker in assignment.items()
    ]
    sim_own_mean = float(np.mean(own_sims)) if own_sims else 0.0

    margins = []
    for cluster, speaker in assignment.items():
        other_sims = [
            sim_matrix[(cluster, other)] for other in speakers if other != speaker
        ]
        if other_sims:
            margins.append(sim_matrix[(cluster, speaker)] - max(other_sims))
    margin_mean = float(np.mean(margins)) if margins else None

    return SimResult(
        sim_matrix=sim_matrix,
        assignment=assignment,
        sim_own_mean=sim_own_mean,
        margin_mean=margin_mean,
    )


def cluster_cross_similarity(
    gen_wav: str,
    gen_segments: list[DiarSegment],
    gt_wav: str,
    gt_segments: list[DiarSegment],
    embed_fn: Callable[[np.ndarray], np.ndarray],
    min_sec: float = 1.0,
) -> float:
    """Set B path: mean matched cosine similarity between generated-run
    cluster mean embeddings and ground-truth-run cluster mean embeddings,
    under the optimal injective assignment.

    Unlike `segment_similarities` (which averages per-segment cosine
    similarities), this averages the *embeddings* within each cluster
    first - on both the `gen` and `gt` sides - via
    `_embed_segments_by_cluster`, then takes cosine similarity between
    cluster means. Segments shorter than `min_sec` are skipped on both
    sides; a cluster with no segment >= `min_sec` has no mean embedding
    and is excluded from both the assignment and the returned mean.
    Returns 0.0 if either side has no embeddable cluster.
    """
    gen_by_cluster = _embed_segments_by_cluster(
        gen_wav, gen_segments, embed_fn, min_sec
    )
    gt_by_cluster = _embed_segments_by_cluster(gt_wav, gt_segments, embed_fn, min_sec)

    gen_means = {
        cluster: np.mean(embs, axis=0) for cluster, embs in gen_by_cluster.items()
    }
    gt_means = {
        cluster: np.mean(embs, axis=0) for cluster, embs in gt_by_cluster.items()
    }

    gen_clusters = sorted(gen_means.keys())
    gt_clusters = sorted(gt_means.keys())

    sim_matrix = {
        (g, t): _cosine(gen_means[g], gt_means[t])
        for g in gen_clusters
        for t in gt_clusters
    }
    assignment = _best_injective_assignment(gen_clusters, gt_clusters, sim_matrix)
    if not assignment:
        return 0.0

    sims = [sim_matrix[(g, t)] for g, t in assignment.items()]
    return float(np.mean(sims))
