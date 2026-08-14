"""Conversation dataset backed by session manifests, plus the packed collator.

Windows are planned online: ``__init__`` reads a manifest of
``SessionRecord``s (build-time output, one row per session/utterance) and
derives ``records`` by running the planner in frozen mode (``epoch=None``),
bit-identical to the retired offline window manifests.  Training loops that
want fresh windows per epoch call ``plan_windows(epoch)`` directly instead of
relying on ``records``.

Audio is seek-read from the session FLAC (only the window's segment), kept
per-channel, and resampled 48 -> 24 kHz on the fly.  The dataset is
vocab-agnostic: items carry the raw turns and tokenization happens in
``preprocessor.ConversationalTextPreprocessor`` (the ``DataOrganizer``
``preprocessor:`` slot), which fills in the ``text`` key the collator packs.
Batches use the packed layout of the ``branch_exchange`` package: a
per-conversation ``counts`` list plus row-stacked ``(sum(counts), ...)``
tensors with no padding rows on the branch axis, so mixed channel counts per
batch need no special casing.
"""

from __future__ import annotations

import dataclasses
import random
from importlib import resources
from pathlib import Path
from typing import Any, Sequence

import torch
import torchaudio
from torch.utils.data import Dataset as TorchDataset

from espnet2.fileio.sound_scp import soundfile_read
from espnet3.utils.config_utils import load_config_with_defaults

from .builder import resolve_dataset_root
from .preprocessing.planner import WindowParams, plan_sessions
from .preprocessing.sessions import read_session_manifest
from .preprocessing.windows import WindowRecord

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)
_DATASET_CFG = _CONFIG["dataset"]
_BUILDER_CFG = _CONFIG["builder"]


class ConversationDataset(TorchDataset):
    """Multi-channel conversation windows for F5-TTS fine-tuning.

    Each item:
      - ``window_id``    : str
      - ``num_channels`` : int, N of this conversation
      - ``speech``       : float32 tensor (N, T) at ``fs`` (default 24 kHz)
      - ``turns``        : the window's ``Turn`` records in conversation
                           order, with ``channel`` remapped to the
                           post-permutation row index (``speaker``/``text``/
                           ``start``/``end`` kept verbatim)
      - ``perm``         : int64 tensor (N,), the channel permutation applied
                           to the audio rows (row k holds original channel
                           ``perm[k]``)

    Records carrying a special-token conditioning plan
    (``WindowRecord.chunk_task``, see ``preprocessing.chunk_task``) assemble
    their ``speech`` as ``[P | H | target]`` - per-speaker voice prompt,
    previous-chunk slice, then the window - and carry three extra int keys:
    ``prompt_frames``, ``prev_frames`` and ``cond_frames`` (their sum), all in
    mel frames of ``hop`` samples.  Ordinary infill records carry none of them,
    so nothing downstream of the untouched majority path changes.

    No token ids here: tokenization lives in the recipe preprocessor
    (``preprocessor.ConversationalTextPreprocessor``), which derives the
    per-branch ``text`` tensors from ``turns``.  Because ``channel`` is
    pre-remapped, everything downstream is permutation-agnostic.

    ``permute_channels`` defaults to ``split == "train"``; it guards against
    systematic ch0/ch1 artifacts in the corpus and is applied consistently to
    audio rows and turn channels (turn markers carry no identity, so nothing
    else needs re-indexing).

    ``min_active_speakers`` drops windows with fewer active speakers (the
    planned windows keep single-speaker windows; the POC trains with ``2`` so
    gradient mass concentrates on actual interactions - a knob, not a
    rebuild).

    ``hop`` is the dataset-side single source of truth for the mel frame rate
    (``fs / hop`` = 93.75 frames/s at the defaults).  It MUST equal the mel
    hop the model's feature extractor uses (``hop_length`` in the training
    configs): chunk-task assembly snaps the conditioning blocks to whole hops
    so ``cond_frames`` indexes the assembled sample's mel frames exactly, and
    a mismatch would silently shift the conditioning/target boundary.

    ``window_params``/``window_seed`` control the online planner
    (``preprocessing.planner.plan_sessions``): ``self.records`` is the frozen
    plan (``plan_windows(None)``, seed 0 by default so it reproduces the
    retired offline manifests bit-for-bit) and serves ``__len__``/
    ``__getitem__(int)``, inference, and any init-time probe.  Training loops
    that want fresh windows per epoch call ``plan_windows(epoch)`` directly.
    """

    def __init__(
        self,
        split: str,
        recipe_dir: str | Path | None = None,
        manifest_path: str | Path | None = None,
        dataset_root: str | Path | None = None,
        fs: int | None = None,
        hop: int = 256,
        permute_channels: bool | None = None,
        seed: int = 0,
        inference: bool = False,
        min_active_speakers: int = 1,
        window_params: dict | None = None,
        window_seed: int = 0,
    ) -> None:
        self.split = split
        self.fs = int(fs if fs is not None else _DATASET_CFG["sample_rate"])
        self.hop = int(hop)
        self.permute_channels = (
            permute_channels if permute_channels is not None else split == "train"
        )
        self.seed = seed
        self.inference = inference
        self.dataset_root = resolve_dataset_root(dataset_root)

        recipe_root = (
            Path(recipe_dir).resolve()
            if recipe_dir is not None
            else Path(__file__).resolve().parents[1]
        )
        data_dir = recipe_root / _BUILDER_CFG["data_path"]
        if manifest_path is None:
            if split not in _DATASET_CFG["split_manifest_paths"]:
                raise KeyError(f"unknown split {split!r} and no manifest_path given")
            manifest_path = data_dir / _DATASET_CFG["split_manifest_paths"][split]
        manifest_path = Path(manifest_path)
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"session manifest not found: {manifest_path}. Run the "
                "corresponding dataset builder first (SSSD: python -m "
                "egs3.conversational.tts.dataset.builder; LibriTTS: python -m "
                "egs3.conversational.tts.dataset.libritts_builder)."
            )
        self.sessions = read_session_manifest(manifest_path)
        self.window_params = WindowParams(**(window_params or {}))
        self.window_seed = int(window_seed)
        self.min_active_speakers = int(min_active_speakers)
        # Frozen plan: bit-identical to the retired offline manifests (same
        # legacy RNG string, same build seed 0).  Serves valid/test splits,
        # inference, int indexing, and CombinedDataset's init-time probes.
        self.records = self.plan_windows(epoch=None)
        if not self.records:
            raise RuntimeError(
                f"no plannable window in {manifest_path} with >= "
                f"{min_active_speakers} active speakers"
            )

        # Test hook: a fixed permutation (sequence of ints) overrides the RNG.
        self._fixed_perm: Sequence[int] | None = None
        # Created lazily on first __getitem__ so each DataLoader worker gets
        # its own stream (torch re-seeds workers per epoch); creating it here
        # would fork one shared state into every worker.
        self._worker_rng: random.Random | None = None

    def __len__(self) -> int:
        return len(self.records)

    def plan_windows(self, epoch: int | None) -> list[WindowRecord]:
        """Plan this split's windows for ``epoch`` (None = frozen legacy plan).

        Pure function of (window_seed, epoch, session metadata): every DDP
        rank and DataLoader worker derives the identical plan.
        """
        records, _stats = plan_sessions(
            self.sessions,
            params=self.window_params,
            seed=self.window_seed,
            epoch=epoch,
        )
        if self.min_active_speakers > 1:
            records = [
                r for r in records if r.num_active_speakers >= self.min_active_speakers
            ]
        return records

    def _draw_perm(self, n: int) -> list[int]:
        if self._fixed_perm is not None:
            perm = list(self._fixed_perm)
            if sorted(perm) != list(range(n)):
                raise ValueError(
                    f"fixed perm {perm} is not a permutation of range({n})"
                )
            return perm
        if not self.permute_channels:
            return list(range(n))
        if self._worker_rng is None:
            self._worker_rng = random.Random(f"{self.seed}:{torch.initial_seed()}")
        return self._worker_rng.sample(range(n), n)

    def _load_span(self, record: WindowRecord, a: float, b: float) -> torch.Tensor:
        """Seek-read ``[a, b)`` SESSION-ABSOLUTE seconds of ``record``'s audio.

        Same validation and resample as the plain window read, but ``a``/``b``
        are unconstrained by ``record.t0``/``t1``: chunk-task assembly pulls
        prompt and previous-chunk material from anywhere in the session, and
        ``ChunkTaskPlan``'s spans live in the same absolute coordinates as
        ``t0``/``t1`` (see ``preprocessing.chunk_task``).
        """
        path = self.dataset_root / record.audio_relpath
        start = round(a * record.sample_rate)
        stop = round(b * record.sample_rate)
        array, rate = soundfile_read(
            str(path), dtype="float32", start=start, end=stop, always_2d=True
        )
        if rate != record.sample_rate:
            raise RuntimeError(
                f"{path}: sample rate {rate} != manifest rate {record.sample_rate}"
            )
        if array.shape[1] != record.num_channels:
            raise RuntimeError(
                f"{path}: {array.shape[1]} channels != manifest {record.num_channels}"
            )
        speech = torch.from_numpy(array.T.copy())  # (N, T_src)
        if self.fs != record.sample_rate:
            speech = torchaudio.functional.resample(
                speech, orig_freq=record.sample_rate, new_freq=self.fs
            )
        return speech

    def _load_speech(self, record: WindowRecord) -> torch.Tensor:
        return self._load_span(record, record.t0, record.t1)

    def _assemble_chunk_task(
        self, record: WindowRecord, perm: Sequence[int]
    ) -> tuple[torch.Tensor, int, int]:
        """Assemble a chunk-task record as ``[P | H | target]`` (rows in
        ``perm`` order) and return ``(speech, prompt_frames, prev_frames)``.

        ``P`` is the per-speaker voice prompt, ``H`` the previous-chunk slice
        immediately preceding the window (empty for a ``prompt_only`` plan),
        and ``target`` the window itself.  Both conditioning blocks are snapped
        to a whole number of ``self.hop`` samples so that the target starts at
        frame ``prompt_frames + prev_frames`` - the model masks conditioning by
        frame index, so a partial hop anywhere before ``t0`` would offset every
        downstream frame.

        How exact that boundary is: sample-exact when the source is already at
        ``self.fs``, and within one output sample otherwise, because ``b``
        below predicts t0's offset from the span in SECONDS while the block's
        real offset comes from rounding each edge to SOURCE samples - the two
        can disagree by half a source sample.  One sample is two orders of
        magnitude below ``hop`` (256), and the mel boundary is softer than the
        sample boundary anyway: with ``n_fft`` (1024) > ``hop``, the frame at
        ``cond_frames`` straddles the seam and mixes conditioning with target
        audio regardless.  Same convention as inference (``src/inference.py``).
        """
        plan = record.chunk_task

        # --- P: one voice reference per ORIGINAL channel -------------------
        # Each channel's prompt span is drawn independently (different anchors,
        # different times), so this is one read per channel of which only that
        # channel's column is kept.  Iterating over ``perm`` puts channel
        # perm[r] in row r, matching the row order the H+target block gets.
        rows = [self._load_span(record, *plan.prompt_spans[c])[c] for c in perm]
        # The spans share a length in SECONDS, but rounding each edge to source
        # samples and resampling can leave rows a sample or two apart; P has to
        # be rectangular, so level down to the shortest row before snapping.
        prompt_frames = min(row.shape[0] for row in rows) // self.hop
        prompt_samples = prompt_frames * self.hop
        # Trim P's TAIL: each span begins on its anchor turn's onset (see
        # draw_chunk_task step 4), so the head is the part of the reference
        # worth hearing and the far end is arbitrary continuation.  Same
        # convention as inference's prompt trim (src/inference.py).
        prompt = torch.stack([row[:prompt_samples] for row in rows])

        # --- H + target: ONE read, so the t0 seam is a slice, not a join ----
        # Reading [prev_start, t1) in a single call keeps H and the target
        # sample-contiguous and, at 48 kHz sources, resampled as one signal -
        # two reads would put a resampler edge transient right at t0.
        prev_start = plan.prev_span[0] if plan.prev_span is not None else record.t0
        block = self._load_span(record, prev_start, record.t1)[perm]
        b = round((record.t0 - prev_start) * self.fs)
        prev_frames = b // self.hop
        # Trim H's HEAD, not its tail: t0 sits at sample ``b`` of this block and
        # must land on frame prompt_frames + prev_frames, and the target after
        # it must stay whole and contiguous - so the sub-hop remainder can only
        # come off the far (earliest) end, which is expendable context.  A
        # prompt_only plan has prev_start == t0, hence b == 0, and this slice
        # is a no-op.
        block = block[:, b - prev_frames * self.hop :]
        return torch.cat([prompt, block], dim=1), prompt_frames, prev_frames

    def load_window(self, record: WindowRecord) -> dict[str, Any]:
        n = record.num_channels
        # Drawn once and shared by every read below: a second _draw_perm call
        # would consume the worker RNG at a chunk-task-dependent rate and
        # desync the permutation stream from the infill path.
        perm = self._draw_perm(n)
        chunk_frames: dict[str, int] = {}
        if record.chunk_task is None:
            speech = self._load_speech(record)[perm]
        else:
            speech, prompt_frames, prev_frames = self._assemble_chunk_task(record, perm)
            chunk_frames = {
                "prompt_frames": prompt_frames,
                "prev_frames": prev_frames,
                "cond_frames": prompt_frames + prev_frames,
            }
        inv = {orig: row for row, orig in enumerate(perm)}
        sample: dict[str, Any] = {
            "window_id": record.window_id,
            "num_channels": n,
            "speech": speech,
            "turns": [
                dataclasses.replace(t, channel=inv[t.channel])  # row index
                for t in record.turns
            ],
            "perm": torch.tensor(perm, dtype=torch.long),
        }
        # Only chunk-task records grow keys: an ordinary infill sample stays
        # exactly the dict every existing consumer already expects.
        sample.update(chunk_frames)
        if self.inference:
            sample.update(
                session_id=record.session_id,
                t0=record.t0,
                t1=record.t1,
                audio_path=str(self.dataset_root / record.audio_relpath),
            )
        return sample

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.load_window(self.records[idx])


def collate_conversations(
    samples: list[dict[str, Any]], *, text_pad_value: int | None = None
) -> dict[str, Any]:
    """Pack a batch of conversations into the branch_exchange layout.

    Rows are conversation-contiguous (sample order, branch order within each
    sample), matching ``conv_id = arange(B).repeat_interleave(counts)`` as
    built by ``BranchContext.branches(counts)``.  Audio pads with 0.0 plus a
    validity mask; text pads with -1 (the F5 model shifts ids by +1 and treats
    0 as its internal filler).
    """
    if text_pad_value is None:
        text_pad_value = int(_DATASET_CFG["text_pad_value"])
    counts = [s["num_channels"] for s in samples]
    m = sum(counts)
    speech_rows = [row for s in samples for row in s["speech"]]
    text_rows = [t for s in samples for t in s["text"]]

    t_max = max(row.shape[0] for row in speech_rows)
    speech = torch.zeros(m, t_max, dtype=torch.float32)
    speech_mask = torch.zeros(m, t_max, dtype=torch.bool)
    speech_lengths = torch.zeros(m, dtype=torch.long)
    for i, row in enumerate(speech_rows):
        speech[i, : row.shape[0]] = row
        speech_mask[i, : row.shape[0]] = True
        speech_lengths[i] = row.shape[0]

    l_max = max(t.shape[0] for t in text_rows)
    text = torch.full((m, l_max), text_pad_value, dtype=torch.long)
    text_lengths = torch.zeros(m, dtype=torch.long)
    for i, t in enumerate(text_rows):
        text[i, : t.shape[0]] = t
        text_lengths[i] = t.shape[0]

    # Per-conversation cond_frames (Task 6's chunk-task samples), -1 sentinel
    # for infill samples that carry no such key.  Key ALWAYS present so the
    # model kwarg's all-(-1) default is a well-formed sentinel batch.
    cond_frames = torch.tensor(
        [s.get("cond_frames", -1) for s in samples], dtype=torch.long
    )

    return {
        "counts": counts,
        "speech": speech,
        "speech_lengths": speech_lengths,
        "speech_mask": speech_mask,
        "text": text,
        "text_lengths": text_lengths,
        "cond_frames": cond_frames,
        "window_ids": [s["window_id"] for s in samples],
    }
