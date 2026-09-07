"""LEMAS training dataset: online prompt draws + dual-prompt layout (spec 3.4, 5).

Every row draws its two prompt partners at access time, seeded by
``(seed, epoch, row)``: reproducible within a run, fresh every epoch. The
sample is ``[speaker prompt | language prompt | target]`` at 24 kHz with a text
lane of ``<spk>`` and ``<lang>`` repeated per prompt frame, the language tag,
and the target phones. Dropout is omission of a region, so ``cond_frames``
shrinks with the row and reaches 0 when both prompts are dropped.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torchaudio
from dataset.manifest import ManifestColumns
from src.layout import (
    HOP,
    SR,
    SRC_SR,
    TokenTable,
    build_text_ids,
    cond_frames,
    quantize_prompt_16k,
    region_frames,
)
from src.text.lemas_phonemizer import LANGS

from espnet3.utils.config_utils import load_config_with_defaults

DEFAULT_PROMPT_CONFIG = dict(
    spk_prompt_sec=[1.0, 6.0],
    lang_prompt_sec=[1.0, 6.0],
    split_frac=[0.2, 0.4],
    split_min_prompt_sec=1.0,
    spk_neighbor_k=8,
    p_drop_spk=0.3,
    p_drop_lang=0.1,
    # pack blocks: the read/cache unit (4 MB) and how many neighbouring blocks
    # the language prompt may come from (see src/sampler.py)
    block_samples=2_097_152,
    lang_block_span=1,
)


@dataclass(frozen=True)
class Draw:
    """One row's prompt decision for one epoch (all lengths in 16 kHz samples)."""

    spk_row: Optional[int]  # None = no speaker prompt
    spk_start16: int
    spk_len16: int  # 0 for split and none modes
    split_k: Optional[int]  # split mode: prompt = words[:k]
    lang_row: Optional[int]
    lang_start16: int
    lang_len16: int
    drop_spk: bool
    drop_lang: bool


class LEMASDataset(torch.utils.data.Dataset):
    """Dual-prompt training samples drawn online from a LEMAS manifest."""

    def __init__(
        self,
        split: str,
        recipe_dir=None,
        manifest_path=None,
        token_list=None,
        audio_root=None,
        load_speech: bool = True,
        prompt_config: Optional[Dict[str, Any]] = None,
        seed: int = 0,
        train: Optional[bool] = None,
    ):
        """Load the manifest columns and index rows by language and group.

        Args:
            split: ``train`` or ``valid``; selects the default manifest and,
                unless ``train`` is given, whether draws vary per epoch.
            recipe_dir: Recipe root for the default manifest path.
            manifest_path: Explicit manifest tsv (overrides the default).
            token_list: Token list file; required when samples need ``text``.
            audio_root: Pack root holding ``<lang>/<shard>.pcm`` (default from
                ``dataset/config.yaml``).
            load_speech: Skip audio when False (``create_shape``).
            prompt_config: Overrides of ``DEFAULT_PROMPT_CONFIG``.
            seed: Base seed of every draw.
            train: Force per-epoch (True) or fixed (False) draws.

        Example:
            .. code-block:: yaml

                train:
                  - data_src_args:
                      split: train
                      manifest_path: ${data_dir}/manifest/train.tsv
                      token_list: ${token_list}
                      prompt_config: ${prompt_config}

        Note:
            Validation draws ignore the epoch, so the valid loss is
            comparable across epochs, and its dropout coins are pinned too.
        """
        self.split, self.load_speech = split, load_speech
        self.train = (split == "train") if train is None else bool(train)
        self.seed, self.epoch = int(seed), 0
        cfg = dict(DEFAULT_PROMPT_CONFIG)
        cfg.update(dict(prompt_config or {}))
        self.cfg = cfg
        root = (
            Path(recipe_dir).resolve()
            if recipe_dir
            else Path(__file__).resolve().parents[1]
        )
        res = resources.files("dataset").joinpath("config.yaml")
        with resources.as_file(res) as p:
            dcfg = load_config_with_defaults(str(p), resolve=False)
        if manifest_path is None:
            manifest_path = (
                root
                / dcfg["builder"]["data_path"]
                / dcfg["dataset"]["split_manifest_paths"][split]
            )
        self.audio_root = Path(audio_root or dcfg["builder"]["audio_root"])
        self.cols = ManifestColumns.load(manifest_path)
        self.table = TokenTable(token_list) if token_list else None
        self._index_groups()

    # ---- indexes -----------------------------------------------------------
    def _index_groups(self) -> None:
        c = self.cols
        # position of every row in pack order (pack, a_start): after the
        # chunk-contiguous regroup, a speaker's neighbours in pack order are
        # its chunk mates, so partner draws stay inside the cached blocks
        order = np.lexsort((c.a_start, c.pack))
        self._pack_order = order
        self._pos = np.empty(c.n_rows, dtype=np.int64)
        self._pos[order] = np.arange(c.n_rows)
        self._pack_start = np.searchsorted(
            c.pack[order], np.arange(len(c.pack_names) + 1)
        )
        # rows of each group sorted by pack position (CSR)
        g_order = np.lexsort((self._pos, c.group))
        self._group_rows = g_order[c.group[g_order] >= 0]
        g_sorted = c.group[self._group_rows]
        n_groups = len(c.group_names)
        self._group_start = np.searchsorted(g_sorted, np.arange(n_groups + 1))
        self._pos_in_group = np.full(c.n_rows, -1, dtype=np.int64)
        self._pos_in_group[self._group_rows] = np.arange(len(self._group_rows))
        self._lang_rows = np.argsort(c.lang, kind="stable")
        self._lang_start = np.searchsorted(
            c.lang[self._lang_rows], np.arange(len(LANGS) + 1)
        )
        self.n_lang_fallback = 0  # language prompts that left the block window

    def block_of(self, idx: int) -> int:
        """Index of the pack block holding the start of row ``idx``."""
        return int(self.cols.a_start[idx]) // int(self.cfg["block_samples"])

    def set_epoch(self, epoch: int) -> None:
        """Advance the draw seed; a no-op for fixed (validation) datasets."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        """Return the number of manifest rows."""
        return self.cols.n_rows

    def _rng(self, idx: int) -> np.random.Generator:
        epoch = self.epoch if self.train else 0
        return np.random.default_rng([self.seed, epoch, int(idx)])

    # ---- draws -------------------------------------------------------------
    def _window16(self, rng, dur: float, sec_range) -> Tuple[int, int]:
        n_avail = int(dur * SRC_SR)
        want = quantize_prompt_16k(int(rng.uniform(*sec_range) * SRC_SR))
        length = max(min(want, quantize_prompt_16k(n_avail)), 512)
        start = int(rng.integers(0, n_avail - length + 1)) if n_avail > length else 0
        return start, length

    def _draw_spk(self, rng, idx: int):
        c = self.cols
        mode = int(c.spk_mode[idx])
        if mode == 0:
            return None, 0, 0, None
        if mode == 2:
            from dataset.builder import split_candidates

            ks = split_candidates(c.word_bounds(idx), float(c.dur[idx]), self.cfg)
            if not ks:  # cannot happen after the ms rounding at build; be safe
                return None, 0, 0, None
            return idx, 0, 0, int(rng.choice(ks))
        g = int(c.group[idx])
        lo, hi = self._group_start[g], self._group_start[g + 1]
        members = self._group_rows[lo:hi]
        # the k nearest group mates in pack order: for recordings these are
        # the neighbouring segments, for speakers the chunk mates; either way
        # they sit in the same 4 MB block as the target
        pos = int(self._pos_in_group[idx]) - int(lo)
        k = int(self.cfg["spk_neighbor_k"])
        members = members[max(0, pos - k) : pos + k + 1]
        cands = members[members != idx]
        row = int(rng.choice(cands))
        start, length = self._window16(
            rng, float(c.dur[row]), self.cfg["spk_prompt_sec"]
        )
        return row, start, length, None

    def _draw_lang(self, rng, idx: int):
        c = self.cols
        block = int(self.cfg["block_samples"])
        span = int(self.cfg["lang_block_span"])
        pack = int(c.pack[idx])
        b = self.block_of(idx)
        # rows of the same pack whose start lies in blocks [b - span, b + span]
        p0, p1 = int(self._pack_start[pack]), int(self._pack_start[pack + 1])
        starts = c.a_start[self._pack_order[p0:p1]]
        lo = p0 + int(np.searchsorted(starts, max(0, b - span) * block))
        hi = p0 + int(np.searchsorted(starts, (b + span + 1) * block))
        for _ in range(20):
            if hi - lo <= 1:
                break
            row = int(self._pack_order[int(rng.integers(lo, hi))])
            if row == idx or (c.group[idx] >= 0 and c.group[row] == c.group[idx]):
                continue
            start, length = self._window16(
                rng, float(c.dur[row]), self.cfg["lang_prompt_sec"]
            )
            return row, start, length
        # the window is one speaker (a giant group): any row of the language
        self.n_lang_fallback += 1
        lang = int(c.lang[idx])
        lo, hi = int(self._lang_start[lang]), int(self._lang_start[lang + 1])
        for _ in range(100):
            row = int(self._lang_rows[int(rng.integers(lo, hi))])
            if row == idx:
                continue
            if c.group[idx] >= 0 and c.group[row] == c.group[idx]:
                continue
            start, length = self._window16(
                rng, float(c.dur[row]), self.cfg["lang_prompt_sec"]
            )
            return row, start, length
        raise RuntimeError(f"no language-prompt partner for row {idx}")

    def draw(self, idx: int) -> Draw:
        """Draw both prompts and the two dropout coins for ``idx``."""
        rng = self._rng(idx)
        spk_row, s0, sl, k = self._draw_spk(rng, idx)
        lang_row, l0, ll = self._draw_lang(rng, idx)
        drop_spk = bool(rng.random() < float(self.cfg["p_drop_spk"]))
        drop_lang = bool(rng.random() < float(self.cfg["p_drop_lang"]))
        return Draw(spk_row, s0, sl, k, lang_row, l0, ll, drop_spk, drop_lang)

    # ---- audio -------------------------------------------------------------
    def _pack_fd(self, pack: int) -> Tuple[int, int]:
        # One descriptor per pack per process, opened on first use so each
        # dataloader worker (forked every epoch) opens its own; pread carries
        # no file offset, so a descriptor inherited across fork is safe too.
        fds = self.__dict__.setdefault("_fds", {})
        ent = fds.get(pack)
        if ent is None:
            fd = os.open(str(self.audio_root / self.cols.pack_names[pack]), os.O_RDONLY)
            ent = fds[pack] = (fd, os.fstat(fd).st_size // 2)
        return ent

    def _block(self, pack: int, b: int) -> np.ndarray:
        """Return block ``b`` of ``pack`` (int16), from a small per-process cache.

        One aligned 4 MB read costs ~100 ms on Delta /work/hdd against ~70 ms
        (p90 300 ms) for a random 64 KB one, so the loader reads whole blocks
        and serves every row of a batch, its speaker-prompt partner and its
        language prompt from the few blocks it has cached.
        """
        cache = self.__dict__.setdefault("_blocks", {})
        key = (pack, b)
        hit = cache.get(key)
        if hit is not None:
            return hit
        block = int(self.cfg["block_samples"])
        fd, n_samples = self._pack_fd(pack)
        n = max(0, min(block, n_samples - b * block))
        buf = os.pread(fd, 2 * n, 2 * b * block)
        arr = np.frombuffer(buf, dtype="<i2")
        if len(cache) >= int(self.cfg.get("block_cache", 6)):
            cache.pop(next(iter(cache)))  # oldest
        cache[key] = arr
        self.n_block_reads = getattr(self, "n_block_reads", 0) + 1
        return arr

    def __getstate__(self):
        """Drop the per-process descriptor and block caches when pickled."""
        state = dict(self.__dict__)
        state.pop("_fds", None)
        state.pop("_blocks", None)
        return state

    def _read16(
        self, row: int, start: int = 0, stop: Optional[int] = None
    ) -> np.ndarray:
        """Read ``[start, stop)`` samples (16 kHz) of ``row`` via the block cache."""
        c = self.cols
        a_len = int(c.a_len[row])
        stop = a_len if stop is None else min(int(stop), a_len)
        n = max(0, stop - int(start))
        block = int(self.cfg["block_samples"])
        pack = int(c.pack[row])
        s0 = int(c.a_start[row]) + int(start)
        pieces = []
        pos = s0
        while pos < s0 + n:
            b, off = divmod(pos, block)
            arr = self._block(pack, b)
            take = min(s0 + n - pos, block - off)
            pieces.append(arr[off : off + take])
            pos += take
        out = np.concatenate(pieces) if len(pieces) != 1 else pieces[0]
        return out.astype(np.float32) / 32768.0

    @staticmethod
    def _quantize16(wav16: np.ndarray) -> np.ndarray:
        """Trim a prompt window to a multiple of 512 samples (3 hops at 24 kHz).

        The manifest duration comes from the LEMAS jsonl and can overstate the
        FLAC by up to 64 samples (measured), so a window drawn up to the
        nominal end reads a few samples short and would break the frame
        alignment.
        """
        return wav16[: quantize_prompt_16k(len(wav16))]

    @staticmethod
    def _to24(wav16: np.ndarray) -> np.ndarray:
        if len(wav16) == 0:
            return np.zeros(0, dtype=np.float32)
        out = torchaudio.functional.resample(torch.from_numpy(wav16), SRC_SR, SR)
        return out.numpy().astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return ``text``, ``cond_frames`` and (unless disabled) ``speech``."""
        idx = int(idx)
        c = self.cols
        d = self.draw(idx)
        lang = LANGS[int(c.lang[idx])]
        phones = c.phones(idx).split(" ")
        spk16 = lang16 = target16 = None
        if d.split_k is not None:
            wb = c.word_bounds(idx)
            p_end16 = quantize_prompt_16k(int(wb[d.split_k - 1][1] * SRC_SR))
            t_start16 = int(wb[d.split_k][0] * SRC_SR)
            phones = [p for w in c.phones_by_word(idx)[d.split_k :] for p in w]
            if self.load_speech:
                full = self._read16(idx)
                spk16 = self._quantize16(full[:p_end16])
                target16 = full[t_start16:]
        elif d.spk_row is not None and self.load_speech:
            spk16 = self._quantize16(
                self._read16(d.spk_row, d.spk_start16, d.spk_start16 + d.spk_len16)
            )
        if self.load_speech:
            if target16 is None:
                target16 = self._read16(idx)
            lang16 = self._quantize16(
                self._read16(d.lang_row, d.lang_start16, d.lang_start16 + d.lang_len16)
            )
        spk_present = d.spk_row is not None and not d.drop_spk
        lang_present = not d.drop_lang
        if self.load_speech:
            spk24 = self._to24(spk16) if spk_present else np.zeros(0, np.float32)
            lang24 = self._to24(lang16) if lang_present else np.zeros(0, np.float32)
            sf_, lf_ = region_frames(len(spk24)), region_frames(len(lang24))
        else:  # frame counts from the draw alone (create_shape never reads audio)
            if d.split_k is not None:
                n_spk16 = quantize_prompt_16k(
                    int(c.word_bounds(idx)[d.split_k - 1][1] * SRC_SR)
                )
            else:
                n_spk16 = d.spk_len16
            sf_ = region_frames(n_spk16 * 3 // 2) if spk_present else 0
            lf_ = region_frames(d.lang_len16 * 3 // 2) if lang_present else 0
        text = (
            build_text_ids(sf_, lf_, lang, phones, self.table) if self.table else None
        )
        sample: Dict[str, Any] = {
            "cond_frames": np.asarray([cond_frames(sf_, lf_)], dtype=np.int64)
        }
        if text is not None:
            sample["text"] = text
        if self.load_speech:
            sample["speech"] = np.concatenate(
                [spk24, lang24, self._to24(target16)]
            ).astype(np.float32)
            if text is not None:
                assert len(text) <= len(sample["speech"]) // HOP + 1, (idx, len(text))
        return sample

    def frames_bound(self, hop_length: int = HOP, sample_rate: int = SR) -> np.ndarray:
        """Per-row upper bound on frames for batching, aware of the prompt mode.

        ``none``/``split`` rows never get a separate speaker-prompt region
        (the split prompt lies inside the row), so only the language prompt is
        added; ``group`` rows add both.
        """
        c = self.cols
        lang_hi = float(self.cfg["lang_prompt_sec"][1])
        spk_hi = float(self.cfg["spk_prompt_sec"][1])
        extra = np.where(c.spk_mode == 1, spk_hi + lang_hi, lang_hi)
        n = ((c.dur.astype(np.float64) + extra) * sample_rate).astype(np.int64)
        return (1 + n // hop_length).astype(np.int64)

    def n_frames(self, hop_length: int, sample_rate: int) -> np.ndarray:
        """Upper-bound frame count per row at the longest prompt layout."""
        extra = float(self.cfg["spk_prompt_sec"][1]) + float(
            self.cfg["lang_prompt_sec"][1]
        )
        n = ((self.cols.dur.astype(np.float64) + extra) * sample_rate).astype(np.int64)
        return (1 + n // hop_length).astype(np.int32)


Dataset = LEMASDataset
