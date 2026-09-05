"""LEMAS training dataset: online prompt draws + dual-prompt layout (spec 3.4, 5).

Every row draws its two prompt partners at access time, seeded by
``(seed, epoch, row)``: reproducible within a run, fresh every epoch. The
sample is ``[speaker prompt | language prompt | target]`` at 24 kHz with a text
lane of ``<spk>`` and ``<lang>`` repeated per prompt frame, the language tag,
and the target phones. Dropout is omission of a region, so ``cond_frames``
shrinks with the row and reaches 0 when both prompts are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
from dataset.keys import SOURCES, is_recording_group
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
            audio_root: FLAC root (default from ``dataset/config.yaml``).
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
        order = np.lexsort((c.seg, c.group))  # rows sorted by (group, seg)
        self._group_rows = order[c.group[order] >= 0]
        g_sorted = c.group[self._group_rows]
        n_groups = len(c.group_names)
        self._group_start = np.searchsorted(g_sorted, np.arange(n_groups + 1))
        self._pos_in_group = np.full(c.n_rows, -1, dtype=np.int64)
        self._pos_in_group[self._group_rows] = np.arange(len(self._group_rows))
        self._lang_rows = np.argsort(c.lang, kind="stable")
        self._lang_start = np.searchsorted(
            c.lang[self._lang_rows], np.arange(len(LANGS) + 1)
        )

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
            return idx, 0, 0, int(rng.choice(ks))
        g = int(c.group[idx])
        lo, hi = self._group_start[g], self._group_start[g + 1]
        members = self._group_rows[lo:hi]
        if is_recording_group(SOURCES[int(c.source[idx])]):
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
    def _read16(
        self, row: int, start: int = 0, stop: Optional[int] = None
    ) -> np.ndarray:
        path = self.audio_root / self.cols.audio(row)
        wav, sr = sf.read(str(path), start=start, stop=stop, dtype="float32")
        assert sr == SRC_SR, (path, sr)
        return wav if wav.ndim == 1 else wav.mean(axis=1)

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
                spk16, target16 = full[:p_end16], full[t_start16:]
        elif d.spk_row is not None and self.load_speech:
            spk16 = self._read16(d.spk_row, d.spk_start16, d.spk_start16 + d.spk_len16)
        if self.load_speech:
            if target16 is None:
                target16 = self._read16(idx)
            lang16 = self._read16(
                d.lang_row, d.lang_start16, d.lang_start16 + d.lang_len16
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

    def n_frames(self, hop_length: int, sample_rate: int) -> np.ndarray:
        """Upper-bound frame count per row at the longest prompt layout."""
        extra = float(self.cfg["spk_prompt_sec"][1]) + float(
            self.cfg["lang_prompt_sec"][1]
        )
        n = ((self.cols.dur.astype(np.float64) + extra) * sample_rate).astype(np.int64)
        return (1 + n // hop_length).astype(np.int32)


Dataset = LEMASDataset
