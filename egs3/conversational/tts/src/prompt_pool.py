"""External prompt pool for the single-call path: the CoVoMix2 protocol on
AMI windows (design note "Beyond Two Speakers Evaluation on AMI", prompt
section, 2026-09-06).

Instead of cutting each participant's voice reference from their own AMI
headset (bleed, a 35 dB level spread across headsets, and sub-second
backchannel fallbacks), every window draws K DISTINCT speakers from a clean
one-channel corpus manifest (LibriTTS test-clean) - one in-band utterance
each, seeded per WINDOW id so the draw is reproducible and identical across
modes, and gender-matched to the AMI participant when a ``SPEAKERS.txt`` is
given (AMI participant ids encode sex in their first letter: ``FEE013``,
``MEO015``).  The draw lives here so the frozen-manifest writer and the
seeded inference path call the SAME function (draw once, use twice).

Pool rows are the training-style one-channel manifest shape
(``num_channels == 1``, one turn): the utterance is ``channels[0].gt_wav``
and its transcript ``turns[0].text`` (``channels[0].prompt_wav`` is a
DIFFERENT utterance of the same speaker and is not used).

Lifted from the Fisher long-form builder's pool code (2026-09-04) with two
changes: the draw is keyed by window id, and the result is a ``PoolTurn``
the inference path can treat like any prompt turn.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import soundfile as sf
import torch
import torchaudio

from egs3.conversational.tts.dataset.preprocessing.sssd import Turn

# AMI participant ids: sex letter, role letters, three digits (FEE013, MEO015).
_AMI_ID = re.compile(r"^([FM])[A-Z]{2}\d{3}$")


@dataclass(frozen=True)
class PoolTurn(Turn):
    """A prompt whose audio is an external utterance rather than a span of the
    session file.  ``channel`` is the SOURCE channel it prompts (row space
    after the usual remap), ``start``/``end`` are ``0``/``duration`` so the
    meta and the duration rule read like a corpus turn; ``wav`` is absolute."""

    wav: str
    pool_id: str
    gender: str | None = None


@dataclass(frozen=True)
class PoolItem:
    speaker: str
    wav: Path
    text: str
    source_id: str
    duration: float


def ami_gender(speaker_id: str | None) -> str | None:
    """``F``/``M`` from an AMI participant id, ``None`` for anything else."""
    if not speaker_id:
        return None
    m = _AMI_ID.match(str(speaker_id))
    return m.group(1) if m else None


def load_prompt_pool(manifest, band: tuple[float, float]) -> list[PoolItem]:
    """One-channel external manifest -> in-band pool items.  Paths resolve
    against the manifest directory; durations come from the audio headers."""
    manifest = Path(manifest)
    items: list[PoolItem] = []
    for line in manifest.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if int(r["num_channels"]) != 1 or len(r["turns"]) != 1:
            raise ValueError(
                f"{manifest}: pool rows must be one-channel single-turn, got {r['window_id']}"
            )
        ch = r["channels"][0]
        wav = Path(ch["gt_wav"])
        if not wav.is_absolute():
            wav = manifest.parent / wav
        info = sf.info(str(wav))
        dur = info.frames / info.samplerate
        if band[0] - 1e-6 <= dur <= band[1] + 1e-6:
            items.append(
                PoolItem(
                    str(ch.get("speaker") or r["turns"][0]["speaker"]),
                    wav.resolve(),
                    r["turns"][0]["text"],
                    r["window_id"],
                    dur,
                )
            )
    if not items:
        raise ValueError(f"{manifest}: no pool utterance inside the {band} s band")
    return items


def load_pool_genders(path) -> dict[str, str]:
    """LibriTTS/LibriSpeech ``SPEAKERS.txt``: ``ID | SEX | SUBSET | ...`` -> {id: sex}."""
    out: dict[str, str] = {}
    for line in Path(path).read_text("utf-8").splitlines():
        if not line.strip() or line.startswith(";"):
            continue
        cols = [c.strip() for c in line.split("|")]
        if len(cols) >= 2:
            out[cols[0]] = cols[1].upper()
    return out


def draw_prompts(
    pool: Sequence[PoolItem],
    key: str,
    num_channels: int,
    seed: Any,
    *,
    channel_genders: Sequence[str | None] | None = None,
    pool_genders: dict[str, str] | None = None,
) -> list[PoolItem]:
    """K distinct pool speakers for one window, one utterance each, seeded by
    ``f"{seed}:{key}"``.  Gender-matched per channel when both gender sources
    are given, the channel's gender is known and a speaker of that gender is
    left; otherwise the draw is over every remaining speaker."""
    rng = random.Random(f"{seed}:{key}")
    by_spk: dict[str, list[PoolItem]] = {}
    for it in pool:
        by_spk.setdefault(it.speaker, []).append(it)
    remaining = sorted(by_spk)
    if len(remaining) < num_channels:
        raise ValueError(f"{key}: pool has {len(remaining)} speakers, need {num_channels}")
    chosen: list[PoolItem] = []
    for ch in range(num_channels):
        cands = remaining
        if channel_genders is not None and pool_genders is not None:
            want = channel_genders[ch]
            if want:
                matched = [s for s in remaining if pool_genders.get(s, "").upper() == str(want).upper()]
                if matched:
                    cands = matched
        spk = rng.choice(cands)
        remaining = [s for s in remaining if s != spk]
        chosen.append(rng.choice(sorted(by_spk[spk], key=lambda it: it.source_id)))
    return chosen


def channel_genders_from_turns(turns, source_channels: Sequence[int]) -> list[str | None]:
    """Per source channel, the gender of the participant speaking on it
    (AMI id letter), or ``None`` when no turn names a recognisable speaker."""
    spk: dict[int, str] = {}
    for t in turns:
        spk.setdefault(int(t.channel), str(t.speaker))
    return [ami_gender(spk.get(int(c))) for c in source_channels]


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class PromptPool:
    """The configured pool (``prompt.pool`` block): manifest, duration band,
    optional ``SPEAKERS.txt`` for gender matching, and the draw seed."""

    def __init__(self, manifest, band, seed=0, speakers_txt=None):
        self.manifest = Path(manifest)
        self.band = (float(band[0]), float(band[1]))
        self.seed = seed
        self.speakers_txt = Path(speakers_txt) if speakers_txt else None
        self.items = load_prompt_pool(self.manifest, self.band)
        self.genders = load_pool_genders(self.speakers_txt) if self.speakers_txt else None

    @classmethod
    def from_config(cls, pool_cfg) -> "PromptPool | None":
        """``None`` when the block is absent/null (corpus prompts as before)."""
        if not pool_cfg or not pool_cfg.get("manifest"):
            return None
        return cls(
            pool_cfg["manifest"],
            tuple(pool_cfg.get("band", (2.5, 3.5))),
            seed=pool_cfg.get("seed", 0),
            speakers_txt=pool_cfg.get("speakers_txt"),
        )

    def provenance(self) -> dict[str, Any]:
        return {
            "manifest": str(self.manifest),
            "manifest_md5": _md5(self.manifest),
            "speakers_txt": str(self.speakers_txt) if self.speakers_txt else None,
            "band": list(self.band),
            "seed": self.seed,
            "num_items": len(self.items),
            "num_speakers": len({it.speaker for it in self.items}),
        }

    def draw(self, window_id: str, source_channels: Sequence[int], turns) -> list[PoolTurn]:
        """One ``PoolTurn`` per source channel, in the given channel order."""
        genders = channel_genders_from_turns(turns, source_channels) if self.genders else None
        items = draw_prompts(
            self.items,
            window_id,
            len(source_channels),
            self.seed,
            channel_genders=genders,
            pool_genders=self.genders,
        )
        return [
            PoolTurn(
                channel=int(ch),
                speaker=it.speaker,
                text=it.text,
                start=0.0,
                end=round(it.duration, 6),
                wav=str(it.wav),
                pool_id=it.source_id,
                gender=(self.genders or {}).get(it.speaker),
            )
            for ch, it in zip(source_channels, items)
        ]


def pool_turn_from_entry(entry: dict, channel: int) -> PoolTurn:
    """A frozen-manifest prompt entry carrying ``wav`` -> ``PoolTurn``.  The
    file must exist: a pool that moved under the manifest is an error."""
    wav = Path(entry["wav"])
    if not wav.is_file():
        raise FileNotFoundError(f"pool prompt {entry.get('pool_id')} missing: {wav}")
    info = sf.info(str(wav))
    return PoolTurn(
        channel=int(channel),
        speaker=str(entry.get("speaker", "")),
        text=str(entry["text"]),
        start=0.0,
        end=round(info.frames / info.samplerate, 6),
        wav=str(wav),
        pool_id=str(entry.get("pool_id", wav.stem)),
        gender=entry.get("gender"),
    )


def pool_entry(t: PoolTurn) -> dict[str, Any]:
    """The frozen-manifest record of a pool prompt (no span: the audio is a file)."""
    return {
        "channel": int(t.channel),
        "pool_id": t.pool_id,
        "wav": t.wav,
        "text": t.text,
        "speaker": t.speaker,
        "gender": t.gender,
    }


def read_pool_prompt(wav: str | Path, target_fs: int) -> torch.Tensor:
    """Mono utterance at ``target_fs`` (resampled only if the rate differs), ``(T,)``."""
    array, rate = sf.read(str(wav), dtype="float32", always_2d=True)
    mono = torch.from_numpy(array[:, 0].copy())
    if rate != target_fs:
        mono = torchaudio.functional.resample(mono, orig_freq=rate, new_freq=target_fs)
    return mono
