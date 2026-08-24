"""Reformat ZipVoice-Dialog test-en into the training-style external manifest.

Source: ``dialog_testset.tar.gz`` from HF ``k2-fsa/TTS_eval_datasets``
(ZipVoice-Dialog, arXiv 2507.09318), extracted to ``dialog_testset/en/``::

    test.tsv              280 rows x 6 tab-separated columns:
                          id, prompt_text_A, prompt_text_B,
                          prompt_wav_A, prompt_wav_B, "[S1] ... [S2] ..."
    prompt_wavs/*.wav     16 files, 24 kHz STEREO with ONE active track
    ground_truth_wavs/    280 files, 24 kHz STEREO dual-track (one speaker
                          per track, tracks uncorrelated)

Output (``--out``): ``manifest.jsonl`` (one training-style record per row,
read by ``src/external_testset.py::load_external_manifest``), ``prompt/``
(the active track of each prompt, mono), ``gt/<id>_ch<k>.wav`` (one mono
file per speaker channel), id lists, and ``build_meta.json`` with the
provenance a paper table needs (archive md5, file list, every edge case).

Conventions established by inspecting the real archive (2026-08-23):

* Speaker tags are ``[S1]`` / ``[S2]``; every transcript starts with
  ``[S1]``.  A tag is a turn: consecutive same-speaker tags stay separate
  turns (2 rows), an empty segment is dropped and counted (1 row).
* 10 rows carry only ``[S1]``: monologues.  They become ONE-channel records
  with one prompt - the shape LibriTTS utterances have in training - rather
  than two-channel records with an empty channel.
* The prompt wavs' active track is L or R per file.  S1's ground-truth
  track is the active track of prompt A (verified against first-onset
  order on 264/270 two-speaker rows; the 6 exceptions are short leads of
  the other track consistent with untranscribed backchannels and are
  recorded as anomalies, mapping kept).  Prompt B must be active on the
  other track, or the row is an error.
* Text is stored RAW; the loader normalizes it against the training vocab
  exactly as the SSSD/LibriTTS builders do.  The build's dry run reports
  which characters normalization drops, so nothing is lost silently.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from egs3.conversational.tts.dataset.preprocessing.text import vocab_charset
from egs3.conversational.tts.dataset.preprocessor import read_vocab
from egs3.conversational.tts.src.external_testset import load_external_manifest

TAG_RE = re.compile(r"\[S(\d+)\]")
KNOWN_TAGS = ("S1", "S2")
ONSET_FRAME_SEC = 0.1
ONSET_RMS_THRESHOLD = 0.01


def parse_dialogue_text(text: str) -> tuple[list[tuple[int, str]], int]:
    """``"[S1] a [S2] b"`` -> ``([(0, "a"), (1, "b")], n_dropped_empty)``.

    Channel = tag index - 1 (S1 -> 0, S2 -> 1) BEFORE compaction; the caller
    compacts to the speakers actually present so a monologue is channel 0.
    """
    text = text.strip()
    first = TAG_RE.search(text)
    if first is None:
        raise ValueError(f"no speaker tag in {text[:60]!r}")
    if text[: first.start()].strip():
        raise ValueError(f"text before the first speaker tag: {text[:60]!r}")
    parts = TAG_RE.split(text)  # ["", "1", " a ", "2", " b ", ...]
    turns: list[tuple[int, str]] = []
    dropped = 0
    for i in range(1, len(parts), 2):
        tag = f"S{parts[i]}"
        if tag not in KNOWN_TAGS:
            raise ValueError(f"unknown speaker tag {tag} in {text[:60]!r}")
        seg = " ".join(parts[i + 1].split())
        if not seg:
            dropped += 1
            continue
        turns.append((KNOWN_TAGS.index(tag), seg))
    return turns, dropped


def active_track(array: np.ndarray) -> int:
    """Index of the louder track of a ``(T, C)`` array (RMS); mono -> 0."""
    if array.ndim == 1 or array.shape[1] == 1:
        return 0
    rms = np.sqrt(np.mean(np.square(array.astype(np.float64)), axis=0))
    return int(np.argmax(rms))


def _onsets(array: np.ndarray, sr: int) -> list[float | None]:
    """First frame (s) whose RMS exceeds the threshold, per track; ``None``
    for a track that never does."""
    frame = int(sr * ONSET_FRAME_SEC)
    n = array.shape[0] // frame
    out: list[float | None] = []
    for c in range(array.shape[1]):
        frames = array[: n * frame, c].reshape(n, frame).astype(np.float64)
        rms = np.sqrt(np.mean(np.square(frames), axis=1))
        hit = np.flatnonzero(rms > ONSET_RMS_THRESHOLD)
        out.append(float(hit[0] * frame / sr) if len(hit) else None)
    return out


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dropped_chars(texts: list[str], charset: frozenset[str]) -> dict[str, int]:
    """Characters ``normalize_text`` would drop, counted over ``texts``
    (mirrors its per-character rule after the F5 tokenizer pass)."""
    from espnet2.text.f5_pinyin import convert_char_to_pinyin

    counts: collections.Counter = collections.Counter()
    for text in texts:
        for c in "".join(convert_char_to_pinyin([text])[0]):
            if c in charset or c.isspace() or c.lower() in charset:
                continue
            counts[c] += 1
    return dict(sorted(counts.items()))


def _read_tsv(path: Path) -> list[list[str]]:
    rows = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        cols = line.split("\t")
        if len(cols) != 6:
            raise ValueError(f"{path}:{lineno}: expected 6 columns, got {len(cols)}")
        rows.append(cols)
    return rows


def build(
    src: str | Path,
    out: str | Path,
    token_list: str | Path,
    *,
    archive: str | Path | None = None,
) -> dict[str, Any]:
    """Build ``out`` from the extracted ``src`` (= ``dialog_testset/en``).

    Returns the ``build_meta.json`` content.  ``archive`` (the tarball) is
    optional and only used to record its md5.
    """
    src, out = Path(src), Path(out)
    tsv = src / "test.tsv"
    rows = _read_tsv(tsv)
    for sub in ("prompt", "gt"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # -- prompts: active track -> mono ------------------------------------ #
    prompt_tracks: dict[str, int] = {}
    prompt_sr: dict[str, int] = {}
    for row in rows:
        for col in (3, 4):
            name = Path(row[col]).name
            if name in prompt_tracks:
                continue
            array, sr = sf.read(str(src / "prompt_wavs" / name), always_2d=True)
            track = active_track(array)
            prompt_tracks[name] = track
            prompt_sr[name] = sr
            sf.write(str(out / "prompt" / name), array[:, track], sr, subtype="PCM_16")

    # -- rows -> records --------------------------------------------------- #
    lines: list[dict[str, Any]] = []
    meta: dict[str, Any] = {
        "n_dialogues": len(rows),
        "single_speaker_ids": [],
        "dropped_empty_segments": [],
        "consecutive_same_speaker_ids": [],
        "anomalies": [],
    }
    gt_names: list[str] = []
    for wid, ptext_a, ptext_b, pwav_a, pwav_b, text in rows:
        raw_turns, dropped = parse_dialogue_text(text)
        if dropped:
            meta["dropped_empty_segments"].append({"window_id": wid, "count": dropped})
        if any(a == b for (a, _), (b, _) in zip(raw_turns, raw_turns[1:])):
            meta["consecutive_same_speaker_ids"].append(wid)

        # Speakers present, in order of first appearance -> compact channels.
        present: list[int] = []
        for spk, _ in raw_turns:
            if spk not in present:
                present.append(spk)
        if len(present) == 1:
            meta["single_speaker_ids"].append(wid)
        channel_of = {spk: ch for ch, spk in enumerate(present)}

        name_a, name_b = Path(pwav_a).name, Path(pwav_b).name
        track_a, track_b = prompt_tracks[name_a], prompt_tracks[name_b]
        if track_a == track_b:
            raise ValueError(
                f"{wid}: prompt A ({name_a}) and prompt B ({name_b}) are active "
                f"on the same track {track_a}; the S1/S2 -> track mapping is "
                "undefined"
            )
        speaker_prompt = {0: (name_a, ptext_a), 1: (name_b, ptext_b)}
        speaker_track = {0: track_a, 1: track_b}

        gt_name = f"{wid}.wav"
        gt_names.append(gt_name)
        gt, sr = sf.read(str(src / "ground_truth_wavs" / gt_name), always_2d=True)
        if gt.shape[1] != 2:
            raise ValueError(
                f"{wid}: expected a 2-track ground truth, got {gt.shape[1]}"
            )

        onsets = _onsets(gt, sr)
        for spk in present:
            if onsets[speaker_track[spk]] is None:
                meta["anomalies"].append(
                    {
                        "window_id": wid,
                        "kind": "silent_gt_channel",
                        "speaker": KNOWN_TAGS[spk],
                        "track": speaker_track[spk],
                    }
                )
        if len(present) == 2:
            s1_track = speaker_track[present[0]]
            other = 1 - s1_track
            if onsets[s1_track] is not None and onsets[other] is not None:
                lead = onsets[s1_track] - onsets[other]
                if lead > 0:
                    meta["anomalies"].append(
                        {
                            "window_id": wid,
                            "kind": "first_onset_not_s1",
                            "s1_track": s1_track,
                            "lead_sec": round(lead, 3),
                        }
                    )

        channels = []
        for ch, spk in enumerate(present):
            gt_rel = f"gt/{wid}_ch{ch}.wav"
            sf.write(str(out / gt_rel), gt[:, speaker_track[spk]], sr, subtype="PCM_16")
            pname, ptext = speaker_prompt[spk]
            channels.append(
                {
                    "prompt_wav": f"prompt/{pname}",
                    "prompt_text": ptext.strip(),
                    "gt_wav": gt_rel,
                    "speaker": KNOWN_TAGS[spk],
                    "gt_track": speaker_track[spk],
                }
            )
        lines.append(
            {
                "window_id": wid,
                "session_id": wid.rsplit("-", 1)[0],
                "num_channels": len(present),
                "turns": [
                    {
                        "channel": channel_of[spk],
                        "speaker": KNOWN_TAGS[spk],
                        "text": seg,
                    }
                    for spk, seg in raw_turns
                ],
                "channels": channels,
            }
        )

    manifest = out / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines),
        encoding="utf-8",
    )
    ids = {
        "all": [ln["window_id"] for ln in lines],
        "2spk": [ln["window_id"] for ln in lines if ln["num_channels"] == 2],
        "1spk": [ln["window_id"] for ln in lines if ln["num_channels"] == 1],
    }
    for key, values in ids.items():
        (out / f"ids_{key}.txt").write_text(
            "".join(v + "\n" for v in values), encoding="utf-8"
        )

    # -- dry run through the real loader ----------------------------------- #
    charset = vocab_charset(read_vocab(token_list))
    records = load_external_manifest(manifest, token_list)
    raw_texts = [t["text"] for ln in lines for t in ln["turns"]] + [
        c["prompt_text"] for ln in lines for c in ln["channels"]
    ]
    n_channels = collections.Counter(str(r.num_channels) for r in records)
    meta.update(
        {
            "n_single_speaker": len(meta["single_speaker_ids"]),
            "n_dropped_empty_segments": sum(
                d["count"] for d in meta["dropped_empty_segments"]
            ),
            "prompt_tracks": prompt_tracks,
            "prompt_sample_rates": sorted(set(prompt_sr.values())),
            "source_files": {
                "tsv": str(tsv),
                "tsv_md5": _md5(tsv),
                "prompt_wavs": sorted(prompt_tracks),
                "ground_truth_wavs": gt_names,
            },
            "archive": (
                {"path": str(archive), "md5": _md5(Path(archive))}
                if archive is not None
                else None
            ),
            "loader_dry_run": {
                "n_records": len(records),
                "n_channels": dict(sorted(n_channels.items())),
                "total_gt_hours": sum(float(r.gt_duration_sec) for r in records)
                / 3600.0,
                "chars_dropped_by_normalization": _dropped_chars(raw_texts, charset),
            },
            "token_list": str(token_list),
        }
    )
    (out / "build_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return meta


def main(argv=None) -> None:
    """CLI entry point (see module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", required=True, help="extracted dialog_testset/en dir")
    parser.add_argument("--out", required=True, help="output test-set directory")
    parser.add_argument("--token_list", required=True, help="training vocab.txt")
    parser.add_argument("--archive", default=None, help="the tarball, for its md5")
    args = parser.parse_args(argv)
    meta = build(args.src, args.out, args.token_list, archive=args.archive)
    print(
        f"{meta['n_dialogues']} dialogues ({meta['n_single_speaker']} single-speaker), "
        f"{meta['loader_dry_run']['total_gt_hours']:.3f} h ground truth, "
        f"{len(meta['anomalies'])} anomalies, dropped chars "
        f"{meta['loader_dry_run']['chars_dropped_by_normalization']} -> {args.out}"
    )


if __name__ == "__main__":
    main()
