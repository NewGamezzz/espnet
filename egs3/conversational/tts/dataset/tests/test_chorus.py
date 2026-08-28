"""NSF Chorus manifest parsing, text cleaning, N-channel merge, builder."""

import dataclasses
import json
from pathlib import Path

import pytest

from egs3.conversational.tts.dataset.preprocessing import chorus
from egs3.conversational.tts.dataset.preprocessing.fisher import CleanResult
from egs3.conversational.tts.dataset.preprocessing.sssd import Supervision

from .conftest import REPO_ROOT, write_flac  # noqa: F401  (sys.path setup)


def meeting_line(mid="MTG_1", split="train", speakers=None, n_frames=24000 * 6):
    speakers = speakers or {
        "Bert": [[0.5, 2.0, "hello there <ST/>"], [2.5, 3.0, "<ST/> yes"]],
        "Anna": [[1.0, 1.8, "<FILL/> um okay"]],
    }
    return {
        "meeting_id": mid,
        "split": split,
        "n_speakers": len(speakers),
        "duration": n_frames / 24000,
        "room": "ROOM_1",
        "sr": 24000,
        "speakers": {
            name: {
                "ct_device": "CT_1",
                "wav": f"{split}/{mid}/{name}.wav",
                "ipus": [[u[0], u[1]] for u in utts],
                "utterances": utts,
                "speech_seconds": sum(u[1] - u[0] for u in utts),
            }
            for name, utts in speakers.items()
        },
    }


def write_manifest(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))


def test_load_manifest_sorts_speakers_and_indexes_channels(tmp_path):
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, [meeting_line()])
    meetings = chorus.load_chorus_manifest(path)
    m = meetings["MTG_1"]
    assert m.speakers == ("Anna", "Bert")
    assert m.wavs == ("train/MTG_1/Anna.wav", "train/MTG_1/Bert.wav")
    assert m.split == "train" and m.sample_rate == 24000
    assert m.num_channels == 2
    chans = {u.speaker: u.channel for u in m.utterances}
    assert chans == {"Anna": 0, "Bert": 1}
    bert = [u for u in m.utterances if u.speaker == "Bert"]
    assert bert[0].start == 0.5 and bert[0].duration == pytest.approx(1.5)
    assert bert[0].text == "hello there <ST/>"  # raw; cleaning is separate
    assert bert[0].recording_id == "MTG_1"


def test_load_manifest_rejects_duplicate_and_bad_rate(tmp_path):
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, [meeting_line(), meeting_line()])
    with pytest.raises(ValueError, match="duplicate"):
        chorus.load_chorus_manifest(path)
    bad = meeting_line()
    bad["sr"] = 16000
    write_manifest(path, [bad])
    with pytest.raises(ValueError, match="24000"):
        chorus.load_chorus_manifest(path)


def test_load_manifest_rejects_speaker_count_mismatch(tmp_path):
    path = tmp_path / "manifest.jsonl"
    bad = meeting_line()
    bad["n_speakers"] = 5
    write_manifest(path, [bad])
    with pytest.raises(ValueError, match="n_speakers"):
        chorus.load_chorus_manifest(path)


def test_clean_strips_known_tags_and_unwraps_pname():
    assert chorus.clean_chorus_text("hello <ST/> there") == CleanResult(
        "hello there", False
    )
    assert chorus.clean_chorus_text("<FILL/> um <FILLlaugh/> ok <BA/>") == CleanResult(
        "um ok", False
    )
    assert chorus.clean_chorus_text("ask <PName>Sophie</PName> now") == CleanResult(
        "ask Sophie now", False
    )
    assert chorus.clean_chorus_text("<ISSUE/> <FL/> <PAUSE/> <SN/> go") == CleanResult(
        "go", False
    )


def test_clean_unknown_inline_word_is_dropped_but_kept():
    res = chorus.clean_chorus_text("has <UNKNOWN/> been said")
    assert res == CleanResult("has been said", False)


def test_clean_unknown_only_is_unintelligible():
    assert chorus.clean_chorus_text("<UNKNOWN/>").unintelligible
    assert chorus.clean_chorus_text("<UNKNOWN/> <ST/>").unintelligible


def test_clean_tag_only_is_benign():
    assert chorus.clean_chorus_text("<ST/>") == CleanResult("", False)
    assert chorus.clean_chorus_text("<FILLlaugh/>") == CleanResult("", False)


def test_clean_unknown_tag_raises():
    with pytest.raises(ValueError, match="<NEWTAG/>"):
        chorus.clean_chorus_text("hi <NEWTAG/> there")


def test_clean_supervisions_splits_drop_classes():
    sups = [
        Supervision("a", "MTG_1", 0, 0.0, 1.0, "fine text", "Bert"),
        Supervision("b", "MTG_1", 0, 1.0, 1.0, "<UNKNOWN/>", "Bert"),
        Supervision("c", "MTG_1", 0, 2.0, 1.0, "<ST/>", "Bert"),
        Supervision("d", "MTG_1", 0, 3.0, 1.0, "a <UNKNOWN/> b", "Bert"),
    ]
    kept, spans, n_benign = chorus.clean_chorus_supervisions(sups)
    assert [s.text for s in kept] == ["fine text", "a b"]
    assert spans == [(1.0, 2.0)]
    assert n_benign == 1


# --- merge -----------------------------------------------------------------


def make_chorus_corpus(tmp_path, meetings):
    """Corpus root with the manifest + one mono 24 kHz wav per speaker."""
    import numpy as np
    import soundfile as sf

    root = tmp_path / "chorus"
    write_manifest(root / "manifest.jsonl", meetings)
    loaded = chorus.load_chorus_manifest(root / "manifest.jsonl")
    for m in loaded.values():
        n = int(round(m.duration * 24000))
        for wav in m.wavs:
            p = root / wav
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(p), np.zeros(n, dtype=np.float32), 24000, subtype="PCM_16")
    return root, loaded


def _fake_ffmpeg_join(monkeypatch, wrong_frames_for=None):
    # The builder's Pool would not see the monkeypatch: run the merge in-process.
    monkeypatch.setitem(chorus_builder._CFG, "merge_workers", 1)

    def run(cmd, check):
        import soundfile as sf

        out = Path(cmd[-1])
        inputs = [Path(cmd[i + 1]) for i, a in enumerate(cmd) if a == "-i"]
        frames = sf.info(str(inputs[0])).frames
        mid = out.name.split(".")[0]
        if wrong_frames_for and mid in wrong_frames_for:
            frames //= 2
        write_flac(out, num_channels=len(inputs), duration_s=frames / 24000, sr=24000)

    monkeypatch.setattr(chorus.subprocess, "run", run)


def test_merged_relpath_uses_split_dir():
    m = chorus.ChorusMeeting(
        "MTG_9", "dev", 1.0, 24000, ("A",), ("dev/MTG_9/A.wav",), ()
    )
    assert chorus.merged_relpath(m) == "dev/MTG_9.flac"


def test_merge_all_joins_sorted_speakers_and_is_idempotent(tmp_path, monkeypatch):
    _fake_ffmpeg_join(monkeypatch)
    root, meetings = make_chorus_corpus(
        tmp_path, [meeting_line("MTG_1"), meeting_line("MTG_2", split="dev")]
    )
    flac_dir = tmp_path / "flac"
    assert chorus.merge_all(meetings, root, flac_dir, workers=1) == 2
    import soundfile as sf

    info = sf.info(str(flac_dir / "train/MTG_1.flac"))
    assert info.channels == 2 and info.samplerate == 24000
    assert (flac_dir / "dev/MTG_2.flac").is_file()
    assert not list(flac_dir.rglob("*.tmp"))
    assert chorus.merge_all(meetings, root, flac_dir, workers=1) == 0


def test_merge_ffmpeg_argv_orders_inputs_by_sorted_speaker(tmp_path, monkeypatch):
    seen = {}

    def run(cmd, check):
        seen["inputs"] = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-i"]
        seen["filter"] = cmd[cmd.index("-filter_complex") + 1]
        write_flac(Path(cmd[-1]), num_channels=2, duration_s=6.0, sr=24000)

    monkeypatch.setattr(chorus.subprocess, "run", run)
    root, meetings = make_chorus_corpus(tmp_path, [meeting_line("MTG_1")])
    chorus.merge_all(meetings, root, tmp_path / "flac", workers=1)
    assert [Path(p).name for p in seen["inputs"]] == ["Anna.wav", "Bert.wav"]
    assert "amerge=inputs=2" in seen["filter"]


def test_merge_frame_mismatch_raises_and_publishes_nothing(tmp_path, monkeypatch):
    _fake_ffmpeg_join(monkeypatch, wrong_frames_for={"MTG_1"})
    root, meetings = make_chorus_corpus(tmp_path, [meeting_line("MTG_1")])
    with pytest.raises(RuntimeError, match="frames"):
        chorus.merge_all(meetings, root, tmp_path / "flac", workers=1)
    assert not (tmp_path / "flac/train/MTG_1.flac").exists()


def test_merge_missing_source_raises(tmp_path):
    root, meetings = make_chorus_corpus(tmp_path, [meeting_line("MTG_1")])
    (root / "train/MTG_1/Anna.wav").unlink()
    with pytest.raises(FileNotFoundError):
        chorus.merge_all(meetings, root, tmp_path / "flac", workers=1)


def test_merge_removes_stale_tmp_files(tmp_path, monkeypatch):
    _fake_ffmpeg_join(monkeypatch)
    root, meetings = make_chorus_corpus(tmp_path, [meeting_line("MTG_1")])
    flac_dir = tmp_path / "flac"
    (flac_dir / "train").mkdir(parents=True)
    stale = flac_dir / "train/MTG_1.flac.12345.tmp"
    stale.write_bytes(b"garbage from a killed prior run")
    assert chorus.merge_all(meetings, root, flac_dir, workers=1) == 1
    assert not stale.exists()


def test_measured_durations_reads_headers_and_checks_channels(tmp_path, monkeypatch):
    _fake_ffmpeg_join(monkeypatch)
    root, meetings = make_chorus_corpus(tmp_path, [meeting_line("MTG_1")])
    flac_dir = tmp_path / "flac"
    chorus.merge_all(meetings, root, flac_dir, workers=1)
    durs = chorus.measured_durations_nch(meetings, flac_dir)
    assert durs["MTG_1"] == pytest.approx(6.0)
    bad = {
        "MTG_1": dataclasses.replace(
            meetings["MTG_1"], speakers=("A", "B", "C"), wavs=("x", "y", "z")
        )
    }
    with pytest.raises(RuntimeError, match="channels"):
        chorus.measured_durations_nch(bad, flac_dir)


# --- builder ---------------------------------------------------------------

from egs3.conversational.tts.dataset import chorus_builder  # noqa: E402
from egs3.conversational.tts.dataset.chorus_builder import ChorusBuilder  # noqa: E402
from egs3.conversational.tts.dataset.preprocessing.sessions import (  # noqa: E402
    read_session_manifest,
)


def _write_vocab(recipe_dir: Path):
    tokens = (
        [" "]
        + list("abcdefghijklmnopqrstuvwxyz")
        + ["<turn>", "<OTHER>", "<speaker_prompt>", "<prev_chunk>", "<turn_fill>"]
    )
    vp = recipe_dir / "data/tokens/vocab.txt"
    vp.parent.mkdir(parents=True, exist_ok=True)
    vp.write_text("\n".join(tokens) + "\n")


def test_builder_end_to_end_uses_given_splits(tmp_path, monkeypatch):
    _fake_ffmpeg_join(monkeypatch)
    lines = [
        meeting_line("MTG_1", "train"),
        meeting_line("MTG_2", "dev"),
        meeting_line(
            "MTG_3",
            "eval",
            speakers={
                "Cid": [[0.2, 1.0, "one <UNKNOWN/>"], [1.5, 2.0, "<UNKNOWN/>"]],
                "Dee": [[0.5, 0.9, "two"]],
                "Eve": [[3.0, 4.0, "three <ST/>"]],
            },
        ),
    ]
    root, _ = make_chorus_corpus(tmp_path, lines)
    recipe = tmp_path / "recipe"
    _write_vocab(recipe)
    flac_dir = tmp_path / "flac"
    b = ChorusBuilder()
    kw = dict(recipe_dir=recipe, dataset_root=root, chorus_flac_dir=flac_dir)
    assert not b.is_source_prepared(**kw)
    b.prepare_source(**kw)
    assert b.is_source_prepared(**kw)
    assert not b.is_built(recipe_dir=recipe)
    b.build(**kw)
    assert b.is_built(recipe_dir=recipe)
    train = read_session_manifest(recipe / "data/manifest/sessions_chorus_train.jsonl")
    valid = read_session_manifest(recipe / "data/manifest/sessions_chorus_valid.jsonl")
    test = read_session_manifest(recipe / "data/manifest/sessions_chorus_test.jsonl")
    assert [s.session_id for s in train] == ["MTG_1"]
    assert [s.session_id for s in valid] == ["MTG_2"]
    assert [s.session_id for s in test] == ["MTG_3"]
    s = train[0]
    assert s.num_channels == 2 and s.audio_relpath == "train/MTG_1.flac"
    assert s.sample_rate == 24000 and s.duration == pytest.approx(6.0)
    assert {t.speaker for t in s.turns} == {"Anna", "Bert"}
    assert all(t.text == t.text.lower() for t in s.turns)  # vocab charset
    assert sum(1 for t in s.turns if t.speaker == "Bert") == 2  # gap 0.5 > 0.2
    t3 = test[0]
    assert t3.num_channels == 3
    # <UNKNOWN/>-only utterances are dropped but never become exclusion spans
    assert t3.exclusion_spans == ()
    assert [t.text for t in t3.turns if t.speaker == "Cid"] == ["one"]


def test_builder_requires_vocab(tmp_path, monkeypatch):
    _fake_ffmpeg_join(monkeypatch)
    root, _ = make_chorus_corpus(tmp_path, [meeting_line("MTG_1")])
    b = ChorusBuilder()
    kw = dict(
        recipe_dir=tmp_path / "r", dataset_root=root, chorus_flac_dir=tmp_path / "flac"
    )
    b.prepare_source(**kw)
    with pytest.raises(RuntimeError, match="vocab"):
        b.build(**kw)


def test_builder_rejects_unknown_split(tmp_path, monkeypatch):
    _fake_ffmpeg_join(monkeypatch)
    root, _ = make_chorus_corpus(tmp_path, [meeting_line("MTG_1", split="weird")])
    recipe = tmp_path / "recipe"
    _write_vocab(recipe)
    b = ChorusBuilder()
    kw = dict(recipe_dir=recipe, dataset_root=root, chorus_flac_dir=tmp_path / "flac")
    b.prepare_source(**kw)
    with pytest.raises(ValueError, match="split"):
        b.build(**kw)
