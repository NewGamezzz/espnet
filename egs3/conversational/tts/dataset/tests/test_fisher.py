"""Fisher manifest parsing, text cleaning, and A/B -> stereo merge machinery."""

import gzip
import json
import string
import subprocess
from pathlib import Path

import pytest

from egs3.conversational.tts.dataset.fisher_builder import _CFG, FisherBuilder
from egs3.conversational.tts.dataset.preprocessing import fisher
from egs3.conversational.tts.dataset.preprocessing.planner import (
    WindowParams,
    plan_sessions,
)
from egs3.conversational.tts.dataset.preprocessing.sessions import (
    read_session_manifest,
)
from egs3.conversational.tts.dataset.preprocessing.sssd import Supervision

from .conftest import REPO_ROOT, write_flac  # noqa: F401  (sys.path setup)


def sup(text, start=0.0, duration=1.0, channel=0, speaker="s0"):
    return Supervision(
        id=f"u-{start}",
        recording_id="fe_03_00001",
        channel=channel,
        start=start,
        duration=duration,
        text=text,
        speaker=speaker,
    )


def test_clean_keeps_plain_text():
    res = fisher.clean_fisher_text("and i generally prefer")
    assert res == fisher.CleanResult("and i generally prefer", False)


def test_clean_strips_event_tags_keeping_words():
    res = fisher.clean_fisher_text("well [laughter] you know [noise] right")
    assert res == fisher.CleanResult("well you know right", False)


def test_clean_unwraps_unclear_markers():
    res = fisher.clean_fisher_text("i think (( yeah maybe )) so")
    assert res == fisher.CleanResult("i think yeah maybe so", False)


def test_clean_maps_underscore_to_space():
    res = fisher.clean_fisher_text("watching t._v. tonight")
    assert res == fisher.CleanResult("watching t. v. tonight", False)


def test_clean_tag_only_utterance_is_benign_empty():
    res = fisher.clean_fisher_text("[laughter]")
    assert res == fisher.CleanResult("", False)
    res = fisher.clean_fisher_text("[sigh] [lipsmack]")
    assert res == fisher.CleanResult("", False)


def test_clean_empty_unclear_is_unintelligible():
    assert fisher.clean_fisher_text("(( ))").unintelligible
    assert fisher.clean_fisher_text("(( [noise] ))").unintelligible


def test_clean_foreign_and_unrepresentable_are_unintelligible():
    assert fisher.clean_fisher_text("<german (( ja wohl )) >").unintelligible
    assert fisher.clean_fisher_text("call me at 1 800").unintelligible
    assert fisher.clean_fisher_text("a. t. & t.").unintelligible
    assert fisher.clean_fisher_text("uh *huh*").unintelligible


def test_clean_double_bracket_skip_tag():
    # "[[skip]" appears in the corpus; the tag regex must consume it whole.
    res = fisher.clean_fisher_text("[[skip] and then")
    assert res == fisher.CleanResult("and then", False)


def test_clean_supervisions_partitions_and_collects_spans():
    sups = [
        sup("hello there", start=0.0, duration=2.0),
        sup("[laughter]", start=2.0, duration=0.5),  # benign drop
        sup("(( ))", start=3.0, duration=1.0),  # span
        sup("so [noise] anyway", start=5.0, duration=2.0),  # kept, cleaned
        sup("call 911 now", start=8.0, duration=1.0),  # span (digits)
    ]
    kept, spans, n_benign = fisher.clean_fisher_supervisions(sups)
    assert [s.text for s in kept] == ["hello there", "so anyway"]
    # cleaned supervisions keep their timing/channel/speaker fields
    assert kept[1].start == 5.0 and kept[1].duration == 2.0
    assert spans == [(3.0, 4.0), (8.0, 9.0)]
    assert n_benign == 1


def write_fisher_recordings(
    path: Path, recs: dict[str, float], *, swap_sources=False, mutate=None
) -> None:
    """recs: id -> duration_s. Real corpus layout/fields; sources point at a
    machine-specific scratch prefix on purpose. ``mutate`` edits each record
    dict before writing (for malformed-input tests)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for rid, duration in sorted(recs.items()):
            shard = rid.split("_")[-1][:3]
            sources = [
                {
                    "type": "file",
                    "channels": [0],
                    "source": f"/scratch/elsewhere/fisher_wavs_sidon_24k/{shard}/{rid}-A.flac",
                },
                {
                    "type": "file",
                    "channels": [1],
                    "source": f"/scratch/elsewhere/fisher_wavs_sidon_24k/{shard}/{rid}-B.flac",
                },
            ]
            if swap_sources:
                sources = sources[::-1]
            record = {
                "id": rid,
                "sources": sources,
                "sampling_rate": 24000,
                "num_samples": round(duration * 24000),
                "duration": duration,
                "channel_ids": [0, 1],
            }
            if mutate:
                mutate(record)
            f.write(json.dumps(record) + "\n")


def test_load_fisher_recordings_merged_relpath(tmp_path):
    path = tmp_path / "recordings.jsonl.gz"
    write_fisher_recordings(path, {"fe_03_00001": 608.484})
    recs = fisher.load_fisher_recordings(path)
    rec = recs["fe_03_00001"]
    # points at the MERGED stereo flac, sharded like the source; never the
    # scratch-absolute mono paths
    assert rec.audio_relpath == "000/fe_03_00001.flac"
    assert (rec.sample_rate, rec.num_channels) == (24000, 2)
    assert rec.duration == 608.484


def test_load_fisher_recordings_source_order_is_channel_driven(tmp_path):
    path = tmp_path / "recordings.jsonl.gz"
    write_fisher_recordings(path, {"fe_03_00001": 10.0}, swap_sources=True)
    rec = fisher.load_fisher_recordings(path)["fe_03_00001"]
    a, b = fisher.channel_source_relpaths(rec)
    assert a == "fisher_wavs_sidon_24k/000/fe_03_00001-A.flac"
    assert b == "fisher_wavs_sidon_24k/000/fe_03_00001-B.flac"


def _break_shard_root(r):
    # No shard directory at all: parent.name is "" when the source sits
    # directly under the filesystem root.
    r["sources"][0]["source"] = "/fe_03_00001-A.flac"
    r["sources"][1]["source"] = "/fe_03_00001-B.flac"


def _break_shard_dotdot(r):
    r["sources"][0]["source"] = "/x/../fe_03_00001-A.flac"
    r["sources"][1]["source"] = "/x/../fe_03_00001-B.flac"


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda r: r["sources"].pop(), "2 sources"),
        (lambda r: r["sources"][0]["channels"].append(1), "single-channel"),
        (
            lambda r: r["sources"][1].__setitem__("channels", [0]),
            "channels",
        ),
        (
            lambda r: r["sources"][0].__setitem__(
                "source", "/x/000/fe_03_00001-B.flac"
            ),
            "-A.flac",
        ),
        (
            lambda r: r["sources"][1].__setitem__(
                "source", "/x/999/fe_03_00001-B.flac"
            ),
            "shard",
        ),
        (
            lambda r: r.__setitem__("num_samples", r["num_samples"] + 1),
            "num_samples",
        ),
        (_break_shard_root, "valid directory"),
        (_break_shard_dotdot, "valid directory"),
    ],
)
def test_load_fisher_recordings_rejects_malformed(tmp_path, mutate, match):
    path = tmp_path / "recordings.jsonl.gz"
    write_fisher_recordings(path, {"fe_03_00001": 10.0}, mutate=mutate)
    with pytest.raises(ValueError, match=match):
        fisher.load_fisher_recordings(path)


def test_load_fisher_recordings_rejects_duplicate_id(tmp_path):
    path = tmp_path / "recordings.jsonl.gz"
    write_fisher_recordings(path, {"fe_03_00001": 10.0})
    # write_fisher_recordings keys by dict, so a duplicate id can't be
    # expressed through its `recs` mapping; append a second record by hand.
    duplicate = {
        "id": "fe_03_00001",
        "sources": [
            {
                "type": "file",
                "channels": [0],
                "source": "/scratch/elsewhere/fisher_wavs_sidon_24k/000/fe_03_00001-A.flac",
            },
            {
                "type": "file",
                "channels": [1],
                "source": "/scratch/elsewhere/fisher_wavs_sidon_24k/000/fe_03_00001-B.flac",
            },
        ],
        "sampling_rate": 24000,
        "num_samples": round(10.0 * 24000),
        "duration": 10.0,
        "channel_ids": [0, 1],
    }
    with gzip.open(path, "at") as f:
        f.write(json.dumps(duplicate) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        fisher.load_fisher_recordings(path)


def make_fisher_corpus(tmp_path, recs: dict[str, float]):
    """Corpus root with mono source flacs in the documented sidon layout."""
    root = tmp_path / "fisher"
    write_fisher_recordings(root / "m" / "recordings.jsonl.gz", recs)
    loaded = fisher.load_fisher_recordings(root / "m" / "recordings.jsonl.gz")
    for rec in loaded.values():
        for rel in fisher.channel_source_relpaths(rec):
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            write_flac(path, num_channels=1, duration_s=rec.duration, sr=24000)
    return root, loaded


def _fake_ffmpeg_merge(monkeypatch, fail_for=None, wrong_frames_for=None):
    """Replace subprocess.run: 'merges' by writing a stereo flac to the last
    argv element, sized from the -i inputs' recording id."""

    def run(cmd, check):
        out = Path(cmd[-1])
        src_a = Path(cmd[cmd.index("-i") + 1])
        rid = src_a.stem[:-2]  # strip "-A"
        if fail_for and rid in fail_for:
            raise subprocess.CalledProcessError(1, cmd)
        import soundfile as sf

        frames = sf.info(str(src_a)).frames
        if wrong_frames_for and rid in wrong_frames_for:
            frames //= 2
        write_flac(out, num_channels=2, duration_s=frames / 24000, sr=24000)
        return None

    monkeypatch.setattr(fisher.subprocess, "run", run)


def test_merge_all_writes_skips_and_is_atomic(tmp_path, monkeypatch):
    _fake_ffmpeg_merge(monkeypatch)
    root, recs = make_fisher_corpus(tmp_path, {"fe_03_00001": 8.0, "fe_03_00002": 8.0})
    flac_dir = tmp_path / "flac"
    assert fisher.merge_all(recs, root, flac_dir, workers=1) == 2
    assert (flac_dir / "000/fe_03_00001.flac").is_file()
    assert not list(flac_dir.rglob("*.tmp"))
    # idempotent: nothing rewritten on the second pass
    assert fisher.merge_all(recs, root, flac_dir, workers=1) == 0


def test_merge_all_removes_stale_tmp_files(tmp_path, monkeypatch):
    _fake_ffmpeg_merge(monkeypatch)
    root, recs = make_fisher_corpus(tmp_path, {"fe_03_00001": 8.0})
    flac_dir = tmp_path / "flac"
    (flac_dir / "000").mkdir(parents=True)
    stale = flac_dir / "000/fe_03_00001.flac.12345.tmp"
    stale.write_bytes(b"garbage from a killed prior run")
    assert fisher.merge_all(recs, root, flac_dir, workers=1) == 1
    assert not stale.exists()


def test_merge_failure_leaves_no_final_file(tmp_path, monkeypatch):
    _fake_ffmpeg_merge(monkeypatch, fail_for={"fe_03_00001"})
    root, recs = make_fisher_corpus(tmp_path, {"fe_03_00001": 8.0})
    with pytest.raises(Exception):
        fisher.merge_all(recs, root, tmp_path / "flac", workers=1)
    assert not (tmp_path / "flac/000/fe_03_00001.flac").exists()


def test_merge_frame_mismatch_raises_and_publishes_nothing(tmp_path, monkeypatch):
    _fake_ffmpeg_merge(monkeypatch, wrong_frames_for={"fe_03_00001"})
    root, recs = make_fisher_corpus(tmp_path, {"fe_03_00001": 8.0})
    with pytest.raises(RuntimeError, match="frames"):
        fisher.merge_all(recs, root, tmp_path / "flac", workers=1)
    assert not (tmp_path / "flac/000/fe_03_00001.flac").exists()


def test_merge_missing_source_raises(tmp_path, monkeypatch):
    _fake_ffmpeg_merge(monkeypatch)
    path = tmp_path / "recordings.jsonl.gz"
    write_fisher_recordings(path, {"fe_03_00001": 8.0})
    recs = fisher.load_fisher_recordings(path)
    with pytest.raises(FileNotFoundError):
        fisher.merge_all(recs, tmp_path / "empty-corpus", tmp_path / "flac", workers=1)


def write_fisher_supervisions(path: Path, sessions: dict[str, list[dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for rid, sups in sorted(sessions.items()):
            for i, s in enumerate(sups):
                f.write(
                    json.dumps(
                        {
                            "id": f"{rid}-{i:03d}",
                            "recording_id": rid,
                            "language": "English",
                            **s,
                        }
                    )
                    + "\n"
                )


def fabricate_recipe(recipe_dir: Path) -> None:
    tokens = [" "] + list(string.ascii_lowercase) + [".", ","] + ["<turn>", "<OTHER>"]
    vocab = recipe_dir / "data/tokens/vocab.txt"
    vocab.parent.mkdir(parents=True, exist_ok=True)
    vocab.write_text("\n".join(tokens) + "\n", encoding="utf-8")


def two_speaker_session(duration: float) -> list[dict]:
    sups, t, ch = [], 1.0, 0
    while t + 4.0 < duration - 1.0:
        sups.append(
            {
                "start": round(t, 2),
                "duration": 3.0,
                "channel": ch,
                "text": "hello there, how are you doing today",
                "speaker": f"spk_{ch}",
            }
        )
        t += 4.0
        ch = 1 - ch
    return sups


def fabricate_fisher(tmp_path, sessions: dict[str, tuple[float, list[dict]]]):
    """Corpus manifests in the fixed/ layout + PRE-MERGED stereo flacs
    (prepare_source not needed)."""
    root = tmp_path / "fisher"
    flac_dir = tmp_path / "fisher_flac"
    manifests = root / _CFG["manifests_subdir"]
    write_fisher_recordings(
        manifests / _CFG["recordings_file"],
        {rid: dur for rid, (dur, _s) in sessions.items()},
    )
    write_fisher_supervisions(
        manifests / _CFG["supervisions_file"],
        {rid: sups for rid, (_d, sups) in sessions.items()},
    )
    for rid, (dur, _sups) in sessions.items():
        shard = rid.split("_")[-1][:3]
        write_flac(
            flac_dir / shard / f"{rid}.flac", num_channels=2, duration_s=dur, sr=24000
        )
    return root, flac_dir


def test_fisher_builder_end_to_end(tmp_path):
    sessions = {f"fe_03_0000{i}": (40.0, two_speaker_session(40.0)) for i in range(4)}
    root, flac_dir = fabricate_fisher(tmp_path, sessions)
    recipe = tmp_path / "recipe"
    fabricate_recipe(recipe)
    builder = FisherBuilder()
    assert builder.is_source_prepared(dataset_root=root, fisher_flac_dir=flac_dir)
    builder.build(
        recipe_dir=recipe, dataset_root=root, fisher_flac_dir=flac_dir, seed=0
    )
    assert builder.is_built(recipe_dir=recipe)

    records = []
    for split in ("train", "valid", "test"):
        path = recipe / f"data/manifest/sessions_fisher_{split}.jsonl"
        assert path.is_file()
        try:
            records.extend(read_session_manifest(path))
        except RuntimeError:
            pass  # tiny fixture: a split may legitimately be empty
    assert records
    for r in records:
        assert r.num_channels == 2
        assert r.sample_rate == 24000
        # duration is the MEASURED flac duration, not a windowed span
        assert r.duration <= 40.0 + 1e-6
        # normalized text: lowercase charset survives
        assert all(t.text == t.text.lower() for t in r.turns)
        # no unintelligible utterances in this fixture: nothing to exclude
        assert r.exclusion_spans == ()


def test_fisher_builder_populates_exclusion_spans(tmp_path):
    """An empty (( )) span must land on the session's ``exclusion_spans``
    (the planner drops any window overlapping it, tested in
    ``test_planner.py::TestExclusionSpans``), while a standalone [laughter]
    must not: it is benign (tag-only), never unintelligible speech.
    """
    duration = 40.0
    sups = two_speaker_session(duration)
    sups.append(
        {
            "start": 20.0,
            "duration": 1.0,
            "channel": 0,
            "text": "(( ))",
            "speaker": "spk_0",
        }
    )
    sups.append(
        {
            "start": 25.0,
            "duration": 0.5,
            "channel": 1,
            "text": "[laughter]",
            "speaker": "spk_1",
        }
    )
    root, flac_dir = fabricate_fisher(tmp_path, {"fe_03_00001": (duration, sups)})
    recipe = tmp_path / "recipe"
    fabricate_recipe(recipe)
    FisherBuilder().build(
        recipe_dir=recipe, dataset_root=root, fisher_flac_dir=flac_dir, seed=0
    )
    records = []
    for split in ("train", "valid", "test"):
        try:
            records.extend(
                read_session_manifest(
                    recipe / f"data/manifest/sessions_fisher_{split}.jsonl"
                )
            )
        except (RuntimeError, FileNotFoundError):
            pass
    assert len(records) == 1
    # exactly the (( )) span, unrounded; the benign [laughter] contributes
    # nothing
    assert records[0].exclusion_spans == ((20.0, 21.0),)


def test_fisher_build_then_plan_drops_windows_over_unintelligible_spans(tmp_path):
    """End-to-end proof that relocating the unintelligible-span filter from
    build time into the online planner changed nothing observable: an empty
    (( )) span must still kill any planned window overlapping it, while a
    standalone [laughter] must not.

    This is the direct descendant of the pre-refactor
    ``test_fisher_builder_drops_windows_over_unintelligible_spans`` (see git
    history), rewritten for the two-stage pipeline: the OLD test called only
    ``FisherBuilder().build()`` and read pre-cut windows straight off disk;
    the NEW builder never cuts windows, so this version chains
    ``FisherBuilder().build()`` (which now only records the span on
    ``SessionRecord.exclusion_spans``) into ``plan_sessions`` (frozen,
    seed=0, epoch=None -- the same RNG seed string
    ``f"{seed}:window:{session_id}"`` the old build-time ``build_windows``
    call used, so window boundaries are unchanged) with the ratified
    ``WindowParams`` defaults (window_max=80.0, matching this fixture's old
    config values). ``TestFisherParity`` in test_parity.py is NOT this
    coverage: the committed golden Fisher fixture's sessions carry no
    unintelligible utterances (``two_speaker_session`` only ever emits
    plain, representable text), so ``exclusion_spans`` is empty on every
    golden record and the ``planner.py:118`` branch this task relocated the
    filter into is never exercised by that test. This test is the one that
    actually exercises it.

    ``duration`` is deliberately well above ``window_max`` (60.0): a session
    no longer than window_max always collapses to a single whole-session
    window (see ``select_window_spans``' tail shortcut, which never
    consults blocked/unintelligible spans), so a single window covering the
    (( )) span would make ``assert planned`` and the no-overlap assertion
    below mutually unsatisfiable. At 180s the session is planned into
    several pieces, so the piece(s) touching the (( )) span can be dropped
    while others -- including the one covering the benign [laughter] --
    survive. Neither extra utterance ever reaches ``merge_turns`` (both
    clean to empty text), so the window boundaries are unaffected by where
    these two are placed; positions 50.0 and 80.0 are the old test's
    empirically-chosen values, re-verified here to still fall in two
    DIFFERENT planned windows under the unchanged RNG seed, so the test
    actually exercises "one window dropped, another one survives" rather
    than both spans landing in the same window.
    """
    duration = 180.0
    sups = two_speaker_session(duration)
    sups.append(
        {
            "start": 50.0,
            "duration": 1.0,
            "channel": 0,
            "text": "(( ))",
            "speaker": "spk_0",
        }
    )
    sups.append(
        {
            "start": 80.0,
            "duration": 0.5,
            "channel": 1,
            "text": "[laughter]",
            "speaker": "spk_1",
        }
    )
    root, flac_dir = fabricate_fisher(tmp_path, {"fe_03_00001": (duration, sups)})
    recipe = tmp_path / "recipe"
    fabricate_recipe(recipe)
    FisherBuilder().build(
        recipe_dir=recipe, dataset_root=root, fisher_flac_dir=flac_dir, seed=0
    )
    sessions = []
    for split in ("train", "valid", "test"):
        try:
            sessions.extend(
                read_session_manifest(
                    recipe / f"data/manifest/sessions_fisher_{split}.jsonl"
                )
            )
        except (RuntimeError, FileNotFoundError):
            pass
    assert len(sessions) == 1
    assert sessions[0].exclusion_spans == ((50.0, 51.0),)
    planned, _stats = plan_sessions(sessions, params=WindowParams(), seed=0, epoch=None)
    # windows survive somewhere in the session (laughter is benign) ...
    assert planned
    # ... but none of them covers the unintelligible instant
    assert not any(w.t0 < 51.0 and 50.0 < w.t1 for w in planned)
    # ... and the benign [laughter] instant did NOT kill its window: some
    # surviving window still covers it.
    assert any(w.t0 < 80.5 and 80.0 < w.t1 for w in planned)


def test_fisher_builder_requires_vocab(tmp_path):
    sessions = {"fe_03_00001": (40.0, two_speaker_session(40.0))}
    root, flac_dir = fabricate_fisher(tmp_path, sessions)
    with pytest.raises(RuntimeError, match="vocab"):
        FisherBuilder().build(
            recipe_dir=tmp_path / "empty-recipe",
            dataset_root=root,
            fisher_flac_dir=flac_dir,
            seed=0,
        )


def test_fisher_builder_uses_measured_duration_not_manifest(tmp_path):
    """Manifest lies about duration; windows must respect the real audio."""
    sessions = {"fe_03_00001": (999.0, two_speaker_session(35.0))}
    root, flac_dir = fabricate_fisher(tmp_path, sessions)
    # overwrite the flac with the TRUE 35 s audio
    write_flac(
        flac_dir / "000/fe_03_00001.flac", num_channels=2, duration_s=35.0, sr=24000
    )
    recipe = tmp_path / "recipe"
    fabricate_recipe(recipe)
    FisherBuilder().build(
        recipe_dir=recipe, dataset_root=root, fisher_flac_dir=flac_dir, seed=0
    )
    records = []
    for split in ("train", "valid", "test"):
        try:
            records.extend(
                read_session_manifest(
                    recipe / f"data/manifest/sessions_fisher_{split}.jsonl"
                )
            )
        except (RuntimeError, FileNotFoundError):
            pass
    assert records
    assert all(r.duration <= 35.0 + 1e-6 for r in records)
