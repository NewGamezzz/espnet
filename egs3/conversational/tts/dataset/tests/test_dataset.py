"""Tests for ConversationDataset, preprocessor composition, and the packed
collator (AC5-AC8)."""

import dataclasses

import pytest
import torch
import torchaudio
from .conftest import FAKE_SESSIONS, _alternating_sups, channel_tone_hz, write_flac

from espnet2.fileio.sound_scp import soundfile_read

from egs3.conversational.tts.dataset.dataset import (
    ConversationDataset,
    collate_conversations,
)
from egs3.conversational.tts.dataset.preprocessing.chunk_task import (
    ChunkTaskPlan,
    assembled_duration,
)
from egs3.conversational.tts.dataset.preprocessing.sessions import (
    SessionRecord,
    write_session_manifest,
)
from egs3.conversational.tts.dataset.preprocessing.sssd import Turn
from egs3.conversational.tts.dataset.preprocessing.text import (
    OTHER_TOKEN,
    TURN_TOKEN,
    build_branch_texts,
    encode_tokens,
    extend_vocab,
    make_token2id,
)
from egs3.conversational.tts.dataset.preprocessing.windows import WindowRecord
from egs3.conversational.tts.dataset.preprocessor import (
    ConversationalTextPreprocessor,
)

FS = 24000
SRC_SR = 48000


def _atomic_session(session_id, num_channels, duration, turns, widx=0):
    """A session that bypasses planning: one window, verbatim turns, exactly
    like the retired hand-built window fixtures used to encode - see
    plan_session's atomic branch (t0=0.0, t1=duration, turns unchanged)."""
    return SessionRecord(
        session_id=session_id,
        audio_relpath=f"original/{session_id}_mixed.flac",
        num_channels=num_channels,
        sample_rate=SRC_SR,
        duration=duration,
        turns=tuple(turns),
        atomic=True,
        window_id=f"{session_id}_w{widx:05d}",
    )


def _expected_window(session: SessionRecord) -> WindowRecord:
    """The WindowRecord plan_session's atomic branch derives from ``session``
    (mirrored here, not imported, so the fixture stays a plain expectation
    rather than a planner-dependent computation)."""
    return WindowRecord(
        window_id=session.window_id,
        session_id=session.session_id,
        audio_relpath=session.audio_relpath,
        num_channels=session.num_channels,
        sample_rate=session.sample_rate,
        t0=0.0,
        t1=session.duration,
        turns=session.turns,
    )


def _sessions_from_fixture(root) -> list[SessionRecord]:
    """Non-atomic SessionRecords over fake_corpus's audio + FAKE_SESSIONS
    table, turns generated the same way the fixture builds its lhotse
    supervisions (``_alternating_sups``), so the real planner has enough
    turn coverage to produce several windows per session."""
    sessions = []
    for session_id, num_channels, duration in FAKE_SESSIONS:
        sups = _alternating_sups(session_id, num_channels, duration)
        turns = tuple(
            Turn(
                channel=s["channel"],
                speaker=s["speaker"],
                text=s["text"],
                start=s["start"],
                end=round(s["start"] + s["duration"], 3),
            )
            for s in sups
        )
        sessions.append(
            SessionRecord(
                session_id=session_id,
                audio_relpath=f"original/{session_id}_mixed.flac",
                num_channels=num_channels,
                sample_rate=SRC_SR,
                duration=duration,
                turns=turns,
            )
        )
    return sessions


def _make_dataset(fake_corpus, tmp_path, **kw):
    root, recipe_dir = fake_corpus["root"], fake_corpus["recipe_dir"]
    manifest = tmp_path / "sessions_train.jsonl"
    write_session_manifest(manifest, _sessions_from_fixture(root))
    defaults = dict(
        split="train",
        recipe_dir=recipe_dir,
        manifest_path=manifest,
        dataset_root=root,
        window_params={"window_min": 4.0, "window_max": 10.0, "tail_min": 2.0},
    )
    defaults.update(kw)
    return ConversationDataset(**defaults)


@pytest.fixture
def corpus(tmp_path, base_vocab):
    """Synthetic dataset root + session manifest (atomic sessions, so each
    yields exactly one deterministic window) + extended vocab on disk."""
    sessions = [
        _atomic_session(
            "sess2ch",
            2,
            22.0,
            [
                Turn(0, "spk_a", "good afternoon. how are you?", 10.5, 13.0),
                Turn(1, "spk_b", "good. what about you?", 13.6, 15.8),
                Turn(0, "spk_a", "good, but i have a problem", 16.4, 19.0),
            ],
        ),
        _atomic_session(
            "sess3ch",
            3,
            21.0,
            [
                Turn(0, "spk_a", "hello everyone", 5.5, 7.0),
                Turn(2, "spk_c", "hi hi", 7.6, 8.4),
                Turn(1, "spk_b", "good to see you", 9.0, 11.0),
            ],
            widx=0,
        ),
    ]
    write_flac(tmp_path / "original" / "sess2ch_mixed.flac", 2, 30.0)
    write_flac(tmp_path / "original" / "sess3ch_mixed.flac", 3, 25.0)
    manifest = tmp_path / "manifest.jsonl"
    write_session_manifest(manifest, sessions)
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("\n".join(extend_vocab(base_vocab)) + "\n", encoding="utf-8")
    return {
        "root": tmp_path,
        "manifest": manifest,
        "vocab": vocab_path,
        "windows": [_expected_window(s) for s in sessions],
        "sessions": sessions,
    }


def make_dataset(corpus, **kwargs):
    kwargs.setdefault("permute_channels", False)
    return ConversationDataset(
        split="valid",
        manifest_path=corpus["manifest"],
        dataset_root=corpus["root"],
        **kwargs,
    )


def make_preprocessor(corpus):
    return ConversationalTextPreprocessor(token_list=corpus["vocab"])


def load_item(ds, preprocess, idx):
    """The training-time composition: DataOrganizer calls the preprocessor as
    ``preprocessor(uid, sample)`` on top of the raw dataset item."""
    return preprocess(str(idx), ds[idx])


def dominant_hz(row: torch.Tensor, fs: int) -> float:
    spectrum = torch.fft.rfft(row).abs()
    return torch.argmax(spectrum).item() * fs / row.shape[0]


class TestRawItem:
    """The dataset is vocab-agnostic: raw turns + audio, no token ids."""

    def test_raw_contract(self, corpus):
        ds = make_dataset(corpus)
        item = ds[0]
        assert "text" not in item
        assert item["num_channels"] == 2
        assert item["speech"].shape[0] == 2
        assert item["speech"].dtype == torch.float32
        w = corpus["windows"][0]
        assert [
            (t.channel, t.speaker, t.text, t.start, t.end) for t in item["turns"]
        ] == [(t.channel, t.speaker, t.text, t.start, t.end) for t in w.turns]

    def test_turn_channels_are_row_indices_under_perm(self, corpus):
        ds = make_dataset(corpus)
        perm = [2, 0, 1]
        ds._fixed_perm = perm
        item = ds[1]
        inv = {orig: row for row, orig in enumerate(perm)}
        w = corpus["windows"][1]
        assert [t.channel for t in item["turns"]] == [inv[t.channel] for t in w.turns]
        # Everything but the channel stays verbatim.
        assert [t.text for t in item["turns"]] == [t.text for t in w.turns]

    def test_resampled_sample_count(self, corpus):
        """AC8: sample count matches round(fs * (t1 - t0)) within one sample."""
        ds = make_dataset(corpus)
        for idx, w in enumerate(corpus["windows"]):
            expected = round(FS * (w.t1 - w.t0))
            assert abs(ds[idx]["speech"].shape[1] - expected) <= 1

    def test_tone_survives_resampling(self, corpus):
        """AC8: a pure tone lands at the expected frequency after 48 -> 24 kHz."""
        ds = make_dataset(corpus)
        item = ds[1]  # 3-channel session
        for row_idx in range(3):
            measured = dominant_hz(item["speech"][row_idx], FS)
            assert measured == pytest.approx(channel_tone_hz(row_idx), abs=5.0)


class TestComposedItem:
    """Dataset + ConversationalTextPreprocessor = the training-time item."""

    def test_shapes_and_types(self, corpus):
        item = load_item(make_dataset(corpus), make_preprocessor(corpus), 0)
        assert len(item["text"]) == 2
        assert all(t.dtype == torch.long for t in item["text"])
        assert item["text"][0].shape != item["text"][1].shape or not torch.equal(
            item["text"][0], item["text"][1]
        )

    def test_text_matches_branch_construction(self, corpus):
        pre = make_preprocessor(corpus)
        item = load_item(make_dataset(corpus), pre, 0)
        w = corpus["windows"][0]
        expected = build_branch_texts(w.turns, w.num_channels)
        for i in range(w.num_channels):
            expected_ids = [pre.token2id[t] for t in expected[i]]
            assert item["text"][i].tolist() == expected_ids

    def test_valid_split_deterministic(self, corpus):
        """AC5: permutation off -> bit-identical repeated reads."""
        ds, pre = make_dataset(corpus), make_preprocessor(corpus)
        a, b = load_item(ds, pre, 0), load_item(ds, pre, 0)
        assert torch.equal(a["speech"], b["speech"])
        assert all(torch.equal(x, y) for x, y in zip(a["text"], b["text"]))
        assert torch.equal(a["perm"], torch.arange(2))

    def test_equivalent_to_previous_dataset_contract(self, corpus):
        """The composition reproduces exactly what the old __getitem__ (vocab
        held by the dataset, branch p = perm[k] encoded inline) returned."""
        perm = [2, 0, 1]
        ds = make_dataset(corpus)
        ds._fixed_perm = perm
        item = load_item(ds, make_preprocessor(corpus), 1)

        w = corpus["windows"][1]
        token2id = make_token2id(
            corpus["vocab"].read_text(encoding="utf-8").splitlines()
        )
        branch_tokens = build_branch_texts(w.turns, w.num_channels)
        expected_text = [
            torch.tensor(encode_tokens(branch_tokens[p], token2id), dtype=torch.long)
            for p in perm
        ]
        ds_id = make_dataset(corpus)
        expected_speech = ds_id[1]["speech"][perm]

        assert torch.equal(item["speech"], expected_speech)
        assert len(item["text"]) == len(expected_text)
        for got, want in zip(item["text"], expected_text):
            assert torch.equal(got, want)
        assert item["perm"].tolist() == perm


class TestPermutation:
    """AC6: permuting channels then building texts == building then permuting."""

    def test_injected_perm_consistency(self, corpus):
        pre = make_preprocessor(corpus)
        ds_id = make_dataset(corpus)
        ds_perm = make_dataset(corpus)
        perm = [2, 0, 1]
        ds_perm._fixed_perm = perm
        ref, item = load_item(ds_id, pre, 1), load_item(ds_perm, pre, 1)
        assert torch.equal(item["speech"], ref["speech"][perm])
        for k, p in enumerate(perm):
            assert torch.equal(item["text"][k], ref["text"][p])
        assert item["perm"].tolist() == perm

    def test_tone_frequency_follows_perm(self, corpus):
        ds = make_dataset(corpus)
        perm = [1, 2, 0]
        ds._fixed_perm = perm
        item = ds[1]
        for k, p in enumerate(perm):
            assert dominant_hz(item["speech"][k], FS) == pytest.approx(
                channel_tone_hz(p), abs=5.0
            )

    def test_marker_positions_unaffected_by_perm(self, corpus):
        pre = make_preprocessor(corpus)
        ds = make_dataset(corpus)
        ref = load_item(ds, pre, 1)
        ds._fixed_perm = [2, 0, 1]
        item = load_item(ds, pre, 1)
        turn_id = pre.token2id[TURN_TOKEN]
        other_id = pre.token2id[OTHER_TOKEN]
        for texts in (ref["text"], item["text"]):
            marker_positions = {
                tuple((t == turn_id).nonzero().flatten().tolist()) for t in texts
            }
            assert len(marker_positions) == 1  # identical across branches
        # <OTHER> totals are branch-dependent but perm-invariant as a multiset.
        ref_counts = sorted(int((t == other_id).sum()) for t in ref["text"])
        perm_counts = sorted(int((t == other_id).sum()) for t in item["text"])
        assert ref_counts == perm_counts

    def test_train_permutation_varies(self, corpus):
        ds = make_dataset(corpus, permute_channels=True, seed=0)
        perms = {tuple(ds[1]["perm"].tolist()) for _ in range(20)}
        assert len(perms) > 1


class TestCollator:
    """AC7: packed layout over a mixed [2, 3] batch."""

    def batch_and_items(self, corpus):
        ds, pre = make_dataset(corpus), make_preprocessor(corpus)
        items = [load_item(ds, pre, 0), load_item(ds, pre, 1)]
        return collate_conversations(items), items

    def test_packed_layout(self, corpus):
        batch, (item0, item1) = self.batch_and_items(corpus)
        assert batch["counts"] == [2, 3]
        m = sum(batch["counts"])
        assert batch["speech"].shape[0] == m
        assert batch["text"].shape[0] == m
        # Row order is conversation-contiguous: rows 0-1 from ds[0], 2-4 from ds[1].
        conv_id = torch.arange(2).repeat_interleave(torch.tensor(batch["counts"]))
        assert conv_id.tolist() == [0, 0, 1, 1, 1]
        for row, source in enumerate([*item0["speech"], *item1["speech"]]):
            n = source.shape[0]
            assert torch.equal(batch["speech"][row, :n], source)
            assert batch["speech_lengths"][row] == n
            assert batch["speech_mask"][row, :n].all()
            assert not batch["speech_mask"][row, n:].any()
            assert torch.all(batch["speech"][row, n:] == 0.0)

    def test_text_padding_convention(self, corpus):
        batch, (item0, item1) = self.batch_and_items(corpus)
        rows = [*item0["text"], *item1["text"]]
        for row, source in enumerate(rows):
            n = source.shape[0]
            assert batch["text_lengths"][row] == n
            assert torch.equal(batch["text"][row, :n], source)
            assert torch.all(batch["text"][row, n:] == -1)

    def test_works_with_dataloader(self, corpus):
        ds, pre = make_dataset(corpus), make_preprocessor(corpus)

        def collate(samples):
            return collate_conversations([pre(s["window_id"], s) for s in samples])

        loader = torch.utils.data.DataLoader(
            ds, batch_size=2, collate_fn=collate, shuffle=False
        )
        batch = next(iter(loader))
        assert batch["counts"] == [2, 3]
        assert batch["window_ids"] == ["sess2ch_w00000", "sess3ch_w00000"]


class TestMinActiveSpeakersFilter:
    """min_active_speakers drops windows with too few active speakers."""

    def _manifest_with_monologue(self, corpus, tmp_path):
        monologue = _atomic_session(
            "sess2ch",
            2,
            8.0,
            [Turn(0, "spk_a", "a long monologue", 0.5, 7.0)],
            widx=9,
        )
        manifest = tmp_path / "manifest_with_mono.jsonl"
        write_session_manifest(manifest, corpus["sessions"] + [monologue])
        return manifest

    def test_filter_drops_single_speaker_windows(self, corpus, tmp_path):
        manifest = self._manifest_with_monologue(corpus, tmp_path)
        common = dict(
            split="valid",
            manifest_path=manifest,
            dataset_root=corpus["root"],
            permute_channels=False,
        )
        unfiltered = ConversationDataset(**common)
        filtered = ConversationDataset(**common, min_active_speakers=2)
        assert len(unfiltered) == 3
        assert len(filtered) == 2
        assert all(r.num_active_speakers >= 2 for r in filtered.records)

    def test_filter_to_empty_raises(self, corpus, tmp_path):
        manifest = self._manifest_with_monologue(corpus, tmp_path)
        with pytest.raises(RuntimeError, match="active speakers"):
            ConversationDataset(
                split="valid",
                manifest_path=manifest,
                dataset_root=corpus["root"],
                permute_channels=False,
                min_active_speakers=4,
            )


def test_collate_mixed_channel_counts():
    """N=1 (LibriTTS-style) and N=2 windows pack into one batch."""
    samples = [
        {
            "window_id": "libritts_x",
            "num_channels": 1,
            "speech": torch.zeros(1, 2400),
            "text": [torch.tensor([1, 2, 3])],
        },
        {
            "window_id": "sssd_y",
            "num_channels": 2,
            "speech": torch.ones(2, 4800),
            "text": [torch.tensor([4]), torch.tensor([5, 6])],
        },
    ]
    batch = collate_conversations(samples, text_pad_value=-1)
    assert batch["counts"] == [1, 2]
    assert batch["speech"].shape == (3, 4800)
    assert batch["speech_lengths"].tolist() == [2400, 4800, 4800]
    assert batch["speech_mask"][0, 2400:].sum() == 0
    assert batch["text"].shape == (3, 3)
    assert batch["text"][1].tolist() == [4, -1, -1]


class TestSessionBacked:
    """ConversationDataset consumes session manifests: records is the frozen
    plan_windows(None) plan, and plan_windows(epoch) re-derives fresh windows
    for training epochs on demand."""

    def test_records_are_frozen_plan(self, fake_corpus, tmp_path):
        ds = _make_dataset(fake_corpus, tmp_path)
        assert ds.records == ds.plan_windows(epoch=None)

    def test_plan_windows_epoch_varies(self, fake_corpus, tmp_path):
        ds = _make_dataset(fake_corpus, tmp_path)
        plans = {tuple((w.t0, w.t1) for w in ds.plan_windows(e)) for e in range(6)}
        assert len(plans) > 1

    def test_min_active_speakers_filters_plan(self, fake_corpus, tmp_path):
        ds = _make_dataset(fake_corpus, tmp_path, min_active_speakers=2)
        plan = ds.plan_windows(3)
        assert plan  # non-vacuous: epoch 3 actually yields windows to check
        assert all(w.num_active_speakers >= 2 for w in plan)

    def test_load_window_equals_getitem(self, fake_corpus, tmp_path):
        ds = _make_dataset(fake_corpus, tmp_path, permute_channels=False)
        a, b = ds.load_window(ds.records[0]), ds[0]
        assert a["window_id"] == b["window_id"]
        assert torch.equal(a["speech"], b["speech"])

    def test_load_window_seeks_mid_file(self, fake_corpus, tmp_path):
        """Planned (non-atomic) windows exercise the t0 > 0 seek-read path in
        _load_speech, unlike the atomic fixtures in ``corpus`` above where
        t0 is always 0.0."""
        ds = _make_dataset(fake_corpus, tmp_path, permute_channels=False)
        mid_file = next(r for r in ds.records if r.t0 > 0.0)
        item = ds.load_window(mid_file)
        expected = round(FS * (mid_file.t1 - mid_file.t0))
        assert abs(item["speech"].shape[1] - expected) <= 1


# --------------------------------------------------------------------------
# Chunk-task assembly ([P | H | target]) fixtures
# --------------------------------------------------------------------------

CHUNK_SESSION_ID = "sesschunk"
CHUNK_AUDIO_RELPATH = f"original/{CHUNK_SESSION_ID}_mixed.flac"
CHUNK_SESSION_SEC = 40.0
CHUNK_T0, CHUNK_T1 = 18.0, 28.0
CHUNK_TURNS = (
    Turn(0, "spk_a", "hello there", 18.5, 21.0),
    Turn(1, "spk_b", "hi to you", 21.5, 24.0),
    Turn(0, "spk_a", "yes indeed", 25.0, 27.0),
)


def _write_noise_flac(path, num_channels, duration_s, sr):
    """Per-channel seeded white noise, one FLAC per session.

    Deliberately NOT conftest's pure tones: a 1 kHz tone at 24 kHz repeats
    every 24 samples, so a seam misaligned by any multiple of 24 would still
    compare equal in the interior.  Noise has no period, so the hop-snap /
    boundary assertions below fail on an off-by-anything.  PCM_16 quantization
    is deterministic, so two reads of the same span are bit-identical.
    """
    import numpy as np
    import soundfile as sf

    n = int(round(duration_s * sr))
    data = np.stack(
        [
            np.random.default_rng(1234 + c).uniform(-0.3, 0.3, n).astype(np.float32)
            for c in range(num_channels)
        ],
        axis=1,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr, subtype="PCM_16", format="FLAC")


def _make_dataset_with_session(tmp_path, src_sr=SRC_SR, num_channels=2, **kw):
    """One synthetic session on disk + ONE hand-built mid-session WindowRecord.

    The record is hand-built rather than planner-derived so the chunk-task
    tests can pin exact numbers on t0/t1 (and hence on the prompt/prev spans
    placed around them); the manifest still carries a real session because
    ``ConversationDataset.__init__`` insists on a non-empty frozen plan.

    ``src_sr=FS`` gives a session whose source rate already equals the training
    rate, so ``_load_span`` skips the resample entirely and every read is an
    exact slice - that is what lets the seam tests assert ``torch.equal``
    rather than a tolerance (LibriTTS sessions are natively 24 kHz, so this is
    a real config, not a test-only shortcut).
    """
    _write_noise_flac(
        tmp_path / CHUNK_AUDIO_RELPATH, num_channels, CHUNK_SESSION_SEC, src_sr
    )
    session = SessionRecord(
        session_id=CHUNK_SESSION_ID,
        audio_relpath=CHUNK_AUDIO_RELPATH,
        num_channels=num_channels,
        sample_rate=src_sr,
        duration=CHUNK_SESSION_SEC,
        turns=CHUNK_TURNS,
        atomic=True,
        window_id=f"{CHUNK_SESSION_ID}_w00000",
    )
    manifest = tmp_path / "chunk_manifest.jsonl"
    write_session_manifest(manifest, [session])
    kw.setdefault("permute_channels", False)
    ds = ConversationDataset(
        split="valid", manifest_path=manifest, dataset_root=tmp_path, **kw
    )
    record = WindowRecord(
        window_id=f"{CHUNK_SESSION_ID}_w00001",
        session_id=CHUNK_SESSION_ID,
        audio_relpath=CHUNK_AUDIO_RELPATH,
        num_channels=num_channels,
        sample_rate=src_sr,
        t0=CHUNK_T0,
        t1=CHUNK_T1,
        turns=CHUNK_TURNS,
    )
    return ds, record


def _spans_outside(record, lp, start=1.0, gap=1.0):
    """One prompt span per ORIGINAL channel, all of length ``lp``, laid out
    back-to-back in the session prefix so none of them touches the window (or,
    for the tests here, the prev slice immediately before it)."""
    return tuple(
        (round(start + c * (lp + gap), 6), round(start + c * (lp + gap) + lp, 6))
        for c in range(record.num_channels)
    )


def _read_span(tmp_path, span, channel, fs, src_sr=SRC_SR):
    """Independent reference read of one channel of ``span`` (session-absolute
    seconds), built from soundfile + torchaudio directly rather than through
    the dataset, so the assertions do not restate the implementation."""
    array, rate = soundfile_read(
        str(tmp_path / CHUNK_AUDIO_RELPATH),
        dtype="float32",
        start=round(span[0] * src_sr),
        end=round(span[1] * src_sr),
        always_2d=True,
    )
    row = torch.from_numpy(array[:, channel].copy())
    if fs != rate:
        row = torchaudio.functional.resample(row, orig_freq=rate, new_freq=fs)
    return row


class TestChunkTaskAssembly:
    """Task-6 contract: chunk-task records assemble as [P | H | target] with
    both conditioning blocks snapped to whole mel hops."""

    def test_load_window_assembles_chunk_task(self, tmp_path):
        ds, record = _make_dataset_with_session(tmp_path)
        plan = ChunkTaskPlan(
            kind="full",
            prev_span=(record.t0 - 4.0, record.t0),
            prompt_spans=_spans_outside(record, lp=3.5),
        )
        rec = dataclasses.replace(record, chunk_task=plan)
        ds._fixed_perm = [1, 0]
        s = ds.load_window(rec)
        assert s["cond_frames"] == s["prompt_frames"] + s["prev_frames"]
        assert s["prompt_frames"] == int(3.5 * ds.fs) // ds.hop
        assert s["prev_frames"] == int(4.0 * ds.fs) // ds.hop
        total = s["speech"].shape[1]
        target = total - s["cond_frames"] * ds.hop
        assert abs(target - int(rec.duration * ds.fs)) <= ds.hop
        # P rows follow the permutation: row 0 carries channel 1's prompt slice.
        expected = _read_span(tmp_path, plan.prompt_spans[1], channel=1, fs=ds.fs)
        torch.testing.assert_close(
            s["speech"][0, : s["prompt_frames"] * ds.hop],
            expected[: s["prompt_frames"] * ds.hop],
        )
        # The packer prices this record at round(fs * assembled_duration); the
        # two hop floors (P tail, H head) each shed < hop samples, so the
        # estimate may overshoot the real length by at most 2 * hop.
        estimate = round(ds.fs * assembled_duration(rec))
        assert 0 <= estimate - total <= 2 * ds.hop

    def test_load_window_infill_unchanged(self, tmp_path):
        ds, record = _make_dataset_with_session(tmp_path)
        s = ds.load_window(record)
        assert "cond_frames" not in s and "prompt_frames" not in s
        assert "prev_frames" not in s
        assert list(s) == ["window_id", "num_channels", "speech", "turns", "perm"]

    def test_head_trim_lands_t0_on_a_frame_boundary(self, tmp_path):
        """The interesting case: a prev slice whose length is NOT a whole
        number of hops, on a 24 kHz session so every read is an exact slice."""
        ds, record = _make_dataset_with_session(tmp_path, src_sr=FS)
        prev_start = record.t0 - 3.7
        plan = ChunkTaskPlan(
            kind="full",
            prev_span=(prev_start, record.t0),
            prompt_spans=_spans_outside(record, lp=3.5),
        )
        rec = dataclasses.replace(record, chunk_task=plan)
        perm = [1, 0]
        ds._fixed_perm = perm
        s = ds.load_window(rec)

        b = round(3.7 * ds.fs)  # 88800 samples of prev material
        assert s["prev_frames"] == b // ds.hop  # 346 frames, 224 samples over
        assert b % ds.hop != 0  # non-vacuous: the head trim actually fires
        head_drop = b - s["prev_frames"] * ds.hop

        cond = s["cond_frames"] * ds.hop
        for row, channel in enumerate(perm):
            # Target region starts EXACTLY at the cond_frames boundary.
            torch.testing.assert_close(
                s["speech"][row, cond:],
                _read_span(tmp_path, (record.t0, record.t1), channel, ds.fs, src_sr=FS),
                rtol=0,
                atol=0,
            )
            # H is the TAIL of the prev slice: the head is what got dropped.
            torch.testing.assert_close(
                s["speech"][row, s["prompt_frames"] * ds.hop : cond],
                _read_span(tmp_path, (prev_start, record.t0), channel, ds.fs, FS)[
                    head_drop:
                ],
                rtol=0,
                atol=0,
            )
            # P is the HEAD of that channel's prompt span (tail trimmed).
            torch.testing.assert_close(
                s["speech"][row, : s["prompt_frames"] * ds.hop],
                _read_span(tmp_path, plan.prompt_spans[channel], channel, ds.fs, FS)[
                    : s["prompt_frames"] * ds.hop
                ],
                rtol=0,
                atol=0,
            )
        assert s["speech"].shape[1] - cond == round(rec.duration * ds.fs)

    def test_prompt_only_has_no_prev_block(self, tmp_path):
        """prompt_only: prev_frames == 0 and everything after P is exactly the
        ordinary infill read of the same window (same call, same perm)."""
        ds, record = _make_dataset_with_session(tmp_path)
        plan = ChunkTaskPlan(
            kind="prompt_only",
            prev_span=None,
            prompt_spans=_spans_outside(record, lp=3.5),
        )
        ds._fixed_perm = [1, 0]
        s = ds.load_window(dataclasses.replace(record, chunk_task=plan))
        infill = ds.load_window(record)

        assert s["prev_frames"] == 0
        assert s["cond_frames"] == s["prompt_frames"]
        assert torch.equal(
            s["speech"][:, s["prompt_frames"] * ds.hop :], infill["speech"]
        )
        assert (
            s["speech"].shape[1]
            == s["prompt_frames"] * ds.hop + infill["speech"].shape[1]
        )

    def test_prompt_block_is_hop_snapped_and_rectangular(self, tmp_path):
        ds, record = _make_dataset_with_session(tmp_path, src_sr=FS)
        plan = ChunkTaskPlan(
            kind="prompt_only",
            prev_span=None,
            prompt_spans=_spans_outside(record, lp=3.3),  # 79200 samples: not a hop
        )
        s = ds.load_window(dataclasses.replace(record, chunk_task=plan))
        assert s["prompt_frames"] == round(3.3 * ds.fs) // ds.hop
        assert s["speech"].shape[0] == record.num_channels
