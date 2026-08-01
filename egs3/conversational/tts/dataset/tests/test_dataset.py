"""Tests for ConversationDataset, preprocessor composition, and the packed
collator (AC5-AC8)."""

import json

import pytest
import torch
from .conftest import channel_tone_hz, write_flac

from egs3.conversational.tts.dataset.dataset import (
    ConversationDataset,
    collate_conversations,
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
from egs3.conversational.tts.dataset.preprocessing.windows import (
    WindowRecord,
    to_json,
)
from egs3.conversational.tts.dataset.preprocessor import (
    ConversationalTextPreprocessor,
)

FS = 24000
SRC_SR = 48000


def make_window(session_id, num_channels, t0, t1, turns, widx=0):
    return WindowRecord(
        window_id=f"{session_id}_w{widx:05d}",
        session_id=session_id,
        audio_relpath=f"original/{session_id}_mixed.flac",
        num_channels=num_channels,
        sample_rate=SRC_SR,
        t0=t0,
        t1=t1,
        turns=tuple(turns),
    )


@pytest.fixture
def corpus(tmp_path, base_vocab):
    """Synthetic dataset root + window manifest + extended vocab on disk."""
    windows = [
        make_window(
            "sess2ch",
            2,
            10.0,
            22.0,
            [
                Turn(0, "spk_a", "good afternoon. how are you?", 10.5, 13.0),
                Turn(1, "spk_b", "good. what about you?", 13.6, 15.8),
                Turn(0, "spk_a", "good, but i have a problem", 16.4, 19.0),
            ],
        ),
        make_window(
            "sess3ch",
            3,
            5.0,
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
    with manifest.open("w", encoding="utf-8") as f:
        for w in windows:
            f.write(json.dumps(to_json(w)) + "\n")
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("\n".join(extend_vocab(base_vocab)) + "\n", encoding="utf-8")
    return {
        "root": tmp_path,
        "manifest": manifest,
        "vocab": vocab_path,
        "windows": windows,
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
        from egs3.conversational.tts.dataset.preprocessing.windows import to_json

        monologue = make_window(
            "sess2ch",
            2,
            0.0,
            8.0,
            [Turn(0, "spk_a", "a long monologue", 0.5, 7.0)],
            widx=9,
        )
        manifest = tmp_path / "manifest_with_mono.jsonl"
        lines = corpus["manifest"].read_text(encoding="utf-8").splitlines()
        lines.append(json.dumps(to_json(monologue)))
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
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
