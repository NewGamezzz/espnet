import numpy as np
from dataset.dataset import LEMASDataset
from src.layout import n_frames_total


def _ds(corpus, split="train", **kw):
    cfg = dict(
        spk_prompt_sec=[1.0, 2.0],
        lang_prompt_sec=[1.0, 2.0],
        split_frac=[0.2, 0.4],
        split_min_prompt_sec=1.0,
        spk_neighbor_k=2,
        p_drop_spk=0.0,
        p_drop_lang=0.0,
    )
    cfg.update(kw)
    return LEMASDataset(
        split=split,
        manifest_path=corpus["manifest"],
        token_list=corpus["tokens"],
        audio_root=corpus["audio"],
        prompt_config=cfg,
        seed=3,
    )


def test_draw_invariants(corpus):
    ds = _ds(corpus)
    for idx in range(len(ds)):
        d = ds.draw(idx)
        assert d.lang_row is not None and ds.cols.lang[d.lang_row] == ds.cols.lang[idx]
        assert d.lang_row != idx
        assert (
            ds.cols.group[idx] == -1 or ds.cols.group[d.lang_row] != ds.cols.group[idx]
        )
        mode = int(ds.cols.spk_mode[idx])
        if mode == 1:
            assert (
                d.spk_row not in (None, idx)
                and ds.cols.group[d.spk_row] == ds.cols.group[idx]
            )
            assert ds.cols.group[d.lang_row] != ds.cols.group[d.spk_row]
        if mode == 2:
            assert d.split_k is not None and d.spk_row == idx
        if mode == 0:
            assert d.spk_row is None


def test_neighbor_k_restricts_video_partners(corpus):
    ds = _ds(corpus, spk_neighbor_k=1)
    for e in range(20):
        ds.set_epoch(e)
        d = ds.draw(0)  # segment 0 -> only segment 1 is within k=1
        assert ds.cols.seg[d.spk_row] == 1


def test_sample_layout_and_frame_alignment(corpus):
    ds = _ds(corpus)
    s = ds[0]
    n = len(s["speech"])
    cf = int(s["cond_frames"][0])
    assert s["speech"].dtype == np.float32 and s["cond_frames"].shape == (1,)
    assert cf * 256 <= n and len(s["text"]) <= n_frames_total(n)
    ids = s["text"].tolist()
    assert (
        ids[:cf].count(5) + ids[:cf].count(6) == cf
    )  # role tokens cover every prompt frame
    assert ids[cf] == 7 and ids[cf + 1 :] == [2, 3, 4, 2]


def test_split_row_uses_remaining_words(corpus):
    ds = _ds(corpus)
    zh = [i for i in range(len(ds)) if ds.cols.spk_mode[i] == 2][0]
    s = ds[zh]
    d = ds.draw(zh)
    cf = int(s["cond_frames"][0])
    assert s["text"][cf] == 8  # <zh>
    assert len(s["text"]) - cf - 1 == 6 - d.split_k  # one phone per remaining word
    assert 1.0 <= d.split_k * 1.0 <= 0.4 * 6.0 + 1.0


def test_dropout_omits_regions_and_zero_when_both(corpus):
    ds = _ds(corpus, p_drop_spk=1.0, p_drop_lang=1.0)
    s = ds[0]
    assert int(s["cond_frames"][0]) == 0 and s["text"][0] == 7
    ds2 = _ds(corpus, p_drop_spk=1.0, p_drop_lang=0.0)
    s2 = ds2[0]
    d2 = ds2.draw(0)
    assert d2.drop_spk and not d2.drop_lang and int(s2["cond_frames"][0]) > 0
    assert set(s2["text"][: int(s2["cond_frames"][0])].tolist()) == {6}


def test_dropout_rates(corpus):
    ds = _ds(corpus, p_drop_spk=0.3, p_drop_lang=0.1)
    n_epochs = 4000 // len(ds)
    n_spk = n_lang = 0
    for e in range(n_epochs):
        ds.set_epoch(e)
        for i in range(len(ds)):
            d = ds.draw(i)
            n_spk += d.drop_spk
            n_lang += d.drop_lang
    tot = n_epochs * len(ds)
    assert abs(n_spk / tot - 0.3) < 0.03 and abs(n_lang / tot - 0.1) < 0.03


def test_epoch_changes_draws_and_valid_is_fixed(corpus):
    ds = _ds(corpus)
    a = ds.draw(1)
    ds.set_epoch(1)
    b = ds.draw(1)
    assert (a.spk_row, a.spk_start16, a.lang_row, a.lang_start16) != (
        b.spk_row,
        b.spk_start16,
        b.lang_row,
        b.lang_start16,
    )
    v = _ds(corpus, split="valid")
    x = v.draw(1)
    v.set_epoch(5)
    assert v.draw(1) == x


def test_prompt_lengths_within_config(corpus):
    ds = _ds(corpus, spk_prompt_sec=[1.0, 1.5], lang_prompt_sec=[0.5, 1.0])
    for i in range(len(ds)):
        d = ds.draw(i)
        if d.spk_row is not None and d.split_k is None:
            assert (
                1.0 * 16000 - 512 <= d.spk_len16 <= 1.5 * 16000
                and d.spk_len16 % 512 == 0
            )
        assert (
            0.5 * 16000 - 512 <= d.lang_len16 <= 1.0 * 16000 and d.lang_len16 % 512 == 0
        )


def test_n_frames_upper_bound(corpus):
    ds = _ds(corpus)
    fr = ds.n_frames(256, 24000)
    assert len(fr) == len(ds)
    assert fr[0] == 1 + int((3.0 + 2.0 + 2.0) * 24000) // 256
    assert len(ds[0]["speech"]) // 256 + 1 <= fr[0]


def test_load_speech_false_has_no_speech(corpus):
    ds = LEMASDataset(
        split="train",
        manifest_path=corpus["manifest"],
        token_list=corpus["tokens"],
        audio_root=corpus["audio"],
        load_speech=False,
    )
    assert "speech" not in ds[0] and "text" in ds[0]


def test_prompt_windows_survive_short_reads(corpus, monkeypatch):
    # LEMAS jsonl durations overstate the FLAC by up to 64 samples (measured),
    # so a window drawn up to the nominal end reads short of a 512 multiple.
    # Model that deterministically: every read returns 64 samples fewer.
    orig = LEMASDataset._read16

    def short_read(self, row, start=0, stop=None):
        return orig(self, row, start, stop)[:-64]

    monkeypatch.setattr(LEMASDataset, "_read16", short_read)
    ds = _ds(corpus)
    for epoch in range(3):
        ds.set_epoch(epoch)
        for i in range(len(ds)):
            s = ds[i]  # must not trip region_frames' hop-alignment assertion
            assert int(s["cond_frames"][0]) * 256 <= len(s["speech"])


def test_split_row_without_candidates_falls_back_to_no_speaker_prompt(corpus):
    ds = _ds(corpus, split_frac=[0.99, 0.999])  # no word boundary can qualify
    zh = [i for i in range(len(ds)) if int(ds.cols.spk_mode[i]) == 2]
    assert zh
    for i in zh:
        d = ds.draw(i)
        assert d.spk_row is None and d.split_k is None
        s = ds[i]
        assert int(s["cond_frames"][0]) > 0  # language prompt only


def test_block_cache_reads_match_the_pack_bytes_across_block_edges(corpus):
    ds = _ds(corpus, block_samples=4096)  # tiny blocks: rows span many
    c = ds.cols
    for i in range(len(ds)):
        pack = corpus["audio"] / c.pack_names[int(c.pack[i])]
        raw = np.fromfile(pack, dtype="<i2")[
            int(c.a_start[i]) : int(c.a_start[i] + c.a_len[i])
        ]
        full = ds._read16(i)
        assert np.array_equal(full, raw.astype(np.float32) / 32768.0)
        part = ds._read16(i, 1000, 9000)
        assert np.array_equal(part, raw[1000:9000].astype(np.float32) / 32768.0)
    assert ds.n_block_reads > 0 and len(ds._blocks) <= 6


def test_language_prompt_comes_from_neighbouring_blocks(corpus):
    ds = _ds(corpus, block_samples=32000, lang_block_span=1)
    c = ds.cols
    fallbacks = 0
    for epoch in range(5):
        ds.set_epoch(epoch)
        for i in range(len(ds)):
            d = ds.draw(i)
            assert int(c.pack[d.lang_row]) == int(c.pack[i])
            window = [
                j
                for j in range(len(ds))
                if int(c.pack[j]) == int(c.pack[i])
                and abs(ds.block_of(j) - ds.block_of(i)) <= 1
                and j != i
                and not (c.group[i] >= 0 and c.group[j] == c.group[i])
            ]
            if window:  # a different speaker is nearby: it must be used
                assert d.lang_row in window
            else:
                fallbacks += 1
    assert ds.n_lang_fallback == fallbacks and 0 < fallbacks < 5 * len(ds)


def test_speaker_partner_is_a_pack_neighbour(corpus):
    ds = _ds(corpus, spk_neighbor_k=1)
    ds.set_epoch(0)
    for i in range(len(ds)):
        d = ds.draw(i)
        if d.spk_row is not None and d.split_k is None:
            assert abs(int(ds._pos[d.spk_row]) - int(ds._pos[i])) <= 1
            assert int(ds.cols.group[d.spk_row]) == int(ds.cols.group[i])
