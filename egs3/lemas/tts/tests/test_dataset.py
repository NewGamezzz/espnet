import numpy as np
import pytest
import soundfile as sf

from dataset.dataset import LEMASDataset
from dataset.manifest import ManifestRow, write_manifest
from src.layout import n_frames_total

TOKENS = ["<blank>", "<unk>", "a", "b", "<space>", "<spk>", "<lang>", "<de>", "<zh>", "<sos/eos>"]


@pytest.fixture
def corpus(tmp_path):
    audio = tmp_path / "audio"
    rows = []

    def clip(rel, sec):
        p = audio / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        wav = 0.1 * np.sin(np.arange(int(sec * 16000)) * 0.05).astype(np.float32)
        sf.write(p, wav, 16000, format="FLAC", subtype="PCM_16")

    # de: one video group with 4 segments, one singleton video, one mls speaker with 2 rows
    for i in range(4):
        clip(f"de/d/v1_{i}.flac", 3.0)
        rows.append(ManifestRow(f"de_vidAAAAAAAA-0000{i}-00000000-00000300", f"de/d/v1_{i}.flac",
                                "a b <space> a", "de", "yodas", "vidAAAAAAAA", 3.0, "j", 0, "group", "", ""))
    clip("de/d/v2_0.flac", 2.0)
    rows.append(ManifestRow("de_vidBBBBBBBB-00000-00000000-00000200", "de/d/v2_0.flac", "b", "de", "yodas",
                            "vidBBBBBBBB", 2.0, "j", 0, "none", "", ""))
    for i in range(2):
        clip(f"de/d/m_{i}.flac", 8.0)
        rows.append(ManifestRow(f"de_77_1_00000{i}", f"de/d/m_{i}.flac", "a a", "de", "mls", "77", 8.0,
                                "j", 0, "group", "", ""))
    # zh: split rows (no group)
    for i in range(2):
        clip(f"zh/z/s_{i}.flac", 6.0)
        rows.append(ManifestRow(f"zh_emilia_zh_000000000{i}", f"zh/z/s_{i}.flac", "a b a b a b", "zh",
                                "emilia", "", 6.0, "j", 0, "split",
                                "0.1:1.0,1.1:2.0,2.1:3.0,3.1:4.0,4.1:5.0,5.1:5.9", "a|b|a|b|a|b"))
    m = tmp_path / "train.tsv"
    write_manifest(rows, m)
    tok = tmp_path / "tokens.txt"
    tok.write_text("\n".join(TOKENS) + "\n")
    return dict(manifest=m, tokens=tok, audio=audio)


def _ds(corpus, split="train", **kw):
    cfg = dict(spk_prompt_sec=[1.0, 2.0], lang_prompt_sec=[1.0, 2.0], split_frac=[0.2, 0.4],
               split_min_prompt_sec=1.0, spk_neighbor_k=2, p_drop_spk=0.0, p_drop_lang=0.0)
    cfg.update(kw)
    return LEMASDataset(split=split, manifest_path=corpus["manifest"], token_list=corpus["tokens"],
                        audio_root=corpus["audio"], prompt_config=cfg, seed=3)


def test_draw_invariants(corpus):
    ds = _ds(corpus)
    for idx in range(len(ds)):
        d = ds.draw(idx)
        assert d.lang_row is not None and ds.cols.lang[d.lang_row] == ds.cols.lang[idx]
        assert d.lang_row != idx
        assert ds.cols.group[idx] == -1 or ds.cols.group[d.lang_row] != ds.cols.group[idx]
        mode = int(ds.cols.spk_mode[idx])
        if mode == 1:
            assert d.spk_row not in (None, idx) and ds.cols.group[d.spk_row] == ds.cols.group[idx]
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
    assert ids[:cf].count(5) + ids[:cf].count(6) == cf  # role tokens cover every prompt frame
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
        b.spk_row, b.spk_start16, b.lang_row, b.lang_start16)
    v = _ds(corpus, split="valid")
    x = v.draw(1)
    v.set_epoch(5)
    assert v.draw(1) == x


def test_prompt_lengths_within_config(corpus):
    ds = _ds(corpus, spk_prompt_sec=[1.0, 1.5], lang_prompt_sec=[0.5, 1.0])
    for i in range(len(ds)):
        d = ds.draw(i)
        if d.spk_row is not None and d.split_k is None:
            assert 1.0 * 16000 - 512 <= d.spk_len16 <= 1.5 * 16000 and d.spk_len16 % 512 == 0
        assert 0.5 * 16000 - 512 <= d.lang_len16 <= 1.0 * 16000 and d.lang_len16 % 512 == 0


def test_n_frames_upper_bound(corpus):
    ds = _ds(corpus)
    fr = ds.n_frames(256, 24000)
    assert len(fr) == len(ds)
    assert fr[0] == 1 + int((3.0 + 2.0 + 2.0) * 24000) // 256
    assert len(ds[0]["speech"]) // 256 + 1 <= fr[0]


def test_load_speech_false_has_no_speech(corpus):
    ds = LEMASDataset(split="train", manifest_path=corpus["manifest"], token_list=corpus["tokens"],
                      audio_root=corpus["audio"], load_speech=False)
    assert "speech" not in ds[0] and "text" in ds[0]
