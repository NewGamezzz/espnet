"""Seed-TTS eval sets are split into test-en, test-zh and test-hard."""

import pytest

from egs3.emilia.tts.local.prepare_seedtts_eval import prepare_seedtts


@pytest.fixture
def testset(tmp_path):
    root = tmp_path / "seedtts_testset"
    for lang, n in (("en", 3), ("zh", 2)):
        d = root / lang
        (d / "wavs").mkdir(parents=True)
        (d / "prompt-wavs").mkdir(parents=True)
        lines = []
        for i in range(n):
            (d / "wavs" / f"{lang}_{i}.wav").write_bytes(b"\x00")
            (d / "prompt-wavs" / f"p_{lang}_{i}.wav").write_bytes(b"\x00")
            lines.append(
                f"{lang}_{i}|prompt text {i}|prompt-wavs/p_{lang}_{i}.wav|"
                f"target text {i}"
            )
        (d / "meta.lst").write_text("\n".join(lines) + "\n", encoding="utf-8")
    hard = ["zh_h0|ph|prompt-wavs/p_zh_0.wav|hard target"]
    (root / "zh" / "hardcase.lst").write_text("\n".join(hard) + "\n", encoding="utf-8")
    return root


def test_three_eval_sets_are_written(tmp_path, testset):
    counts = prepare_seedtts(testset, tmp_path / "out")
    assert counts == {"test_en": 3, "test_zh": 2, "test_hard": 1}
    for name in ("test_en", "test_zh", "test_hard"):
        assert (tmp_path / "out" / name / "meta.tsv").is_file()


def test_hardcase_is_not_merged_into_test_zh(tmp_path, testset):
    """Regression for the staged prep, which merged them (spec section 7)."""
    prepare_seedtts(testset, tmp_path / "out")
    zh = (tmp_path / "out" / "test_zh" / "meta.tsv").read_text("utf-8")
    assert "zh_h0" not in zh
    hard = (tmp_path / "out" / "test_hard" / "meta.tsv").read_text("utf-8")
    assert "zh_h0" in hard


def test_prompt_paths_are_absolute(tmp_path, testset):
    prepare_seedtts(testset, tmp_path / "out")
    row = (tmp_path / "out" / "test_en" / "meta.tsv").read_text("utf-8").splitlines()[0]
    _utt, wav, _text, prompt_wav, _prompt_text = row.split("\t")
    assert wav.startswith("/") and prompt_wav.startswith("/")
