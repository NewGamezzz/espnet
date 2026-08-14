"""Training-path vs inference-path pinyin id parity.

Triaged minor (final whole-branch review): the training path
(`espnet2.text.f5_preprocessor.F5PinyinPreprocessor`, used by
`conf/training_f5_tts_base.yaml`'s `dataset.preprocessor`) and the
inference path (`espnet2.text.f5_pinyin.text_to_pinyin_ids`, called
directly by `espnet2/tts/f5/inference.py`) were probe-verified to agree,
but that verification had no regression test. Both call the same
underlying `text_to_pinyin_ids` today, so this test is a structural
tautology as written -- its value is as a trip-wire: if either path is
ever refactored to add extra processing (e.g. F5PinyinPreprocessor
double-tokenizing, or stripping/normalizing text before conversion), this
fails immediately instead of the drift surfacing as silent checkpoint
incompatibility at inference months later.

Needs `downloads/vocab.txt` (gitignored, undocumented until IMPORTANT 2's
README fix); skips cleanly, not failing, when it is absent.
"""

from pathlib import Path

import numpy as np
import pytest

VOCAB_FILE = Path(__file__).resolve().parents[1] / "downloads" / "vocab.txt"

pytestmark = pytest.mark.skipif(
    not VOCAB_FILE.is_file(),
    reason=f"{VOCAB_FILE} not present; see README's vocab provenance section",
)


@pytest.mark.parametrize(
    "text",
    [
        "hello world",
        "你好，世界",
        "mixed 你好 text with numbers 123",
        "",
    ],
)
def test_training_and_inference_paths_produce_identical_ids(text):
    from espnet2.text.f5_pinyin import load_vocab_char_map, text_to_pinyin_ids
    from espnet2.text.f5_preprocessor import F5PinyinPreprocessor

    vocab_char_map = load_vocab_char_map(str(VOCAB_FILE))
    expected = text_to_pinyin_ids(text, vocab_char_map)

    preprocessor = F5PinyinPreprocessor(vocab_file=str(VOCAB_FILE))
    actual = preprocessor({"text": text})["text"]

    np.testing.assert_array_equal(actual, expected)


def test_vocab_file_matches_documented_shape():
    """The verification an operator should run before trusting the vocab
    (README, IMPORTANT 2): 2545 lines, no <unk>/<sos/eos>/<blank>, index 0
    is a literal space."""
    lines = VOCAB_FILE.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2545
    assert lines[0] == " "
    for forbidden in ("<unk>", "<sos/eos>", "<blank>"):
        assert forbidden not in lines
