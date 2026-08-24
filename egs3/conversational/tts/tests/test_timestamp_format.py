"""Format-identity contract for ``text_format: timestamps``.

``golden/timestamp_format.json`` is a BYTE-IDENTICAL copy of the training
branch's fixture (PR #42, ``dataset/tests/golden/timestamp_format.json``).
It pins two Mode T samples built through the training assembly path: an
ordinary infill window and a chunk-task "full" window.  This test rebuilds
both through the INFERENCE-side pieces (the ported preprocessor routing,
fed the same target keys the round loop will feed it) and asserts the
identical payload.  Failure means the inference format drifted from
training: fix the code, never regenerate the fixture here.
"""

from __future__ import annotations

import json
import string
from pathlib import Path

from egs3.conversational.tts.dataset.preprocessing.sssd import Turn
from egs3.conversational.tts.dataset.preprocessing.text import (
    OTHER_TOKEN,
    PREV_CHUNK_TOKEN,
    SPEAKER_PROMPT_TOKEN,
    TURN_FILL_TOKEN,
    TURN_TOKEN,
    extend_vocab,
    make_token2id,
)
from egs3.conversational.tts.dataset.preprocessor import (
    ConversationalTextPreprocessor,
)

GOLDEN_PATH = Path(__file__).resolve().parent / "golden" / "timestamp_format.json"

# The training fixture's synthetic scenario, restated.  Every number is the
# contract (see the training test's docstring for why each boundary sits
# off a rounding tie).
NUM_CHANNELS = 2
BASE_VOCAB = (
    ["<blank>", "<unk>", "<space>"]
    + list(string.ascii_lowercase)
    + [".", ",", "?", "!", "'", "<sos/eos>"]
)
INFILL_T0, INFILL_FRAMES = 20.0, 750  # 8.0 s window, exact
INFILL_TURNS = (
    Turn(0, "spk_a", "hello there", 20.5, 22.5),
    Turn(1, "spk_b", "how are you", 23.0, 25.5),
    Turn(0, "spk_a", "i am fine", 26.1, 27.5),
)
CHUNK_T0, CHUNK_FRAMES = 5.0, 600  # 6.4 s target, exact
CHUNK_TURNS = (
    Turn(0, "spk_a", "yes ok", 5.5, 6.8),
    Turn(1, "spk_b", "sure thing", 7.1, 8.5),
    Turn(0, "spk_a", "got it", 9.5, 10.9),
)
CHUNK_PROMPT_FRAMES, CHUNK_PREV_FRAMES = 240, 300


def _branch_payload(ids, token2id):
    return {
        "length": len(ids),
        "first_8": ids[:8],
        "last_8": ids[-8:],
        "turn_indices": [i for i, x in enumerate(ids) if x == token2id[TURN_TOKEN]],
        "other_count": sum(1 for x in ids if x == token2id[OTHER_TOKEN]),
        "turn_fill_count": sum(1 for x in ids if x == token2id[TURN_FILL_TOKEN]),
    }


def test_inference_mode_t_matches_training_golden(tmp_path):
    # The eval branch's extend_vocab appends the 4-token generation; the
    # timestamp-era vocab adds <turn_fill> last, exactly as PR #42 did.
    vocab_tokens = extend_vocab(BASE_VOCAB) + [TURN_FILL_TOKEN]
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("\n".join(vocab_tokens) + "\n", encoding="utf-8")
    token2id = make_token2id(vocab_tokens)
    pre = ConversationalTextPreprocessor(token_list=vocab_path)

    infill = pre(
        "golden",
        {
            "turns": list(INFILL_TURNS),
            "num_channels": NUM_CHANNELS,
            "timestamp_text": True,
            "target_t0": INFILL_T0,
            "target_frames": INFILL_FRAMES,
        },
    )
    chunk = pre(
        "golden",
        {
            "turns": list(CHUNK_TURNS),
            "num_channels": NUM_CHANNELS,
            "timestamp_text": True,
            "target_t0": CHUNK_T0,
            "target_frames": CHUNK_FRAMES,
            "prompt_frames": CHUNK_PROMPT_FRAMES,
            "prev_frames": CHUNK_PREV_FRAMES,
        },
    )
    got = {
        "vocab": {
            "size": len(token2id),
            "turn": token2id[TURN_TOKEN],
            "other": token2id[OTHER_TOKEN],
            "turn_fill": token2id[TURN_FILL_TOKEN],
            "speaker_prompt": token2id[SPEAKER_PROMPT_TOKEN],
            "prev_chunk": token2id[PREV_CHUNK_TOKEN],
        },
        "infill": {
            "target_t0": INFILL_T0,
            "target_frames": INFILL_FRAMES,
            "branches": [_branch_payload(t.tolist(), token2id) for t in infill["text"]],
        },
        "chunk_full": {
            "target_t0": CHUNK_T0,
            "target_frames": CHUNK_FRAMES,
            "cond_frames": CHUNK_PROMPT_FRAMES + CHUNK_PREV_FRAMES,
            "prompt_frames": CHUNK_PROMPT_FRAMES,
            "prev_frames": CHUNK_PREV_FRAMES,
            "branches": [_branch_payload(t.tolist(), token2id) for t in chunk["text"]],
        },
    }
    expected = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert got == expected
