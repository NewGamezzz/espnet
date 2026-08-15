"""Format-identity contract for ``cond_format: special_tokens``.

``golden/chunk_task_format.json`` is a BYTE-IDENTICAL copy of the fixture
PR #39 committed on the training branch
(``dataset/tests/golden/chunk_task_format.json`` there); it pins the sample
format the training path produces for one fully specified synthetic
scenario.  This test rebuilds that scenario through the INFERENCE-side
pieces - Task 3's frame helpers for the conditioning geometry and the
ported preprocessor for the token stream - and asserts the identical
payload.  Failure means the inference format has drifted from training:
fix the code, NEVER regenerate the fixture here.
"""

from __future__ import annotations

import itertools
import json
import string
from pathlib import Path

from egs3.conversational.tts.dataset.preprocessing.sssd import Turn
from egs3.conversational.tts.dataset.preprocessing.text import (
    OTHER_TOKEN,
    PREV_CHUNK_TOKEN,
    SPEAKER_PROMPT_TOKEN,
    TURN_TOKEN,
    extend_vocab,
    make_token2id,
)
from egs3.conversational.tts.dataset.preprocessor import (
    ConversationalTextPreprocessor,
)
from egs3.conversational.tts.src.chunked_inference import (
    min_truncated_prompt_frames,
    tail_frames,
)

GOLDEN_PATH = Path(__file__).resolve().parent / "golden" / "chunk_task_format.json"

# The training fixture's synthetic scenario, restated in inference terms.
# Training side: prompt spans of 4.0 s each, prev span 5.0 s, window turns
# below, fs 24000, hop 256, no channel permutation.  Inference side: two
# reference prompts of 4.0 s and 6.0 s (min-truncation -> 4.0 s), ample
# generated audio with cond_prev_sec 5.0 (tail -> 5.0 s), chunk-k turns =
# the same turn list.
FS, HOP = 24000, 256
NUM_CHANNELS = 2
BASE_VOCAB = (
    ["<blank>", "<unk>", "<space>"]
    + list(string.ascii_lowercase)
    + [".", ",", "?", "!", "'", "<sos/eos>"]
)
CHUNK_TURNS = (
    Turn(0, "spk_a", "hello there", 20.5, 22.5),
    Turn(1, "spk_b", "how are you", 23.0, 25.5),
    Turn(0, "spk_a", "i am fine", 26.0, 27.5),
)
PROMPT_SAMPLES = [4 * FS, 6 * FS]  # reference prompts: 4.0 s and 6.0 s
GENERATED_SAMPLES = 30 * FS  # assembled generated audio so far
COND_PROMPT_SEC = 8.0  # default cap; min-truncation binds
COND_PREV_SEC = 5.0  # matches the fixture's 5.0 s prev span


def _run_length_encode(ids: list[int]) -> list[list[int]]:
    return [[v, sum(1 for _ in g)] for v, g in itertools.groupby(ids)]


def test_inference_format_matches_training_golden(tmp_path):
    vocab_tokens = extend_vocab(BASE_VOCAB)
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("\n".join(vocab_tokens) + "\n", encoding="utf-8")
    token2id = make_token2id(vocab_tokens)

    prompt_frames = min_truncated_prompt_frames(
        PROMPT_SAMPLES, COND_PROMPT_SEC, FS, HOP
    )
    prev_frames = tail_frames(GENERATED_SAMPLES, COND_PREV_SEC, FS, HOP)
    # Longhand geometry: the numbers the fixture pins, derived independently.
    assert prompt_frames == (4 * FS) // HOP == 375
    assert prev_frames == int(5.0 * FS) // HOP == 468

    # EXACTLY the sample dict Task 5's round loop feeds the preprocessor.
    pre = ConversationalTextPreprocessor(token_list=vocab_path)
    sample = pre(
        "golden_chunk_session",
        {
            "turns": list(CHUNK_TURNS),
            "num_channels": NUM_CHANNELS,
            "prompt_frames": prompt_frames,
            "prev_frames": prev_frames,
        },
    )

    cond_frames = prompt_frames + prev_frames
    branches = []
    for tokens in sample["text"]:
        ids = tokens.tolist()
        branches.append(
            {
                "length": len(ids),
                "first_8": ids[:8],
                "last_8": ids[-8:],
                "prefix_runs": _run_length_encode(ids[:cond_frames]),
                "boundary_8": ids[max(0, prompt_frames - 4) : prompt_frames + 4],
            }
        )
    got = {
        "prompt_frames": prompt_frames,
        "prev_frames": prev_frames,
        "cond_frames": cond_frames,
        "vocab": {
            "size": len(token2id),
            "turn": token2id[TURN_TOKEN],
            "other": token2id[OTHER_TOKEN],
            "speaker_prompt": token2id[SPEAKER_PROMPT_TOKEN],
            "prev_chunk": token2id[PREV_CHUNK_TOKEN],
        },
        "branches": branches,
    }
    expected = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert got == expected
