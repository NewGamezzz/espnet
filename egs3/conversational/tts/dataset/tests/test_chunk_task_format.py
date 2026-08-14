"""Golden format contract for special-token chunk-task samples (Task 11).

Pins the on-disk shape a future inference-side ``cond_format: special_tokens``
implementation must reproduce: for one fully specified synthetic session
(hard-coded turns/spans/fs/hop right here, not shared with any other test
module), the sample ``ConversationDataset.load_window`` +
``ConversationalTextPreprocessor`` produce for a hand-built ``WindowRecord`` +
``ChunkTaskPlan`` built through the REAL assembly path (Task 6's
``_assemble_chunk_task`` and Task 9's per-frame prefix), not reimplemented.

The golden JSON records ``prompt_frames``/``prev_frames``/``cond_frames`` and
the first/last 8 token ids of each branch's text stream - not the raw
waveform (``TestChunkTaskAssembly`` in ``test_dataset.py`` already pins
waveform-level correctness byte-for-byte). This test's job is the FORMAT: how
many conditioning frames land where, and which token ids open/close each
branch's stream.

Regeneration: this repo's other golden fixtures (``golden/generate_goldens.py``)
are a one-shot CLI kept for provenance of goldens frozen from a NOW-RETIRED
builder and never re-run against new code. This fixture is different - it
pins a format that legitimately evolves alongside the chunk-task assembly or
preprocessor prefix code - so it regenerates in place from a deliberate env
var instead of a separate script: set ``REGEN_CHUNK_TASK_FORMAT_GOLDEN=1``
and run this test once to (re)write ``golden/chunk_task_format.json``, then
unset the var and re-run to confirm the freshly written fixture is stable,
then commit the diff under a change that explains why the format moved.
"""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import pytest

from .conftest import write_flac

from egs3.conversational.tts.dataset.dataset import ConversationDataset
from egs3.conversational.tts.dataset.preprocessing.chunk_task import ChunkTaskPlan
from egs3.conversational.tts.dataset.preprocessing.sessions import (
    SessionRecord,
    write_session_manifest,
)
from egs3.conversational.tts.dataset.preprocessing.sssd import Turn
from egs3.conversational.tts.dataset.preprocessing.text import (
    OTHER_TOKEN,
    PREV_CHUNK_TOKEN,
    SPEAKER_PROMPT_TOKEN,
    TURN_TOKEN,
    extend_vocab,
    make_token2id,
)
from egs3.conversational.tts.dataset.preprocessing.windows import WindowRecord
from egs3.conversational.tts.dataset.preprocessor import ConversationalTextPreprocessor

GOLDEN_PATH = Path(__file__).resolve().parent / "golden" / "chunk_task_format.json"
REGEN_ENV_VAR = "REGEN_CHUNK_TASK_FORMAT_GOLDEN"

# ---------------------------------------------------------------------------
# One fully specified synthetic session. Every number below is the contract:
# changing any of them (and regenerating) changes what the golden pins.
# ---------------------------------------------------------------------------
SESSION_ID = "golden_chunk_session"
AUDIO_RELPATH = f"original/{SESSION_ID}_mixed.flac"
NUM_CHANNELS = 2
SRC_SR = 48000  # source FLAC rate
FS = 24000  # training rate (ConversationDataset.fs)
HOP = 256  # mel hop in fs-rate samples (ConversationDataset.hop)
SESSION_DURATION = 40.0

WINDOW_T0, WINDOW_T1 = 20.0, 28.0
WINDOW_TURNS = (
    Turn(0, "spk_a", "hello there", 20.5, 22.5),
    Turn(1, "spk_b", "how are you", 23.0, 25.5),
    Turn(0, "spk_a", "i am fine", 26.0, 27.5),
)

# 5.0 s of previous-chunk context immediately before the window.
PREV_SPAN = (15.0, WINDOW_T0)
# One 4.0 s prompt span per ORIGINAL channel, laid out in the session prefix
# well clear of both the prev span and the window (see ChunkTaskPlan's
# prompt_spans contract: one per channel, all equal length).
PROMPT_SPANS = ((1.0, 5.0), (6.0, 10.0))
CHUNK_TASK_PLAN = ChunkTaskPlan(
    kind="full", prev_span=PREV_SPAN, prompt_spans=PROMPT_SPANS
)


def _make_dataset(tmp_path: Path) -> ConversationDataset:
    """The synthetic session on disk, wrapped in a real ConversationDataset
    (used only for its ``load_window``/``fs``/``hop``/``dataset_root`` -
    ``__init__`` needs a plannable session, so the manifest is atomic: one
    window spanning the whole file, distinct from the hand-built window this
    test actually loads)."""
    write_flac(tmp_path / AUDIO_RELPATH, NUM_CHANNELS, SESSION_DURATION, sr=SRC_SR)
    session = SessionRecord(
        session_id=SESSION_ID,
        audio_relpath=AUDIO_RELPATH,
        num_channels=NUM_CHANNELS,
        sample_rate=SRC_SR,
        duration=SESSION_DURATION,
        turns=WINDOW_TURNS,
        atomic=True,
        window_id=f"{SESSION_ID}_w00000",
    )
    manifest = tmp_path / "sessions.jsonl"
    write_session_manifest(manifest, [session])
    return ConversationDataset(
        split="valid",
        manifest_path=manifest,
        dataset_root=tmp_path,
        fs=FS,
        hop=HOP,
        permute_channels=False,
    )


def _build_sample(tmp_path: Path, vocab_path: Path) -> dict:
    """Run the REAL path: hand-built WindowRecord + ChunkTaskPlan through
    ``ConversationDataset.load_window`` (Task 6/9 assembly), then through
    ``ConversationalTextPreprocessor`` (Task 9's per-frame prefix)."""
    ds = _make_dataset(tmp_path)
    ds._fixed_perm = list(range(NUM_CHANNELS))  # no permutation: deterministic rows
    record = WindowRecord(
        window_id=f"{SESSION_ID}_wgolden",
        session_id=SESSION_ID,
        audio_relpath=AUDIO_RELPATH,
        num_channels=NUM_CHANNELS,
        sample_rate=SRC_SR,
        t0=WINDOW_T0,
        t1=WINDOW_T1,
        turns=WINDOW_TURNS,
        chunk_task=CHUNK_TASK_PLAN,
    )
    raw = ds.load_window(record)
    preprocessor = ConversationalTextPreprocessor(token_list=vocab_path)
    return preprocessor(SESSION_ID, raw)


def _run_length_encode(ids: list[int]) -> list[list[int]]:
    """``[[value, run_length], ...]`` for consecutive equal ids."""
    return [[value, sum(1 for _ in group)] for value, group in itertools.groupby(ids)]


def _payload(sample: dict, token2id: dict[str, int]) -> dict:
    """The format contract: frame counts plus the shape of each branch's text
    stream (not the audio - see module docstring).

    ``first_8``/``last_8`` alone only sample the two ENDS of the stream: a
    preprocessor regression that mis-places the prompt/prev boundary while
    keeping ``prompt_frames``/``prev_frames`` totals unchanged (those two
    integers come from ``load_window``'s assembler dict, a DIFFERENT code
    path from the preprocessor's actual token stream - see
    ``ConversationalTextPreprocessor.__call__``) would still pass. Two
    additions close that gap, both read from the real token stream, not the
    assembler dict:

    - ``prefix_runs``: a run-length encoding of the WHOLE conditioning
      prefix (``ids[:cond_frames]``), so the entire composition - not just
      its ends - is pinned; for a well-formed sample this is exactly
      ``[[speaker_prompt_id, prompt_frames], [prev_chunk_id, prev_frames]]``
      (or just the first run for a prompt_only plan with ``prev_frames==0``),
      derived independently here rather than asserted as that shape, so a
      genuine regression shows up as a value mismatch instead of being
      baked into the golden by construction.
    - ``boundary_8``: an 8-token slice straddling the prompt/prev transition
      (``ids[prompt_frames - 4 : prompt_frames + 4]``), so an off-by-one at
      the exact boundary index is caught even if ``prefix_runs`` were
      somehow computed from the same wrong index (independent redundancy,
      "belt and braces").

    ``vocab`` anchors the raw ids above to what they MEAN: the tiny synthetic
    vocab's exact size and layout is an implementation detail of the
    ``base_vocab``/``extend_vocab`` fixtures, not part of the format contract
    itself, so a future reader can decode the ids without reconstructing that
    fixture (e.g. ``first_8`` entries equal to ``vocab.speaker_prompt``
    confirms the prompt-frame prefix comes first).
    """
    cond_frames = sample["cond_frames"]
    prompt_frames = sample["prompt_frames"]
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
    return {
        "prompt_frames": prompt_frames,
        "prev_frames": sample["prev_frames"],
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


@pytest.fixture
def vocab_tokens(base_vocab) -> list[str]:
    return extend_vocab(base_vocab)


@pytest.fixture
def vocab_path(tmp_path, vocab_tokens) -> Path:
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(vocab_tokens) + "\n", encoding="utf-8")
    return path


def test_chunk_task_format_matches_golden(tmp_path, vocab_path, vocab_tokens):
    sample = _build_sample(tmp_path, vocab_path)
    token2id = make_token2id(vocab_tokens)
    got = _payload(sample, token2id)

    if os.environ.get(REGEN_ENV_VAR) == "1":
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(
            json.dumps(got, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    assert GOLDEN_PATH.is_file(), (
        f"{GOLDEN_PATH} does not exist - regenerate once with "
        f"{REGEN_ENV_VAR}=1 pytest ... and commit the result"
    )
    expected = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert got == expected
