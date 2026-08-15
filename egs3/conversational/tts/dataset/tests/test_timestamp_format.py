"""Golden format contract for Mode T (timestamp-aligned text) samples (Task 7).

Pins the on-disk shape a future inference-side ``text_format: timestamps``
implementation must reproduce: for one fully specified synthetic session
(hard-coded turns/spans/fs/hop right here, not shared with any other test
module), the sample ``ConversationDataset.load_window`` +
``ConversationalTextPreprocessor`` produce for two hand-built
``WindowRecord``s - an ordinary infill Mode T window and a chunk-task "full"
Mode T window - built through the REAL assembly path (Task 5's
``load_window`` target-key derivation and Task 6's
``build_branch_texts_timestamped`` routing), not reimplemented.

The golden JSON records, per sample, ``target_frames`` and (for the chunk
sample) ``cond_frames``/``prompt_frames``/``prev_frames``, plus per branch:
the text stream's ``length``, ``first_8``/``last_8`` token ids, every
``<turn>`` token's index in the stream, and the ``<OTHER>``/``<turn_fill>``
counts. This is deliberately format, not waveform: ``TestTimestampMode`` in
``test_dataset.py`` and ``TestTimestampPreprocessor`` in
``test_preprocessor.py`` already pin key presence and per-call correctness;
this test's job is a single frozen snapshot of what the produced id
sequences actually look like, so a change to the format shows up as an
explicit diff.

Both windows use a source sample rate equal to the training rate (24 kHz, no
resampling) and turn boundaries chosen so every frame count below is an
EXACT integer (no half-open-interval rounding jitter) - unlike
``test_chunk_task_format.py``'s 48 kHz source, this fixture is meant to be
hand-verified against ``round((start - t0) * 93.75)`` (see module-level
docstring of ``preprocessing/text.py``), which a resampled source would only
approximate.

Every turn boundary is also kept off the frame grid's half-integer ties
(``_assert_off_half_frame``, asserted below at collection time): a tie
resolves by Python's banker's rounding (round-half-to-even), and a boundary
merely CLOSE to a tie (e.g. ``(8.6 - 5.0) * 93.75 == 337.49999999999994``,
6e-14 below its true 337.5) resolves by float-representation noise instead -
both are real behavior of ``turn_frame_spans``, but neither is what a reader
doing the hand-check with pen-and-paper real-number arithmetic would predict.
Keeping every boundary comfortably clear of a tie makes the golden pin the
FORMAT, not an arithmetic accident.

Regeneration: this repo's other golden fixtures (``golden/generate_goldens.py``)
are a one-shot CLI kept for provenance of goldens frozen from a NOW-RETIRED
builder and never re-run against new code. This fixture is different - it
pins a format that legitimately evolves alongside the Mode T assembly or
preprocessor code - so it regenerates in place from a deliberate env var
instead of a separate script: set ``REGEN_TIMESTAMP_FORMAT_GOLDEN=1`` and run
this test once to (re)write ``golden/timestamp_format.json``, then unset the
var and re-run to confirm the freshly written fixture is stable, then commit
the diff under a change that explains why the format moved.
"""

from __future__ import annotations

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
    TURN_FILL_TOKEN,
    TURN_TOKEN,
    extend_vocab,
    make_token2id,
)
from egs3.conversational.tts.dataset.preprocessing.windows import WindowRecord
from egs3.conversational.tts.dataset.preprocessor import ConversationalTextPreprocessor

GOLDEN_PATH = Path(__file__).resolve().parent / "golden" / "timestamp_format.json"
REGEN_ENV_VAR = "REGEN_TIMESTAMP_FORMAT_GOLDEN"

# ---------------------------------------------------------------------------
# One fully specified synthetic session. Every number below is the contract:
# changing any of them (and regenerating) changes what the golden pins.
# ---------------------------------------------------------------------------
SESSION_ID = "golden_timestamp_session"
AUDIO_RELPATH = f"original/{SESSION_ID}_mixed.flac"
NUM_CHANNELS = 2
FS = 24000  # training rate (ConversationDataset.fs)
HOP = 256  # mel hop in fs-rate samples (ConversationDataset.hop); fs/hop == 93.75
SRC_SR = FS  # source == training rate: NO resampling, so every sample/frame
# count below is exact (round(seconds * FS) with no resample-filter slop),
# which is what makes the hand-check in the module docstring possible.
SESSION_DURATION = 40.0

# --- Sample 1: ordinary infill Mode T window -------------------------------
# 8.0s * 93.75 fps == 750.0 exactly: target_frames == 750 with no rounding.
INFILL_T0, INFILL_T1 = 20.0, 28.0
INFILL_TURNS = (
    Turn(0, "spk_a", "hello there", 20.5, 22.5),
    Turn(1, "spk_b", "how are you", 23.0, 25.5),
    # 26.1, not 26.0: (26.0 - 20.0) * 93.75 == 562.5 exactly, a banker's-
    # rounding tie the hand-check below would resolve inconsistently with a
    # naive "round half up" reader (see _assert_off_half_frame).
    Turn(0, "spk_a", "i am fine", 26.1, 27.5),
)

# --- Sample 2: chunk-task "full" Mode T window ------------------------------
# 6.4s * 93.75 fps == 600.0 exactly: target_frames == 600 (cond excluded),
# matching load_window's target_frames = speech.shape[1] // hop - cond_frames.
CHUNK_T0, CHUNK_T1 = 5.0, 11.4
CHUNK_TURNS = (
    # 5.5/7.1/8.5, not 5.4/7.0/8.6: those would land turn_frame_spans on or
    # within float noise of a rounding tie (7.0 -> exactly 187.5; 8.6 ->
    # 337.49999999999994, a 6e-14 float-representation error away from its
    # own tie) - see _assert_off_half_frame.
    Turn(0, "spk_a", "yes ok", 5.5, 6.8),
    Turn(1, "spk_b", "sure thing", 7.1, 8.5),
    Turn(0, "spk_a", "got it", 9.5, 10.9),
)
# 3.2s * FS == 76800 samples, an exact multiple of HOP (300 frames, no trim).
PREV_SPAN = (1.8, CHUNK_T0)
# 2.56s * FS == 61440 samples per span, an exact multiple of HOP (240 frames
# each channel, so min()-leveling in _assemble_chunk_task is a no-op) - well
# clear of PREV_SPAN and both target windows.
PROMPT_SPANS = ((30.0, 32.56), (33.0, 35.56))
CHUNK_TASK_PLAN = ChunkTaskPlan(
    kind="full", prev_span=PREV_SPAN, prompt_spans=PROMPT_SPANS
)


def _assert_off_half_frame(turns: tuple, t0: float, fps: float = 93.75) -> None:
    """Every turn boundary's frame position must sit clearly off a
    round()-tie, per the module docstring: within 0.1 frame of a half-integer
    is close enough that either banker's rounding or float-representation
    noise (see the module docstring) could flip which frame a naive
    ``int(x + 0.5)`` reader lands on, versus this fixture's actual
    ``round()``."""
    for turn in turns:
        for label, t in (("start", turn.start), ("end", turn.end)):
            v = (t - t0) * fps
            dist = abs((v % 1.0) - 0.5)
            assert dist > 0.1, (
                f"{turn.text!r} {label}={t} is only {dist:.4f} frame from a "
                "rounding tie - move it off the half-integer grid"
            )


_assert_off_half_frame(INFILL_TURNS, INFILL_T0)
_assert_off_half_frame(CHUNK_TURNS, CHUNK_T0)


def _make_dataset(tmp_path: Path) -> ConversationDataset:
    """The synthetic session on disk, wrapped in a real ConversationDataset
    (used only for its ``load_window``/``fs``/``hop``/``dataset_root`` -
    ``__init__`` needs a plannable session, so the manifest is atomic: one
    window spanning the whole file, distinct from the hand-built windows this
    test actually loads)."""
    write_flac(tmp_path / AUDIO_RELPATH, NUM_CHANNELS, SESSION_DURATION, sr=SRC_SR)
    session = SessionRecord(
        session_id=SESSION_ID,
        audio_relpath=AUDIO_RELPATH,
        num_channels=NUM_CHANNELS,
        sample_rate=SRC_SR,
        duration=SESSION_DURATION,
        turns=CHUNK_TURNS + INFILL_TURNS,
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


def _build_samples(tmp_path: Path, vocab_path: Path) -> tuple[dict, dict]:
    """Run the REAL path for both windows: hand-built WindowRecords through
    ``ConversationDataset.load_window`` (the Mode T target-key derivation
    and, for the chunk sample, the special-token conditioning plan's P/H
    assembly), then through ``ConversationalTextPreprocessor`` (Mode T
    routing and the P/H per-frame prefix)."""
    ds = _make_dataset(tmp_path)
    ds._fixed_perm = list(range(NUM_CHANNELS))  # no permutation: deterministic rows
    infill_record = WindowRecord(
        window_id=f"{SESSION_ID}_wgolden_infill",
        session_id=SESSION_ID,
        audio_relpath=AUDIO_RELPATH,
        num_channels=NUM_CHANNELS,
        sample_rate=SRC_SR,
        t0=INFILL_T0,
        t1=INFILL_T1,
        turns=INFILL_TURNS,
        timestamp_text=True,
    )
    chunk_record = WindowRecord(
        window_id=f"{SESSION_ID}_wgolden_chunk",
        session_id=SESSION_ID,
        audio_relpath=AUDIO_RELPATH,
        num_channels=NUM_CHANNELS,
        sample_rate=SRC_SR,
        t0=CHUNK_T0,
        t1=CHUNK_T1,
        turns=CHUNK_TURNS,
        chunk_task=CHUNK_TASK_PLAN,
        timestamp_text=True,
    )
    preprocessor = ConversationalTextPreprocessor(token_list=vocab_path)
    infill_raw = ds.load_window(infill_record)
    chunk_raw = ds.load_window(chunk_record)
    return preprocessor(SESSION_ID, infill_raw), preprocessor(SESSION_ID, chunk_raw)


def _branch_payload(ids: list[int], token2id: dict[str, int]) -> dict:
    """The format contract for one branch's text stream: length, the two
    ends, every ``<turn>`` token's index (so the frame position of each turn
    marker in the ACTUAL produced sequence is pinned, not just re-derived
    from ``turn_frame_spans``), and the ``<OTHER>``/``<turn_fill>`` counts
    that fill everything else."""
    return {
        "length": len(ids),
        "first_8": ids[:8],
        "last_8": ids[-8:],
        "turn_indices": [i for i, x in enumerate(ids) if x == token2id[TURN_TOKEN]],
        "other_count": sum(1 for x in ids if x == token2id[OTHER_TOKEN]),
        "turn_fill_count": sum(1 for x in ids if x == token2id[TURN_FILL_TOKEN]),
    }


def _payload(sample: dict, token2id: dict[str, int]) -> dict:
    out = {
        "target_t0": sample["target_t0"],
        "target_frames": sample["target_frames"],
        "branches": [t.tolist() for t in sample["text"]],
    }
    out["branches"] = [_branch_payload(ids, token2id) for ids in out["branches"]]
    if "cond_frames" in sample:
        out["cond_frames"] = sample["cond_frames"]
        out["prompt_frames"] = sample["prompt_frames"]
        out["prev_frames"] = sample["prev_frames"]
    return out


@pytest.fixture
def vocab_tokens(base_vocab) -> list[str]:
    return extend_vocab(base_vocab)


@pytest.fixture
def vocab_path(tmp_path, vocab_tokens) -> Path:
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(vocab_tokens) + "\n", encoding="utf-8")
    return path


def test_timestamp_format_matches_golden(tmp_path, vocab_path, vocab_tokens):
    infill_sample, chunk_sample = _build_samples(tmp_path, vocab_path)
    token2id = make_token2id(vocab_tokens)
    got = {
        "vocab": {
            "size": len(token2id),
            "turn": token2id[TURN_TOKEN],
            "other": token2id[OTHER_TOKEN],
            "turn_fill": token2id[TURN_FILL_TOKEN],
            "speaker_prompt": token2id[SPEAKER_PROMPT_TOKEN],
            "prev_chunk": token2id[PREV_CHUNK_TOKEN],
        },
        "infill": _payload(infill_sample, token2id),
        "chunk_full": _payload(chunk_sample, token2id),
    }

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
