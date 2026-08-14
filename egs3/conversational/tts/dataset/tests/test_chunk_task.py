import pytest

from egs3.conversational.tts.dataset.preprocessing.chunk_task import (
    ChunkTaskParams,
    ChunkTaskPlan,
    plan_from_json,
    plan_to_json,
)
from egs3.conversational.tts.dataset.preprocessing.sssd import Turn
from egs3.conversational.tts.dataset.preprocessing.windows import (
    WindowRecord,
    from_json,
    to_json,
)


def _turns():
    return (
        Turn(channel=0, speaker="A", text="hi there", start=100.0, end=102.0),
        Turn(channel=1, speaker="B", text="hello", start=102.0, end=104.0),
    )


def _record(chunk_task=None):
    return WindowRecord(
        window_id="s1_w00003",
        session_id="s1",
        audio_relpath="a.flac",
        num_channels=2,
        sample_rate=48000,
        t0=100.0,
        t1=130.0,
        turns=_turns(),
        chunk_task=chunk_task,
    )


def test_plan_json_roundtrip():
    plan = ChunkTaskPlan(
        kind="full", prev_span=(94.0, 100.0), prompt_spans=((10.0, 14.5), (40.0, 44.5))
    )
    assert plan_from_json(plan_to_json(plan)) == plan


def test_prompt_only_has_no_prev_span():
    with pytest.raises(ValueError):
        ChunkTaskPlan(
            kind="prompt_only",
            prev_span=(94.0, 100.0),
            prompt_spans=((10.0, 14.5), (40.0, 44.5)),
        )
    with pytest.raises(ValueError):
        ChunkTaskPlan(
            kind="full", prev_span=None, prompt_spans=((10.0, 14.5), (40.0, 44.5))
        )


def test_prompt_spans_equal_length_enforced():
    with pytest.raises(ValueError):
        ChunkTaskPlan(
            kind="prompt_only",
            prev_span=None,
            prompt_spans=((10.0, 14.5), (40.0, 43.0)),
        )


def test_window_record_json_omits_absent_chunk_task():
    d = to_json(_record())
    assert "chunk_task" not in d
    assert from_json(d).chunk_task is None


def test_window_record_json_roundtrips_chunk_task():
    plan = ChunkTaskPlan(
        kind="prompt_only", prev_span=None, prompt_spans=((10.0, 14.5), (40.0, 44.5))
    )
    rec = _record(chunk_task=plan)
    d = to_json(rec)
    assert d["chunk_task"]["kind"] == "prompt_only"
    assert from_json(d).chunk_task == plan
