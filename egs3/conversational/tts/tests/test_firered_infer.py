"""Tests for ``local/firered_infer.py``, the FireRedTTS-2 runner.

FireRedTTS-2 ships no batch inference program, so this driver is ours - and
so are the two things a driver decides: how rows are seeded, and what is
recorded about each generation.  Everything here is the model-free part of
that: seeding, sharding, turn accounting, and failure handling.

The fake model reproduces the shape of their ``generate_dialogue``: it calls
``generate`` once per turn and concatenates.  That is what makes the
turn-boundary wrapper genuinely tested rather than mocked away.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "firered_infer",
    Path(__file__).resolve().parents[1] / "local" / "firered_infer.py",
)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


class FakeAudio:
    """Stand-in for the 1-D waveform tensor their codec returns."""

    def __init__(self, samples: int):
        self.samples = samples

    @property
    def shape(self):
        return (self.samples,)


class FakeModel:
    """Their ``generate_dialogue`` shape: one ``generate`` call per turn."""

    def __init__(self, samples_per_turn=1920, fail_on=None):
        self.samples_per_turn = samples_per_turn
        self.fail_on = fail_on
        self.calls: list[dict] = []

    def generate(self, text, speaker, context, **kwargs):
        if self.fail_on is not None and self.fail_on in text:
            raise ValueError("Inputs too long")
        return FakeAudio(self.samples_per_turn)

    def generate_dialogue(self, text_list, prompt_wav_list, prompt_text_list, **kw):
        self.calls.append({"text_list": list(text_list), "kwargs": kw})
        total = 0
        for text in text_list:
            total += self.generate(text[4:], text[:4], []).samples
        return FakeAudio(total)


ROW = {
    "window_id": "d2",
    "text_list": ["[S1] abc", "[S2] def", "[S1] gab"],
    "prompt_wav_list": ["/tmp/a.wav", "/tmp/b.wav"],
    "prompt_text_list": ["[S1] abc", "[S2] de"],
}


class TestSeeding:
    def test_seed_is_a_function_of_the_id_alone(self):
        # The point of per-row seeding: a re-shard or a single-row retry
        # reproduces bit-identically, which a per-process seed cannot give.
        assert mod.row_seed("d2") == mod.row_seed("d2")
        assert mod.row_seed("d2") != mod.row_seed("d3")

    def test_seed_is_stable_across_processes(self):
        # Python's builtin hash() is randomized per interpreter; a seed
        # built on it would silently differ between shards.
        assert mod.row_seed("Group0046_S009_0-019") == 992294967

    def test_seed_is_in_range(self):
        for wid in ("d0", "d1", "x" * 200, ""):
            assert 0 <= mod.row_seed(wid) < 2**31


class TestSharding:
    def test_shards_are_contiguous_and_disjoint(self):
        rows = [{"window_id": f"d{i}"} for i in range(7)]
        shards = [mod.select_shard(rows, 3, i) for i in range(3)]
        assert [len(s) for s in shards] == [3, 3, 1]
        assert [r for s in shards for r in s] == rows

    def test_one_shard_is_everything(self):
        rows = [{"window_id": "d0"}]
        assert mod.select_shard(rows, 1, 0) == rows

    def test_index_out_of_range_is_an_error(self):
        with pytest.raises(ValueError):
            mod.select_shard([{"window_id": "d0"}], 2, 2)


class TestSynthesize:
    def test_turn_boundaries_come_back_in_seconds(self):
        model = FakeModel(samples_per_turn=24000)
        audio, turns = mod.synthesize(model, ROW, seed=1, seeder=lambda s: None)
        assert audio.shape[0] == 72000
        assert [t["speaker"] for t in turns] == ["[S1]", "[S2]", "[S1]"]
        assert [(t["start"], t["end"]) for t in turns] == [
            (0.0, 1.0),
            (1.0, 2.0),
            (2.0, 3.0),
        ]

    def test_turn_frames_are_reported_for_the_runaway_check(self):
        # Their per-turn cap is max_audio_length_ms=30_000, i.e. 375 frames
        # of 80 ms.  A looping turn hides inside a plausible total, so the
        # flag has to sit on the per-turn counts.
        model = FakeModel(samples_per_turn=1920)
        _, turns = mod.synthesize(model, ROW, seed=1, seeder=lambda s: None)
        assert [t["frames"] for t in turns] == [1, 1, 1]

    def test_their_published_sampling_knobs_are_passed(self):
        model = FakeModel()
        mod.synthesize(model, ROW, seed=1, seeder=lambda s: None)
        assert model.calls[0]["kwargs"] == {"temperature": 0.9, "topk": 30}

    def test_the_row_is_seeded_before_generation(self):
        seen: list[int] = []
        model = FakeModel()
        mod.synthesize(model, ROW, seed=7, seeder=seen.append)
        assert seen == [7]

    def test_the_wrapper_is_removed_afterwards(self):
        # A leaked wrapper would make the next row's turn list grow.
        model = FakeModel()
        before = model.generate
        mod.synthesize(model, ROW, seed=1, seeder=lambda s: None)
        assert model.generate == before

    def test_the_wrapper_is_removed_even_when_generation_raises(self):
        model = FakeModel(fail_on="def")
        before = model.generate
        with pytest.raises(ValueError):
            mod.synthesize(model, ROW, seed=1, seeder=lambda s: None)
        assert model.generate == before


class TestRun:
    def _run(self, tmp_path, rows, model, **kw):
        saved: dict[str, object] = {}

        def save(path, audio):
            saved[Path(path).name] = audio
            Path(path).write_bytes(b"")

        report = mod.run(rows, tmp_path, model, save=save, seeder=lambda s: None, **kw)
        return report, saved

    def test_every_row_is_written_under_our_own_id(self, tmp_path):
        # No collector-side renaming: unlike MOSS-TTSD, the runner is ours,
        # so the ingest's wav_dir + suffix contract is satisfied at source.
        report, saved = self._run(tmp_path, [ROW], FakeModel())
        assert set(saved) == {"d2.wav"}
        assert report["ok"] == 1 and report["failed"] == 0

    def test_records_carry_the_accounting(self, tmp_path):
        self._run(tmp_path, [ROW], FakeModel(samples_per_turn=24000))
        record = json.loads((tmp_path / "records.jsonl").read_text().strip())
        assert record["window_id"] == "d2"
        assert record["status"] == "ok"
        assert record["num_turns_in"] == 3
        assert record["num_turns_generated"] == 3
        assert record["duration_sec"] == pytest.approx(3.0)
        assert record["seed"] == mod.row_seed("d2")
        assert record["wall_sec"] >= 0.0
        assert len(record["turns"]) == 3

    def test_a_failing_row_is_recorded_not_raised(self, tmp_path):
        # Each row is an autoregressive generation that can raise on their
        # context cap.  One bad row must not cost the other 279.
        rows = [ROW, {**ROW, "window_id": "d3"}]
        report, saved = self._run(tmp_path, rows, FakeModel(fail_on="def"))
        assert report == {"ok": 0, "failed": 2}
        assert saved == {}
        records = [
            json.loads(line)
            for line in (tmp_path / "records.jsonl").read_text().splitlines()
        ]
        assert [r["status"] for r in records] == ["failed", "failed"]
        assert "Inputs too long" in records[0]["error"]

    def test_turn_split_by_their_own_re_segmentation_is_visible(self, tmp_path):
        # process_text_list splits turns past 80 English words, so the
        # number of generations can exceed the number of turns we sent.
        class Splitting(FakeModel):
            def generate_dialogue(self, text_list, *a, **kw):
                expanded = [t for text in text_list for t in (text, text)]
                return super().generate_dialogue(expanded, *a, **kw)

        self._run(tmp_path, [ROW], Splitting())
        record = json.loads((tmp_path / "records.jsonl").read_text().strip())
        assert record["num_turns_in"] == 3
        assert record["num_turns_generated"] == 6

    def test_records_append_so_a_resumed_shard_keeps_its_history(self, tmp_path):
        self._run(tmp_path, [ROW], FakeModel())
        self._run(tmp_path, [{**ROW, "window_id": "d3"}], FakeModel())
        lines = (tmp_path / "records.jsonl").read_text().splitlines()
        assert [json.loads(line)["window_id"] for line in lines] == ["d2", "d3"]
