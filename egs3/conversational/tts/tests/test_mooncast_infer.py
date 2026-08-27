"""Tests for ``local/mooncast_infer.py``, the MoonCast runner.

MoonCast ships no batch inference program, so this driver is ours - and so
are the three things a driver decides: how rows are seeded, how the audio is
taken out losslessly, and what is recorded about each generation.
Everything here is the model-free part of that.

The fake module reproduces the shape of their ``inference.py``: a
``detokenize`` free function called once per turn, and a ``torchaudio``
module whose ``save`` is the only place the finished concatenation exists as
a tensor.  That is what makes the interception genuinely tested rather than
mocked away.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "mooncast_infer",
    Path(__file__).resolve().parents[1] / "local" / "mooncast_infer.py",
)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


class FakeAudio:
    """Stand-in for the waveform tensor their vocoder returns."""

    def __init__(self, samples: int):
        self.samples = samples

    @property
    def shape(self):
        return (1, self.samples)


class FakeTorchaudio:
    """The module their loop reaches for when it encodes to mp3."""

    def __init__(self):
        self.saved: list = []

    def save(self, target, tensor, **kwargs):
        self.saved.append((target, tensor, kwargs))


class FakeModule:
    """Their ``inference`` module: a ``detokenize`` and a ``torchaudio``."""

    def __init__(self, samples_per_turn=48000, fail_on=None):
        self.samples_per_turn = samples_per_turn
        self.fail_on = fail_on
        self.torchaudio = FakeTorchaudio()
        self.detokenize = self._detokenize

    def _detokenize(self, detokenizer, tokens, ref_wav, ref_tokens):
        return FakeAudio(self.samples_per_turn)


class FakeModel:
    """Their ``Model.inference`` shape: detokenize per turn, then concat."""

    def __init__(self, module: FakeModule):
        self.module = module
        self.seen: list[dict] = []

    def inference(self, js):
        self.seen.append(js)
        # Their _process_text mutates the dict it is handed.
        for turn in js["dialogue"]:
            turn["bpe_ids"] = [1, 2, 3]
        total = 0
        for turn in js["dialogue"]:
            if self.module.fail_on is not None and self.module.fail_on in turn["text"]:
                raise ValueError("context overrun")
            total += self.module.detokenize(None, None, None, None).shape[-1]
        self.module.torchaudio.save(object(), FakeAudio(total), format="mp3")


ROW = {
    "window_id": "d2",
    "role_mapping": {
        "0": {"ref_audio": "/tmp/a.wav", "ref_text": "abc"},
        "1": {"ref_audio": "/tmp/b.wav", "ref_text": "de"},
    },
    "dialogue": [
        {"role": "0", "text": "abc"},
        {"role": "1", "text": "def"},
        {"role": "0", "text": "gab"},
    ],
}


def _pair(**kwargs):
    module = FakeModule(**kwargs)
    return module, FakeModel(module)


class TestSeeding:
    def test_seed_is_a_function_of_the_id_alone(self):
        # The point of per-row seeding: a re-shard or a single-row retry can
        # reproduce, which a per-process seed cannot give.
        assert mod.row_seed("d2") == mod.row_seed("d2")
        assert mod.row_seed("d2") != mod.row_seed("d3")

    def test_seed_is_stable_across_processes(self):
        # Python's builtin hash() is randomized per interpreter; a seed
        # built on it would silently differ between shards.
        assert mod.row_seed("Group0046_S009_0-019") == 992294967

    def test_seed_matches_the_firered_runner(self):
        # Same construction, deliberately: the two arms' seeds are then
        # comparable row by row.
        spec = importlib.util.spec_from_file_location(
            "firered_infer",
            Path(__file__).resolve().parents[1] / "local" / "firered_infer.py",
        )
        firered = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(firered)
        assert mod.row_seed("d2") == firered.row_seed("d2")

    def test_seed_is_in_range(self):
        for wid in ("d0", "d1", "x" * 200, ""):
            assert 0 <= mod.row_seed(wid) < 2**31


class TestSharding:
    def test_shards_are_contiguous_and_disjoint(self):
        rows = [{"window_id": f"d{i}"} for i in range(7)]
        shards = [mod.select_shard(rows, 3, i) for i in range(3)]
        assert [len(s) for s in shards] == [3, 3, 1]
        seen = [row["window_id"] for shard in shards for row in shard]
        assert seen == [f"d{i}" for i in range(7)]

    def test_a_bad_shard_index_is_an_error(self):
        with pytest.raises(ValueError):
            mod.select_shard([], 2, 2)


class TestSynthesize:
    def test_turn_boundaries_tile_the_output_without_gaps(self):
        # Their concatenation inserts nothing between turns, which is the
        # architectural claim this arm has to caption.
        module, model = _pair()
        audio, turns = mod.synthesize(module, model, ROW, 7, seeder=lambda s: None)
        assert len(turns) == 3
        assert [turn["role"] for turn in turns] == ["0", "1", "0"]
        assert turns[0]["start"] == 0.0
        for previous, following in zip(turns, turns[1:]):
            assert previous["end"] == following["start"]
        assert turns[-1]["end"] * mod.SAMPLE_RATE == audio.shape[-1]

    def test_frames_are_counted_at_fifty_hertz(self):
        module, model = _pair(samples_per_turn=mod.SAMPLES_PER_FRAME * 25)
        _, turns = mod.synthesize(module, model, ROW, 7, seeder=lambda s: None)
        assert [turn["frames"] for turn in turns] == [25, 25, 25]

    def test_the_row_is_seeded_before_generation(self):
        seen: list[int] = []
        module, model = _pair()
        mod.synthesize(module, model, ROW, 1234, seeder=seen.append)
        assert seen == [1234]

    def test_their_dict_is_not_mutated(self):
        # Their ``_process_text`` writes ``bpe_ids`` into every turn; the
        # row we record must stay the row we read.
        module, model = _pair()
        mod.synthesize(module, model, ROW, 7, seeder=lambda s: None)
        assert all("bpe_ids" not in turn for turn in ROW["dialogue"])
        assert "bpe_ids" in model.seen[0]["dialogue"][0]

    def test_the_captured_tensor_is_the_one_they_would_encode(self):
        module, model = _pair()
        audio, _ = mod.synthesize(module, model, ROW, 7, seeder=lambda s: None)
        assert module.torchaudio.saved[0][1] is audio

    def test_their_mp3_encode_still_runs(self):
        # The proxy delegates, so their published path is unchanged and only
        # its result is discarded.
        module, model = _pair()
        mod.synthesize(module, model, ROW, 7, seeder=lambda s: None)
        assert module.torchaudio.saved[0][2]["format"] == "mp3"

    def test_the_interception_is_removed_afterwards(self):
        module, model = _pair()
        detokenize = module.detokenize
        torchaudio = module.torchaudio
        mod.synthesize(module, model, ROW, 7, seeder=lambda s: None)
        assert module.detokenize is detokenize
        assert module.torchaudio is torchaudio

    def test_the_interception_is_removed_even_when_generation_raises(self):
        module, model = _pair(fail_on="def")
        detokenize = module.detokenize
        torchaudio = module.torchaudio
        with pytest.raises(ValueError):
            mod.synthesize(module, model, ROW, 7, seeder=lambda s: None)
        assert module.detokenize is detokenize
        assert module.torchaudio is torchaudio

    def test_a_turn_count_mismatch_is_an_error(self):
        module, model = _pair()
        row = dict(ROW, dialogue=ROW["dialogue"][:2])

        def inference(js):
            for _ in range(3):
                module.detokenize(None, None, None, None)
            module.torchaudio.save(object(), FakeAudio(3), format="mp3")

        model.inference = inference
        with pytest.raises(ValueError, match="turns detokenized"):
            mod.synthesize(module, model, row, 7, seeder=lambda s: None)


class TestLastLogitOnly:
    """Exercised against a REAL ``nn.Module``.

    An earlier version of this patch replaced ``lm_head`` with a plain
    function and passed a hand-rolled fake, but ``nn.Module.__setattr__``
    rejects a function where a child module is registered - so the fake
    hid a TypeError that only appeared after a 24-minute model load on a
    GPU.  These tests use the real thing.
    """

    @staticmethod
    def _lm():
        torch = pytest.importorskip("torch")
        lm = torch.nn.Module()
        lm.lm_head = torch.nn.Linear(8, 5, bias=False)
        return torch, lm

    def test_a_multi_position_prefill_is_cut_to_its_last_row(self):
        # The whole point: sampling reads only the last position, so the
        # multi-GB [1, N, vocab] tensor never needs to exist.
        torch, lm = self._lm()
        hidden = torch.randn(1, 40, 8)
        before = lm.lm_head(hidden)
        mod.use_last_logit_only(lm)
        after = lm.lm_head(hidden)
        assert after.shape == (1, 1, 5)
        # and it is the SAME arithmetic on the same hidden state
        assert torch.equal(after[:, 0, :], before[:, -1, :])

    def test_a_single_position_step_is_passed_through_untouched(self):
        torch, lm = self._lm()
        hidden = torch.randn(1, 1, 8)
        before = lm.lm_head(hidden)
        mod.use_last_logit_only(lm)
        assert torch.equal(lm.lm_head(hidden), before)

    def test_applying_it_twice_registers_one_hook(self):
        torch, lm = self._lm()
        mod.use_last_logit_only(lm)
        mod.use_last_logit_only(lm)
        assert len(lm.lm_head._forward_pre_hooks) == 1
        assert lm.lm_head(torch.randn(1, 12, 8)).shape == (1, 1, 5)

    def test_the_head_is_still_a_module_afterwards(self):
        # nn.Module.__setattr__ rejects a plain function in a child-module
        # slot; the patch must not try.
        torch, lm = self._lm()
        mod.use_last_logit_only(lm)
        assert isinstance(lm.lm_head, torch.nn.Module)


class TestRun:
    def test_a_wav_and_a_record_are_written_per_row(self, tmp_path):
        module, model = _pair()
        saved: list[Path] = []
        report = mod.run(
            [ROW],
            tmp_path,
            module,
            model,
            save=lambda path, audio: saved.append(path),
            seeder=lambda s: None,
        )
        assert report == {"ok": 1, "failed": 0, "skipped": 0}
        assert [p.name for p in saved] == ["d2.wav"]
        record = json.loads((tmp_path / "records.jsonl").read_text().strip())
        assert record["window_id"] == "d2"
        assert record["status"] == "ok"
        assert record["seed"] == mod.row_seed("d2")
        assert record["num_turns_in"] == record["num_turns_generated"] == 3
        assert len(record["turns"]) == 3

    def test_a_failing_row_is_recorded_not_raised(self, tmp_path):
        # One bad row must not cost the other 279.
        module, model = _pair(fail_on="def")
        report = mod.run(
            [ROW],
            tmp_path,
            module,
            model,
            save=lambda path, audio: None,
            seeder=lambda s: None,
        )
        assert report == {"ok": 0, "failed": 1, "skipped": 0}
        record = json.loads((tmp_path / "records.jsonl").read_text().strip())
        assert record["status"] == "failed"
        assert "context overrun" in record["error"]
        assert "Traceback" in record["traceback"]

    def test_resume_skips_rows_whose_wav_is_already_on_disk(self, tmp_path):
        # A shard that ran out of walltime must not redo the rows it
        # finished: the queue wait costs far more than the generation.
        (tmp_path / "d2.wav").write_bytes(b"RIFF")
        module, model = _pair()
        report = mod.run(
            [ROW],
            tmp_path,
            module,
            model,
            save=lambda path, audio: None,
            seeder=lambda s: None,
            resume=True,
        )
        assert report == {"ok": 0, "failed": 0, "skipped": 1}
        assert model.seen == []
        assert not (tmp_path / "records.jsonl").read_text().strip()

    def test_resume_still_generates_a_row_with_no_wav(self, tmp_path):
        module, model = _pair()
        report = mod.run(
            [ROW],
            tmp_path,
            module,
            model,
            save=lambda path, audio: None,
            seeder=lambda s: None,
            resume=True,
        )
        assert report == {"ok": 1, "failed": 0, "skipped": 0}

    def test_without_resume_an_existing_wav_is_regenerated(self, tmp_path):
        (tmp_path / "d2.wav").write_bytes(b"RIFF")
        module, model = _pair()
        report = mod.run(
            [ROW],
            tmp_path,
            module,
            model,
            save=lambda path, audio: None,
            seeder=lambda s: None,
        )
        assert report == {"ok": 1, "failed": 0, "skipped": 0}

    def test_records_are_appended_so_a_retry_keeps_its_history(self, tmp_path):
        for _ in range(2):
            module, model = _pair()
            mod.run(
                [ROW],
                tmp_path,
                module,
                model,
                save=lambda path, audio: None,
                seeder=lambda s: None,
            )
        lines = (tmp_path / "records.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2
