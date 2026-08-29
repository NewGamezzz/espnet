"""pyannote VAD backend (paper's activity source) and the tagged span cache."""

from __future__ import annotations

import json

import numpy as np
import pytest

from egs3.conversational.tts.src.metrics.pyannote_vad import (
    PyannoteVADSegmenter,
    _shim_pyannote_compat,
    _torch_load_full,
    spans_from_annotation,
)
from egs3.conversational.tts.src.metrics.turn_taking_judge import (
    TurnTakingJudge,
    TurnTakingJudgeMetric,
    label_rows,
)
from egs3.conversational.tts.tests.test_turn_taking_judge import (
    KeyedFakeVADBackend,
    _oracle_table,
    _unique_wav,
    _write_meta_scp,
    _write_window,
)

SR = 16000


class _Seg:
    def __init__(self, s, e):
        self.start, self.end = s, e


class _Timeline(list):
    def support(self):
        # merge overlapping/adjacent like pyannote's Timeline.support()
        out = []
        for seg in sorted(self, key=lambda x: x.start):
            if out and seg.start <= out[-1].end:
                out[-1] = _Seg(out[-1].start, max(out[-1].end, seg.end))
            else:
                out.append(_Seg(seg.start, seg.end))
        return _Timeline(out)


class _Annotation:
    def __init__(self, segs):
        self._segs = segs

    def get_timeline(self):
        return _Timeline([_Seg(s, e) for s, e in self._segs])


class FakePipeline:
    def __init__(self, segs):
        self.segs = segs
        self.calls = []

    def __call__(self, file):
        self.calls.append((tuple(file["waveform"].shape), file["sample_rate"]))
        return _Annotation(self.segs)


class TestBackend:
    def test_spans_from_annotation_merges_and_sorts(self):
        ann = _Annotation([(2.0, 3.0), (0.5, 1.0), (0.9, 1.4)])
        assert spans_from_annotation(ann) == [(0.5, 1.4), (2.0, 3.0)]

    def test_call_feeds_waveform_dict_and_returns_seconds(self):
        pipe = FakePipeline([(0.25, 1.0), (1.5, 2.0)])
        vad = PyannoteVADSegmenter(pipeline=pipe)
        out = vad(np.zeros(SR * 3, np.float32), SR)
        assert out == [(0.25, 1.0), (1.5, 2.0)]
        assert pipe.calls == [((1, SR * 3), SR)]

    def test_construction_is_offline_and_import_error_is_explicit(self, monkeypatch):
        vad = PyannoteVADSegmenter()  # no pyannote import here
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name.startswith("pyannote"):
                raise ImportError("no pyannote")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="pyannote.audio"):
            vad(np.zeros(SR, np.float32), SR)


class TestSpanCache:
    def test_tagged_run_caches_spans_and_reuses_them(self, tmp_path):
        test_dir = tmp_path / "infer" / "valid"
        dur = 4.0
        g0 = _unique_wav(1, int(dur * SR))
        g1 = _unique_wav(2, int(dur * SR))
        _write_window(test_dir, "w1", dur, [g0, g1], seed=99)
        _write_meta_scp(test_dir, ["w1"])
        pipe = FakePipeline([(0.2, 3.0)])
        vad = PyannoteVADSegmenter(pipeline=pipe)
        table = _oracle_table(label_rows("w1", [[(0.2, 3.0)], [(0.2, 3.0)]], dur))
        judge = TurnTakingJudge(encode_fn=lambda b: np.stack([table[0.24]] * len(b)))
        data = {"meta": test_dir / "meta.scp"}
        m = TurnTakingJudgeMetric(judge=judge, vad_backend=vad, tag="paper")
        m(data, "valid", tmp_path / "infer")
        d = tmp_path / "infer" / "valid" / "scoring" / "turn_taking_judge"
        cache = json.loads((d / "spans_paper" / "w1.json").read_text())
        assert cache == [[[0.2, 3.0]], [[0.2, 3.0]]]
        assert len(pipe.calls) == 2  # one pass per channel
        m(data, "valid", tmp_path / "infer")
        assert len(pipe.calls) == 2  # second run: spans from cache
        # untagged runs never write a span cache
        vad2 = KeyedFakeVADBackend()
        vad2.register(g0, [(0.2, 3.0)])
        vad2.register(g1, [(0.2, 3.0)])
        TurnTakingJudgeMetric(judge=judge, vad_backend=vad2)(data, "valid", tmp_path / "infer")
        assert not (d / "spans_").exists() and not (d / "spans").exists()


class TestCompatShims:
    def test_use_auth_token_is_translated_once(self, monkeypatch):
        import huggingface_hub

        seen = []

        def fake(repo_id, filename, token=None, **k):
            seen.append((repo_id, filename, token))
            return "path"

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)
        _shim_pyannote_compat()
        first = huggingface_hub.hf_hub_download
        _shim_pyannote_compat()  # idempotent
        assert huggingface_hub.hf_hub_download is first
        assert first("r", "f", use_auth_token="tok") == "path"
        assert seen == [("r", "f", "tok")]

    def test_torch_load_full_restores_default(self, tmp_path):
        import torch

        f = tmp_path / "x.pt"
        torch.save({"a": 1}, f)
        orig = torch.load
        with _torch_load_full():
            assert torch.load is not orig
            assert torch.load(f)["a"] == 1
            assert torch.load(f, weights_only=True)["a"] == 1  # explicit True overridden
        assert torch.load is orig
