"""Tests for the vLLM batch generation client (Task 6).

Covers the PURE / dependency-injected functions only - `build_payload`
(exact request shape, proven against the live A/B run), `needs_generation`
(resume-skip logic against tmp files), `summarize` / `all_records_failed`
(pure aggregation over per-record outcomes), and `generate_one` driven with
an injected `post_fn` fake so no real HTTP request or server is ever
involved - plus an import-hygiene check that `eval.generate_vllm` never
pulls in `torch`/`numpy`, since the client must run under any stdlib-only
python on the cluster. The CLI wiring (`main`) and the real `_post_json`
urllib call are exercised only on Delta, per the Task 6 brief.
"""

from __future__ import annotations

import base64
import importlib
import json
import os
import sys
from pathlib import Path

import pytest

from eval.generate_vllm import (
    all_records_failed,
    build_payload,
    generate_one,
    needs_generation,
    summarize,
)

SYSTEM_PROMPT = "You are a multi-talker text-to-speech system."
CAPTION = "Two speakers narrate a short conversation."


def _entry(example_id: str = "ex1") -> dict:
    return {
        "example_id": example_id,
        "set": "sssd",
        "system": SYSTEM_PROMPT,
        "caption": CAPTION,
        "gt_wav": "/abs/gt.wav",
        "turns": [],
        "speakers": None,
        "ref_wavs": None,
    }


# ---------------------------------------------------------------------------
# build_payload: EXACT shape proven against the live A/B run (brief, verbatim)
# ---------------------------------------------------------------------------


class TestBuildPayload:
    def test_exact_shape_with_default_knobs(self):
        entry = _entry()
        payload = build_payload(entry)
        assert payload == {
            "model": "speechlm-qwen3-8b",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": CAPTION},
            ],
            "max_tokens": 12000,
            "temperature": 0.8,
            "top_k": 20,
            "vllm_xargs": {
                "mode": "text_audio",
                "phase": "text",
                "text_temperature": 0.6,
                "audio_temperature": 0.8,
                "audio_topk": 20,
                "cfg": 3.0,
            },
        }

    def test_custom_knobs_flow_through(self):
        entry = _entry()
        payload = build_payload(
            entry,
            max_tokens=500,
            cfg=1.5,
            audio_temperature=0.9,
            text_temperature=0.4,
            audio_topk=10,
        )
        assert payload["max_tokens"] == 500
        assert payload["temperature"] == 0.9
        assert payload["top_k"] == 10
        assert payload["vllm_xargs"] == {
            "mode": "text_audio",
            "phase": "text",
            "text_temperature": 0.4,
            "audio_temperature": 0.9,
            "audio_topk": 10,
            "cfg": 1.5,
        }

    def test_uses_entry_system_and_caption_verbatim(self):
        entry = _entry()
        entry["system"] = "custom system prompt"
        entry["caption"] = "custom caption text"
        payload = build_payload(entry)
        assert payload["messages"][0] == {
            "role": "system",
            "content": "custom system prompt",
        }
        assert payload["messages"][1] == {
            "role": "user",
            "content": "custom caption text",
        }


# ---------------------------------------------------------------------------
# needs_generation: resume-skip against files already on disk
# ---------------------------------------------------------------------------


class TestNeedsGeneration:
    def test_true_when_json_missing(self, tmp_path: Path):
        assert needs_generation(_entry("ex1"), tmp_path) is True

    def test_false_when_json_exists(self, tmp_path: Path):
        (tmp_path / "ex1.json").write_text("{}", encoding="utf-8")
        assert needs_generation(_entry("ex1"), tmp_path) is False

    def test_false_when_existing_json_has_error_field(self, tmp_path: Path):
        # Resume-skip is existence-only, by design: a previously-failed
        # record's error json still counts as "already handled" - retrying
        # a failure requires deleting its json first.
        (tmp_path / "ex1.json").write_text(
            json.dumps({"error": "boom"}), encoding="utf-8"
        )
        assert needs_generation(_entry("ex1"), tmp_path) is False

    def test_only_checks_matching_example_id(self, tmp_path: Path):
        (tmp_path / "ex2.json").write_text("{}", encoding="utf-8")
        assert needs_generation(_entry("ex1"), tmp_path) is True


# ---------------------------------------------------------------------------
# summarize / all_records_failed: pure aggregation over outcomes
# ---------------------------------------------------------------------------


class TestSummarize:
    def test_counts_and_ids_by_outcome(self):
        outcomes = {
            "a": "ok",
            "b": "ok",
            "c": "no_audio",
            "d": "failed",
            "e": "skipped",
        }
        summary = summarize(outcomes)
        assert summary["counts"] == {
            "ok": 2,
            "no_audio": 1,
            "failed": 1,
            "skipped": 1,
        }
        assert sorted(summary["ids"]["ok"]) == ["a", "b"]
        assert summary["ids"]["no_audio"] == ["c"]
        assert summary["ids"]["failed"] == ["d"]
        assert summary["ids"]["skipped"] == ["e"]

    def test_empty_outcomes_gives_zero_counts(self):
        summary = summarize({})
        assert summary["counts"] == {
            "ok": 0,
            "no_audio": 0,
            "failed": 0,
            "skipped": 0,
        }


class TestAllRecordsFailed:
    def test_true_when_every_record_failed(self):
        assert all_records_failed({"a": "failed", "b": "failed"}) is True

    def test_false_when_any_record_ok(self):
        assert all_records_failed({"a": "failed", "b": "ok"}) is False

    def test_false_when_any_record_skipped(self):
        assert all_records_failed({"a": "failed", "b": "skipped"}) is False

    def test_false_when_any_record_no_audio(self):
        assert all_records_failed({"a": "failed", "b": "no_audio"}) is False

    def test_false_when_empty(self):
        assert all_records_failed({}) is False


# ---------------------------------------------------------------------------
# generate_one: driven with an injected post_fn fake - no network, no server
# ---------------------------------------------------------------------------


def _fake_audio_b64() -> str:
    return base64.b64encode(b"RIFF....WAVEfmt ").decode("ascii")


class TestGenerateOne:
    def test_success_writes_wav_txt_json_and_returns_ok(self, tmp_path: Path):
        audio_b64 = _fake_audio_b64()

        def fake_post(url, payload, timeout):
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": "<think>plan</think>hello",
                            "audio": {"data": audio_b64},
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            }

        entry = _entry("ex1")
        outcome = generate_one(
            entry, tmp_path, "http://localhost:9999", 60, post_fn=fake_post
        )

        assert outcome == "ok"
        assert (tmp_path / "ex1.txt").read_text(encoding="utf-8") == (
            "<think>plan</think>hello"
        )
        assert (tmp_path / "ex1.wav").read_bytes() == base64.b64decode(audio_b64)
        record = json.loads((tmp_path / "ex1.json").read_text(encoding="utf-8"))
        assert record["finish_reason"] == "stop"
        assert record["usage"] == {"prompt_tokens": 10, "completion_tokens": 20}
        assert record["has_audio"] is True
        assert isinstance(record["latency_s"], float)

    def test_no_audio_writes_txt_and_json_but_no_wav(self, tmp_path: Path):
        def fake_post(url, payload, timeout):
            return {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": "no audio came back"},
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5},
            }

        entry = _entry("ex2")
        outcome = generate_one(
            entry, tmp_path, "http://localhost:9999", 60, post_fn=fake_post
        )

        assert outcome == "no_audio"
        assert (tmp_path / "ex2.txt").exists()
        assert not (tmp_path / "ex2.wav").exists()
        record = json.loads((tmp_path / "ex2.json").read_text(encoding="utf-8"))
        assert record["has_audio"] is False
        assert record["finish_reason"] == "length"

    def test_request_error_writes_error_json_and_returns_failed(
        self, tmp_path: Path
    ):
        def fake_post(url, payload, timeout):
            raise TimeoutError("server did not respond")

        entry = _entry("ex3")
        outcome = generate_one(
            entry, tmp_path, "http://localhost:9999", 60, post_fn=fake_post
        )

        assert outcome == "failed"
        assert not (tmp_path / "ex3.txt").exists()
        assert not (tmp_path / "ex3.wav").exists()
        record = json.loads((tmp_path / "ex3.json").read_text(encoding="utf-8"))
        assert "error" in record
        assert "server did not respond" in record["error"]

    def test_malformed_response_writes_error_json_and_returns_failed(
        self, tmp_path: Path
    ):
        def fake_post(url, payload, timeout):
            return {"choices": []}  # no message -> IndexError

        entry = _entry("ex4")
        outcome = generate_one(
            entry, tmp_path, "http://localhost:9999", 60, post_fn=fake_post
        )

        assert outcome == "failed"
        record = json.loads((tmp_path / "ex4.json").read_text(encoding="utf-8"))
        assert "error" in record

    def test_uses_custom_sampling_knobs_in_the_posted_payload(self, tmp_path: Path):
        captured = {}

        def fake_post(url, payload, timeout):
            captured["payload"] = payload
            captured["url"] = url
            captured["timeout"] = timeout
            return {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": "hi"}}
                ],
                "usage": {},
            }

        entry = _entry("ex5")
        generate_one(
            entry,
            tmp_path,
            "http://localhost:8123",
            42,
            max_tokens=999,
            cfg=2.0,
            audio_temperature=0.7,
            text_temperature=0.5,
            audio_topk=15,
            post_fn=fake_post,
        )

        assert captured["timeout"] == 42
        assert captured["url"] == "http://localhost:8123/v1/chat/completions"
        assert captured["payload"]["max_tokens"] == 999
        assert captured["payload"]["vllm_xargs"]["cfg"] == 2.0


# ---------------------------------------------------------------------------
# generate_one: explicit null content must not fail the record (review fix)
# ---------------------------------------------------------------------------


class TestNullContentHandling:
    def test_explicit_null_content_writes_empty_txt_not_failure(
        self, tmp_path: Path
    ):
        def fake_post(url, payload, timeout):
            return {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": None}}
                ],
                "usage": {},
            }

        entry = _entry("ex_null")
        outcome = generate_one(
            entry, tmp_path, "http://localhost:9999", 60, post_fn=fake_post
        )

        assert outcome == "no_audio"
        assert (tmp_path / "ex_null.txt").read_text(encoding="utf-8") == ""
        record = json.loads((tmp_path / "ex_null.json").read_text(encoding="utf-8"))
        assert record["has_audio"] is False
        assert "error" not in record


# ---------------------------------------------------------------------------
# generate_one: the .json resume marker is written atomically and LAST
# (review fix) - a wall-time kill mid-write must never leave a truncated
# json that permanently hides the record from resume.
# ---------------------------------------------------------------------------


class TestAtomicJsonWrite:
    def test_no_tmp_file_left_and_json_valid_after_success(self, tmp_path: Path):
        audio_b64 = _fake_audio_b64()

        def fake_post(url, payload, timeout):
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "hi", "audio": {"data": audio_b64}},
                    }
                ],
                "usage": {},
            }

        entry = _entry("ex_atomic_ok")
        generate_one(
            entry, tmp_path, "http://localhost:9999", 60, post_fn=fake_post
        )

        assert (tmp_path / "ex_atomic_ok.json").exists()
        assert not (tmp_path / "ex_atomic_ok.json.tmp").exists()
        json.loads((tmp_path / "ex_atomic_ok.json").read_text(encoding="utf-8"))

    def test_no_tmp_file_left_after_failure(self, tmp_path: Path):
        def fake_post(url, payload, timeout):
            raise TimeoutError("boom")

        entry = _entry("ex_atomic_fail")
        generate_one(
            entry, tmp_path, "http://localhost:9999", 60, post_fn=fake_post
        )

        assert (tmp_path / "ex_atomic_fail.json").exists()
        assert not (tmp_path / "ex_atomic_fail.json.tmp").exists()

    def test_json_committed_via_os_replace_only_after_txt_and_wav_exist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import eval.generate_vllm as gv

        example_id = "ex_order"
        audio_b64 = _fake_audio_b64()
        calls = []
        real_replace = os.replace

        def spy_replace(src, dst):
            calls.append(
                (
                    (tmp_path / f"{example_id}.txt").exists(),
                    (tmp_path / f"{example_id}.wav").exists(),
                )
            )
            return real_replace(src, dst)

        monkeypatch.setattr(gv.os, "replace", spy_replace)

        def fake_post(url, payload, timeout):
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "hi", "audio": {"data": audio_b64}},
                    }
                ],
                "usage": {},
            }

        entry = _entry(example_id)
        outcome = generate_one(
            entry, tmp_path, "http://localhost:9999", 60, post_fn=fake_post
        )

        assert outcome == "ok"
        # exactly one os.replace call (the json commit), and at the moment
        # it fired, .txt and .wav were already on disk.
        assert calls == [(True, True)]


# ---------------------------------------------------------------------------
# import hygiene: the network path must be stdlib-only (no torch/numpy)
# ---------------------------------------------------------------------------


def test_importing_eval_generate_vllm_does_not_load_heavy_deps():
    stale = [
        name
        for name in sys.modules
        if name == "eval.generate_vllm" or name.split(".")[0] in ("torch", "numpy")
    ]
    for name in stale:
        del sys.modules[name]

    importlib.import_module("eval.generate_vllm")

    loaded_heavy = [
        name for name in sys.modules if name.split(".")[0] in ("torch", "numpy")
    ]
    assert not loaded_heavy, f"heavy deps loaded on import: {loaded_heavy}"
