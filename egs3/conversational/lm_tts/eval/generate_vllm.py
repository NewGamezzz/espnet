"""vLLM batch generation client (Task 6): manifest entries -> a vLLM
OpenAI-compatible chat-completions server -> per-record wav/txt/json
artifacts.

The network path (``_post_json``, ``generate_one``, ``main``) is
**stdlib-only** - ``urllib.request``, ``base64``, ``json``,
``concurrent.futures`` - so this module runs under any python on the
cluster with no extra install; it deliberately imports nothing from
``eval.manifest`` beyond ``load_manifest`` (pure I/O, no heavy deps), and
never imports ``torch``/``numpy`` itself.

Request shape (``build_payload``) is EXACT, proven against the live
A/B-test server - see the Task 6 brief; do not "clean up" the field names
or the ``vllm_xargs`` nesting.

Resume-safety contract: ``needs_generation`` is existence-only - a record
whose ``<example_id>.json`` already exists is skipped on the next run,
**including** a previously-failed record (its json has an ``"error"``
field but still exists). Retrying a failure requires deleting its json
first; this is deliberate, not an oversight - see the brief's skip/retry
wording.

Per-record outcome is one of four strings used throughout this module and
``generation_summary.json``:

- ``"ok"``        - request succeeded and returned audio.
- ``"no_audio"``  - request succeeded but the response carried no audio
  (not treated as a failure).
- ``"failed"``    - the request raised, or the response could not be
  parsed into text/audio; the ``.json`` carries an ``"error"`` field.
- ``"skipped"``   - resumed past because the record's ``.json`` already
  existed.

The CLI exits non-zero iff every record's outcome is ``"failed"`` (an
empty manifest, or a manifest where every record is skipped/ok/no_audio,
exits 0).
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from eval.manifest import load_manifest

MODEL_NAME = "speechlm-qwen3-8b"
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"

OUTCOMES = ("ok", "no_audio", "failed", "skipped")


def build_payload(
    entry: dict,
    max_tokens: int = 12000,
    cfg: float = 3.0,
    audio_temperature: float = 0.8,
    text_temperature: float = 0.6,
    audio_topk: int = 20,
) -> dict:
    """Build the OpenAI-style chat-completion request body for ``entry``.

    EXACT shape proven against the live A/B run (Task 6 brief) - field
    names, nesting, and which knob maps to which key are all load-bearing;
    do not restructure.
    """
    return {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": entry["system"]},
            {"role": "user", "content": entry["caption"]},
        ],
        "max_tokens": max_tokens,
        "temperature": audio_temperature,
        "top_k": audio_topk,
        "vllm_xargs": {
            "mode": "text_audio",
            "phase": "text",
            "text_temperature": text_temperature,
            "audio_temperature": audio_temperature,
            "audio_topk": audio_topk,
            "cfg": cfg,
        },
    }


def needs_generation(entry: dict, out_dir: str | Path) -> bool:
    """``True`` iff ``<example_id>.json`` does not yet exist under
    ``out_dir`` - existence-only, so a previously-failed record's error
    json still counts as "handled" (see module docstring).
    """
    return not (Path(out_dir) / f"{entry['example_id']}.json").exists()


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    """POST ``payload`` as JSON to ``url`` and return the parsed JSON
    response. The only network call in this module; ``generate_one``
    takes this as an injectable ``post_fn`` so tests never hit a socket.
    """
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def generate_one(
    entry: dict,
    out_dir: str | Path,
    base_url: str,
    timeout: float,
    max_tokens: int = 12000,
    cfg: float = 3.0,
    audio_temperature: float = 0.8,
    text_temperature: float = 0.6,
    audio_topk: int = 20,
    post_fn: Callable[[str, dict, float], dict] = _post_json,
) -> str:
    """Generate one manifest record, always writing ``<example_id>.json``;
    ``.txt``/``.wav`` are written only on a parseable, non-erroring
    response. Returns the outcome string (``"ok"``/``"no_audio"``/
    ``"failed"``) - never raises.
    """
    example_id = entry["example_id"]
    out_dir = Path(out_dir)
    payload = build_payload(
        entry,
        max_tokens=max_tokens,
        cfg=cfg,
        audio_temperature=audio_temperature,
        text_temperature=text_temperature,
        audio_topk=audio_topk,
    )
    json_path = out_dir / f"{example_id}.json"

    start = time.monotonic()
    try:
        response = post_fn(base_url + CHAT_COMPLETIONS_PATH, payload, timeout)
        latency_s = time.monotonic() - start

        message = response["choices"][0]["message"]
        text = message.get("content", "")
        audio_field = message.get("audio") or {}
        audio_b64 = audio_field.get("data")
        has_audio = audio_b64 is not None

        (out_dir / f"{example_id}.txt").write_text(text, encoding="utf-8")
        if has_audio:
            (out_dir / f"{example_id}.wav").write_bytes(base64.b64decode(audio_b64))

        record = {
            "example_id": example_id,
            "finish_reason": response["choices"][0].get("finish_reason"),
            "usage": response.get("usage"),
            "latency_s": latency_s,
            "has_audio": has_audio,
        }
        json_path.write_text(json.dumps(record, indent=1), encoding="utf-8")
        return "ok" if has_audio else "no_audio"
    except Exception as exc:  # noqa: BLE001 - any failure must be captured, not crash
        latency_s = time.monotonic() - start
        record = {
            "example_id": example_id,
            "error": f"{type(exc).__name__}: {exc}",
            "latency_s": latency_s,
        }
        json_path.write_text(json.dumps(record, indent=1), encoding="utf-8")
        return "failed"


def summarize(outcomes: dict[str, str]) -> dict:
    """Pure aggregation of ``{example_id: outcome}`` into
    ``generation_summary.json``'s shape: per-outcome counts and id lists.
    """
    ids: dict[str, list[str]] = {outcome: [] for outcome in OUTCOMES}
    for example_id, outcome in outcomes.items():
        ids[outcome].append(example_id)
    return {
        "counts": {outcome: len(ids[outcome]) for outcome in OUTCOMES},
        "ids": ids,
    }


def all_records_failed(outcomes: dict[str, str]) -> bool:
    """``True`` iff ``outcomes`` is non-empty and every value is
    ``"failed"`` - the CLI's exit-code condition. An empty manifest, or
    one where every record is skipped/ok/no_audio, is not a failure.
    """
    return bool(outcomes) and all(outcome == "failed" for outcome in outcomes.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="path to a manifest JSON")
    parser.add_argument(
        "--out-dir", required=True, help="directory for per-record artifacts"
    )
    parser.add_argument(
        "--port", type=int, required=True, help="vLLM server port on localhost"
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--audio-temperature", type=float, default=0.8)
    parser.add_argument("--text-temperature", type=float, default=0.6)
    parser.add_argument("--audio-topk", type=int, default=20)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = load_manifest(args.manifest)
    base_url = f"http://localhost:{args.port}"

    outcomes: dict[str, str] = {}
    todo = []
    for entry in entries:
        if needs_generation(entry, out_dir):
            todo.append(entry)
        else:
            outcomes[entry["example_id"]] = "skipped"

    if todo:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(
                    generate_one,
                    entry,
                    out_dir,
                    base_url,
                    args.timeout,
                    args.max_tokens,
                    args.cfg,
                    args.audio_temperature,
                    args.text_temperature,
                    args.audio_topk,
                ): entry["example_id"]
                for entry in todo
            }
            for future in as_completed(futures):
                example_id = futures[future]
                outcomes[example_id] = future.result()

    summary = summarize(outcomes)
    (out_dir / "generation_summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8"
    )

    print(
        f"generation done: {summary['counts']} "
        f"(total {len(outcomes)} of {len(entries)} manifest entries)"
    )
    return 1 if all_records_failed(outcomes) else 0


if __name__ == "__main__":
    sys.exit(main())
