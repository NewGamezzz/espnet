"""espnet-path generation CLI (Task 7): the engine-equivalence anchor
generator.

The eval battery generates most audio through a vLLM server
(``eval/generate_vllm.py``, Task 6), but the future fine-tuned TAC model can
only run through the real espnet decode path - so a handful of manifest
records are also generated here, directly through
``model.prepare_inference()`` / ``model.inference(...)``, to anchor
engine-to-engine comparability (does the espnet path reproduce what the vLLM
server produces, modulo sampling noise).

This module adapts the existing, proven ``scripts/gate_generate.py`` gate
(prompt-batch construction via the real
``SpeechLMJobTemplate.build_preprocessor().collate_fn``, then
``model.inference(...)`` with the SAME ``INFERENCE_CONFIG`` sampling
parameters) to (a) take records from an eval manifest
(``eval.manifest.load_manifest``: ``example_id``/``system``/``caption``)
filtered by an ``--ids`` CLI flag instead of reading one indexed sample from
a dialogues jsonl, and (b) write the Task 6 artifact triple
(``<id>.wav``/``<id>.txt``/``<id>.json``) so the eval battery reads records
from both engines identically.

Lazy-import discipline (binding constraint): this module must be importable,
and its CLI parser testable, with no torch/espnet installed at all - so
NOTHING heavy (``torch``, ``yaml``, ``soundfile``, ``numpy``,
``espnet2.speechlm...``, and ``scripts.load_bagpiper``, which itself pulls
``espnet2.speechlm.model``) is imported at module level. Only
``eval.manifest.load_manifest`` (pure I/O, no heavy deps - see
``eval/generate_vllm.py``'s precedent) is imported at the top.

Sampling parameters are copied verbatim from ``scripts/gate_generate.py``'s
``INFERENCE_CONFIG`` (text temperature 0.6/topk 20, audio temperature
0.8/topk 20/cfg 3.0, max_step 1024, num_hypo 1) and must never drift from
that source of truth.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from eval.manifest import load_manifest

HERE = os.path.dirname(os.path.abspath(__file__))
RECIPE_DIR = os.path.dirname(HERE)
SCRIPTS_DIR = os.path.join(RECIPE_DIR, "scripts")

DEFAULT_CKPT_DIR = os.path.join(
    RECIPE_DIR, "downloads", "bagpiper", "speechlm-qwen3-8b"
)
DEFAULT_CONFIG = os.path.join(RECIPE_DIR, "conf", "bagpiper_train_config.yaml")

# Verbatim copy of scripts/gate_generate.py's INFERENCE_CONFIG - the "must
# equal exactly" binding constraint applies to these values, so this is a
# literal copy, not a derived/re-imported reference (importing
# gate_generate would pull torch/espnet at module scope).
INFERENCE_CONFIG = {
    "num_hypo": 1,
    "text": {"temperature": 0.6, "topk": 20, "max_step": 1024, "cfg": 1},
    "audio": {"temperature": 0.8, "topk": 20, "max_step": 1024, "cfg": 3.0},
}

# SFT-schema modality -> preprocessor IO name (same mapping
# scripts/gate_teacher_forced.py uses); manifest entries only ever supply
# text (system + caption) as the generation prompt.
_IO_FOR_MODALITY = {"text": "text", "audio": "discrete_audio"}


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI arg parser. Pure argparse - no heavy imports - so this
    is testable on a box with no torch/espnet installed."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="path to a manifest JSON")
    ap.add_argument(
        "--out-dir", required=True, help="directory for per-record artifacts"
    )
    ap.add_argument(
        "--ids",
        required=True,
        help="comma-separated example_ids to generate (subset of the manifest)",
    )
    ap.add_argument(
        "--ckpt",
        default=os.environ.get("BAGPIPER_CKPT") or DEFAULT_CKPT_DIR,
        help="BagPiper safetensors shard directory",
    )
    ap.add_argument(
        "--config",
        default=os.environ.get("BAGPIPER_TRAIN_CONFIG") or DEFAULT_CONFIG,
        help="reconstructed BagPiper train config yaml",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    return ap


def parse_ids(ids_arg: str) -> list[str]:
    """Split a comma-separated ``--ids`` value, stripping whitespace and
    dropping empty segments."""
    return [part.strip() for part in ids_arg.split(",") if part.strip()]


def select_entries(entries: list[dict], ids: list[str]) -> list[dict]:
    """Return the manifest entries matching ``ids``, in ``ids`` order.

    Loud failure (no silent skips, matching ``eval.manifest``'s ethos): every
    requested id must exist in ``entries``, or a ``ValueError`` naming the
    missing ids is raised.
    """
    by_id = {entry["example_id"]: entry for entry in entries}
    missing = [example_id for example_id in ids if example_id not in by_id]
    if missing:
        raise ValueError(f"--ids not found in manifest: {missing}")
    return [by_id[example_id] for example_id in ids]


def build_dialogue(entry: dict) -> list:
    """Map a manifest entry's ``system``/``caption`` text to the
    preprocessor's dialogue form: prompt only (system + user), no assistant
    turn - ``model.inference``'s ``inference_segment`` appends the
    ``<|assistant|>`` token itself, completing the trained prompt format
    (see ``scripts/gate_generate.py``'s ``build_prompt_batch`` docstring)."""
    return [
        ["system", _IO_FOR_MODALITY["text"], entry["system"]],
        ["user", _IO_FOR_MODALITY["text"], entry["caption"]],
    ]


def build_prompt_batch(config: dict, entry: dict):
    """Collate the prompt-only (system + user) part of a manifest entry,
    following ``scripts/gate_generate.py::build_prompt_batch`` but taking
    text straight from the manifest entry instead of an indexed dialogues
    jsonl record."""
    from espnet2.speechlm.model.speechlm.speechlm_job import SpeechLMJobTemplate

    job = SpeechLMJobTemplate(config, is_train=False)
    preproc = job.build_preprocessor()

    dialogue = build_dialogue(entry)
    batch = preproc.collate_fn([((None, None, None), {"dialogue": dialogue})])
    return batch


def _write_json_atomic(path: Path, record: dict) -> None:
    """Write ``record`` to ``path`` atomically (write to ``<path>.tmp`` then
    ``os.replace``), same resume-marker contract as
    ``eval/generate_vllm.py``'s helper of the same name: the ``.json`` must
    never appear on disk half-written, since its mere existence is read as
    "this record is already handled."""
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    os.replace(tmp_path, path)


def _collate_messages(messages):
    """Split ``model.inference``'s output messages into concatenated text
    and concatenated audio (following ``scripts/gate_generate.py``'s
    per-segment save loop, but joining every segment into ONE artifact pair
    instead of one file per segment, per the Task 6 artifact contract).

    Returns ``(text, audio, sample_rate)`` where ``audio`` is a 1-D
    ``numpy`` array (``None`` if no audio segment was produced) and
    ``sample_rate`` is the last audio segment's rate (``None`` if none).
    """
    import numpy as np

    texts = []
    audio_segments = []
    sample_rate = None
    for role, modality, content in messages:
        if modality == "audio":
            audio, lengths, sr = content
            wav = audio[0, 0, : lengths[0]].float().cpu().numpy()
            audio_segments.append(wav)
            sample_rate = sr
        else:
            text = content[0] if isinstance(content, (list, tuple)) else content
            texts.append(str(text))

    combined_audio = None
    if audio_segments:
        combined_audio = (
            np.concatenate(audio_segments)
            if len(audio_segments) > 1
            else audio_segments[0]
        )
    return "".join(texts), combined_audio, sample_rate


def generate_one(entry: dict, model, device: str, config: dict, out_dir) -> str:
    """Generate one manifest record through the espnet decode path, always
    writing ``<example_id>.json`` last (the resume/completion marker);
    ``.txt``/``.wav`` are written first, only on success. Returns the
    outcome string (``"ok"``/``"no_audio"``/``"failed"``) - never raises, so
    one record's failure does not abort the batch.
    """
    import soundfile as sf
    import torch

    example_id = entry["example_id"]
    out_dir = Path(out_dir)
    json_path = out_dir / f"{example_id}.json"

    try:
        batch = build_prompt_batch(config, entry)
        batch.pop("keys", None)
        batch.pop("loss_masks", None)  # not used by inference
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

        with torch.no_grad():
            messages, _cache = model.inference(INFERENCE_CONFIG, **batch)

        text, audio, sample_rate = _collate_messages(messages)
        has_audio = audio is not None

        (out_dir / f"{example_id}.txt").write_text(text, encoding="utf-8")
        if has_audio:
            sf.write(str(out_dir / f"{example_id}.wav"), audio, sample_rate)

        record = {
            "example_id": example_id,
            "engine": "espnet",
            "has_audio": has_audio,
        }
        _write_json_atomic(json_path, record)
        return "ok" if has_audio else "no_audio"
    except Exception as exc:  # noqa: BLE001 - any failure must be captured, not crash
        record = {
            "example_id": example_id,
            "engine": "espnet",
            "has_audio": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_json_atomic(json_path, record)
        return "failed"


def _load_model(config_path: str, ckpt_path: str, device: str):
    """Load BagPiper and prepare it for inference, following
    ``scripts/gate_generate.py::main``'s load/prepare sequence exactly:
    build on CPU in bf16 (``load_bagpiper``), call ``prepare_inference()``,
    then move to the target device."""
    import torch

    sys.path.insert(0, SCRIPTS_DIR)
    from load_bagpiper import load_bagpiper  # noqa: E402

    model = load_bagpiper(config_path, ckpt_path, device="cpu", dtype=torch.bfloat16)
    model.prepare_inference()
    return model.to(device).eval()


def main(argv: list[str] | None = None) -> int:
    import torch
    import yaml

    args = build_parser().parse_args(argv)
    ids = parse_ids(args.ids)

    entries = load_manifest(args.manifest)
    selected = select_entries(entries, ids)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model = _load_model(args.config, args.ckpt, args.device)

    outcomes: dict[str, str] = {}
    for entry in selected:
        example_id = entry["example_id"]
        outcomes[example_id] = generate_one(entry, model, args.device, config, out_dir)
        print(f"{example_id}: {outcomes[example_id]}")

    counts = {
        outcome: sum(1 for v in outcomes.values() if v == outcome)
        for outcome in ("ok", "no_audio", "failed")
    }
    print(f"generation done: {counts} (total {len(outcomes)} of {len(ids)} requested)")
    return 1 if outcomes and all(v == "failed" for v in outcomes.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
