r"""Run the MoonCast baseline over our input jsonl.

MoonCast ships no batch inference program - ``inference.py`` is a demo whose
dialogues are hard-coded in ``__main__`` - so this driver is ours.  It runs
inside THEIR pixi environment and imports their module; nothing of theirs is
imported into the recipe.

    cd /work/hdd/bbjs/ttrachu/development/MoonCast
    .pixi/envs/default/bin/python <recipe>/local/mooncast_infer.py \
        --input  input_v2.00.jsonl \
        --out-dir results/v2 \
        --repo-dir /work/hdd/bbjs/ttrachu/development/MoonCast

Every resource path in their code is relative to the current working
directory (``resources/tokenizer/160k.model``, ``resources/audio_tokenizer/
stats.pt``, and ``sys.path.append(".")``), so the runner chdirs into their
repo root before importing anything.

Three things a driver decides, and upstream did not:

**Seeding.**  Nothing in their repo calls ``torch.manual_seed``, and their
``GenerationConfig`` samples (``temperature=0.8``, ``top_k=30``,
``top_p=0.8``).  Each row is therefore seeded from a stable hash of its own
``window_id``, so a re-shard or a single-row retry can reproduce - which a
per-process seed cannot give.  Unlike FireRedTTS-2 there are TWO stochastic
stages here, the autoregressive sampler and the detokenizer's flow-matching
ODE, and one seed per row covers both.  This is OUR addition and the
config's provenance block says so.

**Lossless audio.**  Their ``infer_with_prompt`` ends by encoding the
concatenation to mp3 and base64-ing the bytes, because that is what their
gradio demo wants.  The mp3 container is not part of the model, every prior
arm in this table was stored as wav, and routing this one through a lossy
codec would bias its UTMOS row specifically - so the runner intercepts the
``torchaudio`` module their loop reaches for, keeps the ``concat_wav``
tensor, and writes wav itself.  Their encode still runs and is discarded.

**What is recorded.**  Their loop detokenizes turn by turn and
concatenates, so the exact turn boundaries in the output wav are free to
record, and this runner records them by wrapping ``detokenize``.  The scored
artifact is still the published concatenation; the boundaries are provenance
- in particular they document that this system emits zero inter-speaker gap
and zero overlap by construction, which no reader should mistake for a
measured score.

``detokenize`` is wrapped under the name ``inference.detokenize``, which is
the name their module imported into its own namespace and the one their loop
actually calls; patching the source module would do nothing.  Its return is
pre-normalization, but the peak division that follows leaves lengths alone.

Failures are recorded, not raised.  Each row is an autoregressive generation
against an accumulating 50 Hz context that can overrun the model's position
limit, and one bad row must not cost the other 279.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

#: Their vocoder's output rate.
SAMPLE_RATE = 24000
#: ``PrefixStreamingFlowMatchingDetokenizer.frame_size`` - one semantic
#: token is 480 samples at 24 kHz, i.e. the token rate is 50 Hz.
SAMPLES_PER_FRAME = 480


def row_seed(window_id: str) -> int:
    """Return a seed that depends on the dialogue id and nothing else.

    Not ``hash()``: python randomizes that per interpreter, so a seed built
    on it would silently differ between shards.
    """
    digest = hashlib.blake2b(window_id.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % 2**31


def _torch_seed(seed: int) -> None:
    """Seed torch (imported lazily so the pure parts stay importable)."""
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_shard(rows: list[dict], num_shards: int, shard_index: int) -> list[dict]:
    """Return the ``shard_index``-th of ``num_shards`` contiguous slices."""
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError(f"bad shard {shard_index} of {num_shards}")
    size = -(-len(rows) // num_shards)
    return rows[shard_index * size : (shard_index + 1) * size]


class _SaveProxy:
    """Stand in for the ``torchaudio`` module inside their loop.

    Their ``torchaudio.save(buffer, concat_wav, ...)`` is the only place the
    finished concatenation exists as a tensor.  This proxy keeps that tensor
    and then delegates, so their mp3 encode still happens exactly as shipped
    and only its result is thrown away.
    """

    def __init__(self, module):
        self._module = module
        self.captured = None

    def __getattr__(self, name):
        return getattr(self._module, name)

    def save(self, target, tensor, *args, **kwargs):
        """Record ``tensor``, then hand the call to the real module."""
        self.captured = tensor
        return self._module.save(target, tensor, *args, **kwargs)


@contextmanager
def _observing(module, lengths: list[int]):
    """Record each detokenized turn's length and capture the concatenation.

    Yields the proxy standing in for their ``torchaudio``, whose
    ``captured`` attribute holds the finished waveform once their loop
    returns.
    """
    original_detokenize = module.detokenize
    original_torchaudio = module.torchaudio
    proxy = _SaveProxy(original_torchaudio)

    def wrapper(*args, **kwargs):
        audio = original_detokenize(*args, **kwargs)
        lengths.append(int(audio.shape[-1]))
        return audio

    module.detokenize = wrapper
    module.torchaudio = proxy
    try:
        yield proxy
    finally:
        module.detokenize = original_detokenize
        module.torchaudio = original_torchaudio


def synthesize(module, model, row: dict, seed: int, seeder=_torch_seed):
    """``(audio, turns)`` for one dialogue, turns in output order.

    ``turns`` carries ``role``, ``samples``, ``frames`` and the
    ``start``/``end`` of the turn within the concatenated waveform.

    The row is deep-copied because their ``_process_text`` mutates the dict
    it is handed, adding ``bpe_ids`` to every turn.
    """
    lengths: list[int] = []
    payload = copy.deepcopy(
        {"role_mapping": row["role_mapping"], "dialogue": row["dialogue"]}
    )
    seeder(seed)
    with _observing(module, lengths) as proxy:
        model.inference(payload)
        audio = proxy.captured
    if audio is None:
        raise ValueError("their loop never reached torchaudio.save")
    if len(lengths) != len(row["dialogue"]):
        raise ValueError(
            f"{len(lengths)} turns detokenized for {len(row['dialogue'])} in"
        )
    turns: list[dict] = []
    offset = 0
    for turn, samples in zip(row["dialogue"], lengths):
        turns.append(
            {
                "role": turn["role"],
                "samples": samples,
                "frames": int(round(samples / SAMPLES_PER_FRAME)),
                "start": round(offset / SAMPLE_RATE, 4),
                "end": round((offset + samples) / SAMPLE_RATE, 4),
            }
        )
        offset += samples
    return audio, turns


def _save_wav(path: Path, audio) -> None:
    """Write their waveform at their rate (imported lazily, as torch is)."""
    import torch
    import torchaudio

    audio = audio.detach().to(torch.float32).cpu()
    if not bool(torch.isfinite(audio).all()):
        # Their per-turn ``x / x.abs().max()`` is NaN for an all-zero turn.
        raise ValueError(f"{path.name}: waveform is not finite")
    torchaudio.save(str(path), audio, SAMPLE_RATE)


def run(
    rows: list[dict],
    out_dir: Path,
    module,
    model,
    save=_save_wav,
    seeder=_torch_seed,
) -> dict:
    """Generate every row into ``out_dir/<window_id>.wav``.

    Appends one record per row to ``out_dir/records.jsonl`` - appends, so a
    resumed shard keeps its history rather than erasing the evidence of the
    first attempt.  Returns ``{"ok", "failed"}``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"
    ok = failed = 0
    with records_path.open("a", encoding="utf-8") as records:
        for row in rows:
            wid = row["window_id"]
            seed = row_seed(wid)
            record = {
                "window_id": wid,
                "seed": seed,
                "num_turns_in": len(row["dialogue"]),
            }
            started = time.time()
            try:
                audio, turns = synthesize(module, model, row, seed, seeder=seeder)
                samples = sum(turn["samples"] for turn in turns)
                if int(audio.shape[-1]) != samples:
                    # Their output is the concatenation of the turns and
                    # nothing else; if that stops being true, every
                    # boundary we record is wrong.
                    raise ValueError(
                        f"{wid}: {audio.shape[-1]} samples, turns sum to {samples}"
                    )
                save(out_dir / f"{wid}.wav", audio)
                record.update(
                    status="ok",
                    num_turns_generated=len(turns),
                    duration_sec=round(samples / SAMPLE_RATE, 4),
                    turns=turns,
                )
                ok += 1
            except Exception as error:  # noqa: BLE001 - recorded, not raised
                record.update(
                    status="failed",
                    error=f"{type(error).__name__}: {error}",
                    traceback=traceback.format_exc(),
                )
                failed += 1
                print(f"[FAILED] {wid}: {error}", flush=True)
            record["wall_sec"] = round(time.time() - started, 3)
            records.write(json.dumps(record) + "\n")
            records.flush()
    return {"ok": ok, "failed": failed}


def load_model(repo_dir: Path):
    """``(module, model)`` - their ``inference`` module and a live ``Model``.

    Chdirs into their repo first: every checkpoint path their constructor
    reaches for is relative to the working directory.
    """
    repo_dir = Path(repo_dir).resolve()
    os.chdir(repo_dir)
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    import inference

    return inference, inference.Model()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--repo-dir", required=True, type=Path)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--limit", type=int, default=None, help="first N rows only, for smoke tests"
    )
    parser.add_argument(
        "--only",
        default=None,
        help="comma-separated window_ids, for smoke tests and retries",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    lines = args.input.resolve().read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    rows = select_shard(rows, args.num_shards, args.shard_index)
    if args.only:
        wanted = set(args.only.split(","))
        rows = [row for row in rows if row["window_id"] in wanted]
    if args.limit is not None:
        rows = rows[: args.limit]

    module, model = load_model(args.repo_dir)
    report = run(rows, out_dir, module, model)
    print(f"{report['ok']} ok, {report['failed']} failed -> {out_dir}")
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
