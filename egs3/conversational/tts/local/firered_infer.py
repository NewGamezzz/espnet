r"""Run the FireRedTTS-2 baseline over our input jsonl.

FireRedTTS-2 (FireRedTeam/FireRedTTS2 at 404f3f6) ships no batch inference
program - only a Python API, a gradio demo and finetuning scripts - so this
driver is ours.  It runs inside THEIR pixi environment, from their repo
directory, and imports their package; nothing of theirs is imported into the
recipe.

    cd /work/hdd/bbjs/ttrachu/development/FireRedTTS2
    .pixi/envs/default/bin/python <recipe>/local/firered_infer.py \
        --input  input_v2.00.jsonl \
        --out-dir results/v2 \
        --pretrained-dir pretrained_models/FireRedTTS2

Two things a driver decides, and upstream did not:

**Seeding.**  Nothing in their repo calls ``torch.manual_seed``, and their
dialogue recipe samples (``temperature=0.9``, ``topk=30``).  Each row is
therefore seeded from a stable hash of its own ``window_id``, so a re-shard
or a single-row retry reproduces bit-identically - which a per-process seed
cannot give.  This is OUR addition and the config's provenance block says
so.

**What is recorded.**  Their ``generate_dialogue`` synthesizes turn by turn
and concatenates, so the exact turn boundaries in the output wav are free to
record, and this runner records them.  The scored artifact is still the
published concatenation; the boundaries are provenance - in particular they
document that this system emits zero inter-speaker gap and zero overlap by
construction, which no reader should mistake for a measured score.

The wrapper around ``generate`` is what keeps that honest: their
``generate_dialogue`` is called verbatim, and the per-turn audio is observed
on the way past rather than by re-implementing their loop here.

Failures are recorded, not raised.  Each row is an autoregressive generation
that can hit their context cap (``generate`` raises once the context passes
``max_seq_len - max_generation_len``), and one bad row must not cost the
other 279.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

#: Their codec's output rate (``generate_dialogue`` resamples 24000 -> 16000
#: for context management, so 24 kHz is what reaches the wav).
SAMPLE_RATE = 24000
#: One frame of their 12.5 Hz tokenizer.
SAMPLES_PER_FRAME = SAMPLE_RATE // 12.5
#: Their published dialogue knobs: the README example and gradio_demo.py
#: both use these.  The signature default (topk=20) loses to the two
#: documented call sites.
SAMPLING = {"temperature": 0.9, "topk": 30}


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


def select_shard(rows: list[dict], num_shards: int, shard_index: int) -> list[dict]:
    """Return the ``shard_index``-th of ``num_shards`` contiguous slices."""
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError(f"bad shard {shard_index} of {num_shards}")
    size = -(-len(rows) // num_shards)
    return rows[shard_index * size : (shard_index + 1) * size]


def _speaker_of(args: tuple, kwargs: dict) -> str:
    """Return their ``generate``'s ``speaker`` argument, however passed."""
    if "speaker" in kwargs:
        return kwargs["speaker"]
    return args[1] if len(args) > 1 else ""


@contextmanager
def _recording_turns(model, turns: list[dict]):
    """Observe each ``generate`` call without changing their loop."""
    original = model.generate

    def wrapper(*args, **kwargs):
        audio = original(*args, **kwargs)
        samples = int(audio.shape[-1])
        turns.append(
            {
                "speaker": _speaker_of(args, kwargs),
                "samples": samples,
                "frames": int(round(samples / SAMPLES_PER_FRAME)),
            }
        )
        return audio

    model.generate = wrapper
    try:
        yield
    finally:
        del model.generate


def synthesize(model, row: dict, seed: int, seeder=_torch_seed):
    """``(audio, turns)`` for one dialogue, turns in output order.

    ``turns`` carries ``speaker``, ``samples``, ``frames`` and the
    ``start``/``end`` of the turn within the concatenated waveform.
    """
    observed: list[dict] = []
    seeder(seed)
    with _recording_turns(model, observed):
        audio = model.generate_dialogue(
            text_list=row["text_list"],
            prompt_wav_list=row["prompt_wav_list"],
            prompt_text_list=row["prompt_text_list"],
            **SAMPLING,
        )
    offset = 0
    for turn in observed:
        turn["start"] = round(offset / SAMPLE_RATE, 4)
        offset += turn["samples"]
        turn["end"] = round(offset / SAMPLE_RATE, 4)
    return audio, observed


def _save_wav(path: Path, audio) -> None:
    """Write their waveform at their rate (imported lazily, as torch is)."""
    import torchaudio

    torchaudio.save(str(path), audio.cpu(), SAMPLE_RATE)


def run(
    rows: list[dict],
    out_dir: Path,
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
                "num_turns_in": len(row["text_list"]),
            }
            started = time.time()
            try:
                audio, turns = synthesize(model, row, seed, seeder=seeder)
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


def load_model(pretrained_dir: Path, device: str = "cuda"):
    """Their dialogue model, at their default precision.

    fp32 (``use_bf16=False``) is the class default and what every documented
    call site uses; their bf16 option would be a deviation to caveat.
    """
    from fireredtts2.fireredtts2 import FireRedTTS2

    return FireRedTTS2(
        pretrained_dir=str(pretrained_dir), gen_type="dialogue", device=device
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--pretrained-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--limit", type=int, default=None, help="first N rows only, for smoke tests"
    )
    args = parser.parse_args()

    lines = args.input.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    rows = select_shard(rows, args.num_shards, args.shard_index)
    if args.limit is not None:
        rows = rows[: args.limit]

    model = load_model(args.pretrained_dir, device=args.device)
    report = run(rows, args.out_dir, model)
    print(f"{report['ok']} ok, {report['failed']} failed -> {args.out_dir}")
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
