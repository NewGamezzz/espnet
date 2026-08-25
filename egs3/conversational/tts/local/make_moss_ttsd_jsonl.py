r"""Emit a MOSS-TTSD input jsonl from OUR built set.

MOSS-TTSD runs in its own repo and environment (OpenMOSS/MOSS-TTSD, pinned
at 04fe1d85), so the only thing this recipe hands it is an input file.
Building that file from OUR manifest rather than writing a script by hand is
what makes the comparison clean: the baseline sees the same script
segmentation and the same (v2-normalized, mono) prompt audio our model saw,
and the model is the only thing left different.

Row format (``generation_utils.py::_collect_speaker_fields``)::

    {"window_id": ..., "text": "[S1] ... [S2] ...",
     "prompt_audio_speaker1": ..., "prompt_text_speaker1": ...,
     "prompt_audio_speaker2": ..., "prompt_text_speaker2": ...}

``window_id`` is not one of their keys, which is exactly why it is used:
``_make_output_record`` copies unrecognised fields straight through into
``output.jsonl``, so our dialogue id survives the process boundary.  Their
own ``id`` is overwritten with the line number and cannot carry it.

We send no ``base_path`` and absolute prompt paths, so their
``_resolve_path`` has nothing to resolve.

Monologue rows
--------------
The 1-channel records get ``prompt_audio_speaker1`` / ``prompt_text_speaker1``
alone: their format supports 1 to 5 speakers, so unlike ZipVoice-Dialog-Stereo
there is no need to hand the model a prompt the transcript never uses.  Their
ids are written to ``<out>.mono_ids.txt`` so the decision is auditable.

Sharding
--------
Their ``torch.manual_seed(42)`` runs in ``main()`` only, and the multi-GPU
``mp.spawn`` workers never re-enter it - so a multi-GPU run is UNSEEDED.
``--num-shards N`` therefore splits the input into N contiguous files, each
of which is run as its own single-GPU job that seeds itself.  Contiguous
rather than round-robin so shard membership is reconstructible from the
manifest order alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def dialogue_text(record: dict) -> str:
    """``"[S1] hello [S2] hi"`` - the turns in conversation order.

    Byte-identical to the string ``make_zipvoice_baseline_tsv.py`` emits.
    The duplication is deliberate: the two converters share nothing else,
    and a shared helper would be an abstraction with one real user.
    """
    return " ".join(
        "[{}] {}".format(turn["speaker"], turn["text"].strip())
        for turn in record["turns"]
    )


def build_rows(set_dir: Path) -> tuple[list[dict], list[str]]:
    """``(rows, monologue_ids)`` for every record of the built set."""
    lines = (set_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    rows: list[dict] = []
    mono_ids: list[str] = []
    for record in records:
        wid = record["window_id"]
        text = dialogue_text(record)
        if not text.startswith("[S1]"):
            # Their loader tags speakers positionally; a dialogue that does
            # not open on S1 would silently mis-assign every voice.
            raise ValueError(f"{wid}: dialogue text does not start with [S1]")
        row: dict = {"window_id": wid, "text": text}
        for index, channel in enumerate(record["channels"], start=1):
            wav = (set_dir / channel["prompt_wav"]).resolve()
            if not wav.is_file():
                raise FileNotFoundError(wav)
            prompt_text = channel["prompt_text"].strip()
            tag = f"[S{index}]"
            if not prompt_text.startswith(tag):
                prompt_text = f"{tag} {prompt_text}"
            row[f"prompt_audio_speaker{index}"] = str(wav)
            row[f"prompt_text_speaker{index}"] = prompt_text
        if record["num_channels"] == 1:
            mono_ids.append(wid)
        rows.append(row)
    return rows, mono_ids


def write_shards(rows: list[dict], out: Path, num_shards: int) -> list[Path]:
    """Write ``rows`` as one jsonl, or as ``num_shards`` contiguous ones."""
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if num_shards == 1:
        out.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return [out]
    size = -(-len(rows) // num_shards)
    paths: list[Path] = []
    for shard in range(num_shards):
        chunk = rows[shard * size : (shard + 1) * size]
        path = out.with_suffix(f".{shard:02d}{out.suffix}")
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in chunk), encoding="utf-8"
        )
        paths.append(path)
    return paths


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--set-dir",
        required=True,
        type=Path,
        help="the built set, e.g. downloads/zipvoice-dialog-test-en-v2",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="split into N contiguous files, one per single-GPU job",
    )
    args = parser.parse_args()

    rows, mono_ids = build_rows(args.set_dir.resolve())
    paths = write_shards(rows, args.out, args.num_shards)
    Path(str(args.out) + ".mono_ids.txt").write_text(
        "\n".join(mono_ids) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {len(rows)} rows -> {', '.join(str(p) for p in paths)} "
        f"({len(mono_ids)} monologue rows, speaker 1 only)"
    )


if __name__ == "__main__":
    main()
