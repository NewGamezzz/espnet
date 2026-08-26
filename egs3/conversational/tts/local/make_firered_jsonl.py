r"""Emit a FireRedTTS-2 input jsonl from OUR built set.

FireRedTTS-2 runs in its own repo and environment (FireRedTeam/FireRedTTS2,
pinned at 404f3f6), so the only thing this recipe hands it is an input file
plus the runner in ``local/firered_infer.py``.  Building the input from OUR
manifest rather than by hand is what makes the comparison clean: the
baseline sees the same script segmentation and the same (v2-normalized,
mono) prompt audio our model saw, and the model is the only thing left
different.

Row format, matching ``FireRedTTS2.generate_dialogue``::

    {"window_id": ...,
     "text_list": ["[S1] ...", "[S2] ...", ...],
     "prompt_wav_list": [abs path, ...],
     "prompt_text_list": ["[S1] ...", "[S2] ..."]}

Unlike the MOSS-TTSD converter this emits a LIST of turns rather than one
merged string, because their pipeline synthesizes turn by turn.  The joined
list is nonetheless byte-identical to the merged string the ZipVoice and
MOSS-TTSD converters emit (enforced by a test), so the script text is the
same object for every system in the table.

The tag must occupy exactly the first four characters: both
``generate_dialogue`` and ``process_text_list`` read ``text[:4]`` and assert
membership in ``[S1]``..``[S4]``.  That also caps a dialogue at four
speakers.

Monologue rows
--------------
The 1-channel records get a one-element prompt list, which their format
supports natively - as MOSS-TTSD's did, and unlike ZipVoice-Dialog-Stereo
where a two-prompt format forced us to duplicate.  Their ids are written to
``<out>.mono_ids.txt`` so the decision is auditable.

Sharding
--------
``--num-shards N`` splits the input into N contiguous files, one per
single-GPU job (a 4-GPU MOSS-TTSD job queued 32 hours on Delta while four
1-GPU jobs started within the hour).  Contiguous rather than round-robin so
shard membership is reconstructible from the manifest order alone - though
unlike MOSS-TTSD, the result does not depend on it: ``firered_infer.py``
seeds per row, not per process.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: Their tag vocabulary (``fireredtts2.py::generate_dialogue``).
SPEAKER_TAGS = ("[S1]", "[S2]", "[S3]", "[S4]")


def turn_texts(record: dict) -> list[str]:
    """``["[S1] hello", "[S2] hi"]`` - the turns in conversation order.

    ``" ".join`` of this is byte-identical to the single string
    ``make_moss_ttsd_jsonl.dialogue_text`` and the ZipVoice converter emit.
    The duplication is deliberate: the converters share nothing else, and a
    shared helper would be an abstraction with one real user.
    """
    return [
        "[{}] {}".format(turn["speaker"], turn["text"].strip())
        for turn in record["turns"]
    ]


def build_rows(set_dir: Path) -> tuple[list[dict], list[str]]:
    """``(rows, monologue_ids)`` for every record of the built set."""
    lines = (set_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    rows: list[dict] = []
    mono_ids: list[str] = []
    for record in records:
        wid = record["window_id"]
        texts = turn_texts(record)
        if not texts or not texts[0].startswith("[S1]"):
            # Speakers are tagged positionally; a dialogue that does not
            # open on S1 would silently mis-assign every voice.
            raise ValueError(f"{wid}: dialogue text does not start with [S1]")
        channels = record["channels"]
        if len(channels) > len(SPEAKER_TAGS):
            raise ValueError(
                f"{wid}: {len(channels)} speakers, but their tags stop at [S4]"
            )
        wavs: list[str] = []
        prompt_texts: list[str] = []
        for index, channel in enumerate(channels, start=1):
            wav = (set_dir / channel["prompt_wav"]).resolve()
            if not wav.is_file():
                raise FileNotFoundError(wav)
            tag = f"[S{index}]"
            prompt_text = channel["prompt_text"].strip()
            if not prompt_text.startswith(tag):
                prompt_text = f"{tag} {prompt_text}"
            wavs.append(str(wav))
            prompt_texts.append(prompt_text)
        for text in texts + prompt_texts:
            if text[:4] not in SPEAKER_TAGS:
                raise ValueError(f"{wid}: {text[:4]!r} is not one of {SPEAKER_TAGS}")
        if record["num_channels"] == 1:
            mono_ids.append(wid)
        rows.append(
            {
                "window_id": wid,
                "text_list": texts,
                "prompt_wav_list": wavs,
                "prompt_text_list": prompt_texts,
            }
        )
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
