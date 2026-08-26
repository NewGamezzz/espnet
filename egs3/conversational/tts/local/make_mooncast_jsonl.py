r"""Emit a MoonCast input jsonl from OUR built set.

MoonCast runs in its own repo and environment (``jzq11111/mooncast``), so
the only thing this recipe hands it is an input file plus the runner in
``local/mooncast_infer.py``.  Building the input from OUR manifest rather
than by hand is what makes the comparison clean: the baseline sees the same
script segmentation and the same (v2-normalized, mono) prompt audio our
model saw, and the model is the only thing left different.

Row format, matching ``inference.Model.infer_with_prompt``::

    {"window_id": ...,
     "role_mapping": {"0": {"ref_audio": abs path, "ref_text": "..."},
                      "1": {"ref_audio": abs path, "ref_text": "..."}},
     "dialogue": [{"role": "0", "text": "..."}, ...]}

``window_id`` is ours; everything else is exactly the dict their
``Model.inference`` consumes.

Unlike the MOSS-TTSD converter (one merged tagged string) and the
FireRedTTS-2 converter (a list of tagged strings), MoonCast wants the
speaker tag moved OUT of the text and into a ``role`` field, so the tags are
stripped here.  Re-inserting them reproduces the other converters' text
byte for byte, which a test enforces - the script text is the same object
for every system in the table.

Two speakers, and only two
--------------------------
``infer_with_prompt`` indexes ``role_mapping["0"]`` and ``["1"]``
unconditionally, and routes any turn whose role is not ``"0"`` to speaker 1
via an ``if role_id == "0" else`` fallback.  A third speaker would therefore
not be rejected upstream - it would be silently merged into speaker 1 - so
this converter raises instead.

Monologue rows
--------------
Both roles must exist, and both are injected into the conditioning context
(reference text at ``infer_with_prompt`` lines 125-126, reference audio
tokens at 136-137), so the second prompt of a 1-speaker row is not inert.
Those rows duplicate the S1 prompt into role 1: no turn is ever assigned to
role 1, and the extra context is the same voice the dialogue actually uses.
Their ids are written to ``<out>.mono_ids.txt`` so the decision is
auditable.

Sharding
--------
``--num-shards N`` splits the input into N contiguous files, one per
single-GPU job (a 4-GPU MOSS-TTSD job queued 32 hours on Delta while four
1-GPU jobs started within the hour).  The result does not depend on the
split: ``mooncast_infer.py`` seeds per row, not per process.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: ``[S1]`` -> role ``"0"``, ``[S2]`` -> role ``"1"``.  Their vocabulary
#: stops here: ``Model.__init__`` encodes the literal strings "0" and "1".
TAG_TO_ROLE = {"[S1]": "0", "[S2]": "1"}


def strip_tag(text: str) -> tuple[str, str]:
    """Split ``"[S1] hello"`` into the role ``"0"`` and the text ``"hello"``.

    Raises:
        ValueError: if the leading four characters are not a tag they can
            represent.
    """
    tag = text[:4]
    if tag not in TAG_TO_ROLE:
        raise ValueError(f"{tag!r} is not one of {tuple(TAG_TO_ROLE)}")
    return TAG_TO_ROLE[tag], text[4:].strip()


def build_rows(set_dir: Path) -> tuple[list[dict], list[str]]:
    """``(rows, monologue_ids)`` for every record of the built set."""
    lines = (set_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    rows: list[dict] = []
    mono_ids: list[str] = []
    for record in records:
        wid = record["window_id"]
        channels = record["channels"]
        if len(channels) > len(TAG_TO_ROLE):
            raise ValueError(
                f"{wid}: {len(channels)} speakers, but MoonCast has exactly two"
            )
        dialogue: list[dict] = []
        for turn in record["turns"]:
            role, text = strip_tag("[{}] {}".format(turn["speaker"], turn["text"]))
            dialogue.append({"role": role, "text": text})
        if not dialogue or dialogue[0]["role"] != "0":
            # Speakers are tagged positionally; a dialogue that does not
            # open on S1 would silently mis-assign every voice.
            raise ValueError(f"{wid}: dialogue does not start with [S1]")
        role_mapping: dict[str, dict] = {}
        for index, channel in enumerate(channels, start=1):
            wav = (set_dir / channel["prompt_wav"]).resolve()
            if not wav.is_file():
                raise FileNotFoundError(wav)
            tag = f"[S{index}]"
            prompt_text = channel["prompt_text"].strip()
            if not prompt_text.startswith(tag):
                prompt_text = f"{tag} {prompt_text}"
            role, ref_text = strip_tag(prompt_text)
            role_mapping[role] = {"ref_audio": str(wav), "ref_text": ref_text}
        if record["num_channels"] == 1:
            # Thanapat, 2026-08-26: duplicate rather than borrow a sibling
            # voice or drop the row - see the design note.
            mono_ids.append(wid)
            role_mapping["1"] = dict(role_mapping["0"])
        if set(role_mapping) != {"0", "1"}:
            raise ValueError(f"{wid}: roles are {sorted(role_mapping)}, need 0 and 1")
        rows.append(
            {"window_id": wid, "role_mapping": role_mapping, "dialogue": dialogue}
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
        f"({len(mono_ids)} monologue rows, S1 prompt duplicated into role 1)"
    )


if __name__ == "__main__":
    main()
